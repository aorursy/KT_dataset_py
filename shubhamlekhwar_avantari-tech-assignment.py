# Ensure that any edits to libraries you make are reloaded here automatically, and also that any charts or images displayed are shown in this notebook.

%reload_ext autoreload

%autoreload 2

%matplotlib inline
!pip install lshashpy3
import pandas as pd

import pickle

import numpy as np

from fastai.vision import *

from fastai.callbacks.hooks import *



from fastai.vision.image import pil2tensor, Image

import matplotlib.pyplot as plt

from lshashpy3 import LSHash

from PIL import Image as pil_img

from tqdm import tqdm_notebook

pd.set_option('display.max_columns', 500)

path = Path('../input/similar-images-in-dataset/data/dataset/')
tfms = get_transforms(

    do_flip=False, 

    flip_vert=False, 

    max_rotate=0, 

    max_lighting=0, 

    max_zoom=1, 

    max_warp=0

)

data = (ImageList.from_folder(path)

        .random_split_by_pct(0.2)

        .label_from_folder()

        .transform(tfms=tfms, size=512)

        .databunch(bs=16))
print('Train dataset size: {0}'.format(len(data.train_ds.x)))

print('Test dataset size: {0}'.format(len(data.valid_ds.x)))
## Show sample data

data.show_batch(rows=3, figsize=(10,6), hide_axis=False)
## Creating the model

learn = create_cnn(data, models.resnet50, pretrained=True, metrics=accuracy)
## Finding Ideal learning late

learn.model_dir  ='/kaggle/output/'

learn.lr_find()

learn.recorder.plot()
## Fitting 5 epochs

learn.fit_one_cycle(5,1e-2)
## Saving stage 1

learn.save('stg1-rn34')
## Unfreeing layer and finding ideal learning rate

learn.unfreeze()

learn.lr_find()

learn.recorder.plot()
## Fitting 5 epochs

learn.fit_one_cycle(5, slice(1e-5, 1e-2/5))
## Saving model weights

learn.save('stg2-rn34')
# this is a hook (Reference: https://forums.fast.ai/t/how-to-find-similar-images-based-on-final-embedding-layer/16903/13)

# hooks are used for saving intermediate computations

class SaveFeatures():

    features=None

    def __init__(self, m): 

        self.hook = m.register_forward_hook(self.hook_fn)

        self.features = None

    def hook_fn(self, module, input, output): 

        out = output.detach().cpu().numpy()

        if isinstance(self.features, type(None)):

            self.features = out

        else:

            self.features = np.row_stack((self.features, out))

    def remove(self): 

        self.hook.remove()

        

sf = SaveFeatures(learn.model[1][5]) ## Output before the last FC layer
## By running this feature vectors would be saved in sf variable initated above

_= learn.get_preds(data.train_ds)

_= learn.get_preds(DatasetType.Valid)
img_path = [str(x) for x in (list(data.train_ds.items)+list(data.valid_ds.items))]

feature_dict = dict(zip(img_path,sf.features))
## Exporting as pickle



file = "feature_dict.p"

with open(file,mode='wb') as feature_f:

    pickle.dump(feature_dict,feature_f)
## Loading Feature dictionary



with open(file,mode='rb') as feature_f:

    feature_dict = pickle.load(feature_f)

# feature_dict = pickle.load(open(path/'feature_dict.p','rb'))
## Locality Sensitive Hashing

# params

k = 10 # hash size

L = 5  # number of tables

d = 512 # Dimension of Feature vector

lsh = LSHash(hash_size=k, input_dim=d, num_hashtables=L)



# LSH on all the images

for img_path, vec in tqdm_notebook(feature_dict.items()):

    lsh.index(vec.flatten(), extra_data=img_path)
## Exporting as pickle



lsh_file = "lsh.p"

with open(lsh_file,mode='wb') as lsh_f:

    pickle.dump(lsh,lsh_f)
# Loading Feature dictionary



with open(file,mode='rb') as feature_f:

    feature_dict = pickle.load(feature_f)

    

with open(lsh_file,mode='rb') as lsh_f:

    lsh = pickle.load(lsh_f)



def get_similar_item(idx, feature_dict, lsh_variable, n_items):

    response = lsh_variable.query(feature_dict[list(feature_dict.keys())[idx]].flatten(), 

                     num_results=n_items+2, distance_func='hamming')

    input_img=response[0][0][1]

    print("Input Image Path= ", input_img)

    response=response[1:]

    

    columns = 3

    rows = int(np.ceil(n_items+1/columns))

    

    fig=plt.figure(figsize=(2*rows, 3*rows))

    

    inp_img = pil_img.open(input_img)

    fig.add_subplot(rows, columns, 1)

    plt.imshow(inp_img)

    



    j=4

    for i in range(1, columns*rows +1):     

        if i<n_items+2:

            img = pil_img.open(response[i-1][0][1])

            fig.add_subplot(rows , columns, j)

            plt.imshow(img)

        j=j+1

    return plt.show()
get_similar_item(50, feature_dict, lsh,8)
get_similar_item(40, feature_dict, lsh,8)
get_similar_item(1000, feature_dict, lsh,8)
get_similar_item(3000, feature_dict, lsh,8)
IMAGE_PATH='../input/similar-images-in-dataset/data/dataset/2.jpg'

input_img = pil_img.open(IMAGE_PATH)
def image_to_vec(url_img, hook, learner):

    '''

    Function to convert image to vector

    '''

    print("Convert image to vec")

    _ = learner.predict(Image(pil2tensor(url_img, np.float32).div_(255)))

    vect = hook.features[-1]

    return vect
vect = image_to_vec(input_img, sf, learn)
def get_similar_item_input(idx, feature_dict, lsh_variable, n_items):

    response = lsh_variable.query(vect,  num_results=n_items+2, distance_func='hamming')

#     lsh.query(vect, num_results=n_items + 1, distance_func="hamming")

    input_img=response[0][0][1]

    print("Input Image Path= ", input_img)

    response=response[1:]

    

    columns = 3

    rows = int(np.ceil(n_items+1/columns))

    

    fig=plt.figure(figsize=(2*rows, 3*rows))

    

    inp_img = pil_img.open(input_img)

    fig.add_subplot(rows, columns, 1)

    plt.imshow(inp_img)

    



    j=4

    for i in range(1, columns*rows +1):     

        if i<n_items+2:

            img = pil_img.open(response[i-1][0][1])

            fig.add_subplot(rows , columns, j)

            plt.imshow(img)

        j=j+1

    return plt.show()
get_similar_item_input(vect, feature_dict, lsh,8)
from sklearn.cluster import KMeans
# get the feature vector into a list

feature_vect=list(feature_dict.values())
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=7, random_state=42).fit(feature_vect)

Y=kmeans.labels_
from sklearn.manifold import TSNE

import numpy as np

import glob, json, os



chart_data = []



# build the tsne model on the image vectors



print('*******building tsne model************')

model = TSNE(n_components=2,random_state=42, verbose=1)

np.set_printoptions(suppress=True)

tsne_model = model.fit_transform(feature_vect)
image_path=list(feature_dict.keys())
"""The variable tsne contains an array of unnormalized 2d points, corresponding to the embedding.

In the next cell, we normalize the embedding so that lies entirely in the range (0,1)."""



tx, ty = tsne_model[:,0], tsne_model[:,1]

tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))

ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))
width = 4000

height = 3000

max_dim = 100



full_image = pil_img.new('RGBA', (width, height))

for img, x, y in zip(image_path, tx, ty):

    tile = pil_img.open(img)

    rs = max(1, tile.width/max_dim, tile.height/max_dim)

    tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), pil_img.ANTIALIAS)

    full_image.paste(tile, (int((width-max_dim)*x), int((height-max_dim)*y)), mask=tile.convert('RGBA'))



plt.figure(figsize = (35,35))

plt.imshow(full_image)
import seaborn as sns
def scatter(x, colors):

    # We choose a color palette with seaborn.

    palette = np.array(sns.color_palette("hls", 9))



    # We create a scatter plot.

    f = plt.figure(figsize=(20, 20))

    ax = plt.subplot(aspect='equal')

    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=120, c=palette[colors.astype(np.int)], label=colors)

    #plt.xlim(-25, 25)

    #plt.ylim(-25, 25)

    ax.axis('off')

    ax.axis('tight')

    return f, ax, sc
print(list(range(0,7)))

sns.palplot(np.array(sns.color_palette("hls",9)))

scatter(tsne_model, Y)