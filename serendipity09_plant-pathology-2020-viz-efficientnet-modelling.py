import os

import gc

import re



import glob

import cv2

import math

import numpy as np

import scipy as sp

import pandas as pd

import hashlib

from PIL import Image



import tensorflow as tf

from IPython.display import SVG

#import efficientnet.tfkeras as efn

from keras.utils import plot_model

import tensorflow.keras.layers as L

from keras.utils import model_to_dot

import tensorflow.keras.backend as K

from tensorflow.keras.models import Model

from tensorflow.keras.layers import Dense

from kaggle_datasets import KaggleDatasets

from tensorflow.keras.applications import DenseNet121



import seaborn as sns

from tqdm import tqdm

import matplotlib.cm as cm

import matplotlib.image as mpimg

from sklearn import metrics

import matplotlib.pyplot as plt

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split



tqdm.pandas()

import plotly.express as px

import plotly.graph_objects as go

import plotly.figure_factory as ff

from plotly.subplots import make_subplots



np.random.seed(0)

tf.random.set_seed(0)



import warnings

warnings.filterwarnings("ignore")
image_path = "/kaggle/input/plant-pathology-2020-fgvc7/images/"

train = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/train.csv")

test = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/test.csv")

submission = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv")
train.head()
train.columns
train[['healthy', 'multiple_diseases', 'rust', 'scab']].sum()

      

      
test.head()
submission.head()
print("Shape of Train is {}".format(train.shape))

print("Shape of test is {}".format(test.shape))

print("Shape of submission file| is {}".format(submission.shape))
def load_images(image_id):

    file_path = image_id + '.jpg'

    images = cv2.imread(image_path + file_path)

    return cv2.cvtColor(images, cv2.COLOR_BGR2RGB)

train_image = train['image_id'][:100].progress_apply(load_images)     
# We can plot images from pyplot, however  there are less functionalities in pyplot as compared to plotly.xpress

img = plt.imread("/kaggle/input/plant-pathology-2020-fgvc7/images/Train_0.jpg")

print(img.shape)

plt.imshow(img)
# There are more functionalities in Plotly Express, we can just hover over the images and see the pixel values



fig = px.imshow(train_image[1])

fig.show()
gc.collect()
temp = train[train['healthy'] ==1]

img_ids = ['../input/plant-pathology-2020-fgvc7/images/'+i+'.jpg' for i in temp['image_id']]





# Plotting the healthy images

plt.figure(figsize = (24,5))

for ind, img in enumerate(img_ids[:4]):

    plt.subplot(1,4,ind+1)

    image = mpimg.imread(img)

    plt.imshow(image)
temp = train[train['multiple_diseases'] == 1]



img_ids = ['../input/plant-pathology-2020-fgvc7/images/'+i+'.jpg' for i in temp['image_id']]



## Plotting the multiple disease images

plt.figure(figsize = (24,5))



for ind,img in enumerate(img_ids[:4]):

    plt.subplot(1,4, ind+1)

    image = mpimg.imread(img)

    plt.imshow(image)
temp = train[train['rust']== 1]

img_ids = ['../input/plant-pathology-2020-fgvc7/images/'+i+'.jpg' for i in temp['image_id']]



plt.figure(figsize= (24,5))

for ind, img in enumerate(img_ids[:4]):

    plt.subplot(1,4, ind+1)

    images = mpimg.imread(img)

    plt.imshow(images)
temp = train[train['scab']==1]

img_id = ['../input/plant-pathology-2020-fgvc7/images/'+i+'.jpg' for i in temp['image_id']]



plt.figure(figsize = (24,5))

for ind, img in enumerate(img_id[:4]):

    plt.subplot(1,4, ind+1)

    images = mpimg.imread(img)

    plt.imshow(images)
## No of entries in each category



fig = px.parallel_categories(train[["healthy", "scab", "rust", "multiple_diseases"]], color="healthy", color_continuous_scale="sunset",\

                             title="Parallel categories plot of targets")

fig.show()
def calculate_hash(im):

    md5 = hashlib.md5()

    md5.update(np.array(im).tostring())

    

    return md5.hexdigest()

    

def get_image_meta(image_id, image_src, dataset='train'):

    im = Image.open(image_src)

    extrema = im.getextrema()



    meta = {

        'image_id': image_id,

        'dataset': dataset,

        'hash': calculate_hash(im),

        'r_min': extrema[0][0],

        'r_max': extrema[0][1],

        'g_min': extrema[1][0],

        'g_max': extrema[1][1],

        'b_min': extrema[2][0],

        'b_max': extrema[2][1],

        'height': im.size[0],

        'width': im.size[1],

        'format': im.format,

        'mode': im.mode

    }

    return meta
data = []



DIR_INPUT = '/kaggle/input/plant-pathology-2020-fgvc7'





for i, image_id in enumerate(tqdm(train['image_id'].values, total=train.shape[0])):

    data.append(get_image_meta(image_id, DIR_INPUT + '/images/{}.jpg'.format(image_id)))
for i, image_id in enumerate(tqdm(test['image_id'].values, total=test.shape[0])):

    data.append(get_image_meta(image_id, DIR_INPUT + '/images/{}.jpg'.format(image_id),'test'))
meta_df = pd.DataFrame(data)



meta_df.head()
meta_df.shape
duplicates = meta_df.groupby(by='hash')[['image_id']].count().reset_index()

duplicates = duplicates[duplicates['image_id'] > 1]

duplicates.reset_index(drop=True, inplace=True)



duplicates = duplicates.merge(meta_df[['image_id', 'hash']], on='hash')



duplicates.shape
duplicates.head(10)
# Drop the duplicate images 

# train = train['image_id'!=[]]
fig = px.pie(labels=train.columns[1:],values=train.iloc[:, 1:].sum().values, 

             title="Pie Chart of Targets",

             names= train.columns[1:])

fig.show()
print(tf.__version__)

print(tf.keras.__version__)
!pip install pyyaml h5py   # to save model checkpoint
!pip install efficientnet

import efficientnet.tfkeras as efn
AUTO = tf.data.experimental.AUTOTUNE



# Detect hardware, return appropriate distribution strategy

try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.



print("REPLICAS: ", strategy.num_replicas_in_sync)





# Data access

GCS_DS_PATH = KaggleDatasets().get_gcs_path()
nb_classes = 4

BATCH_SIZE = 8 * strategy.num_replicas_in_sync

img_size = 768

EPOCHS = 40
path='../input/plant-pathology-2020-fgvc7/'

train = pd.read_csv(path + 'train.csv')

test = pd.read_csv(path + 'test.csv')

sub = pd.read_csv(path + 'sample_submission.csv')



train_paths = train.image_id.apply(lambda x: GCS_DS_PATH + '/images/' + x + '.jpg').values

test_paths = test.image_id.apply(lambda x: GCS_DS_PATH + '/images/' + x + '.jpg').values



train_labels = train.loc[:, 'healthy':].values
def decode_image(filename, label=None, image_size=(img_size, img_size)):

    bits = tf.io.read_file(filename)

    image = tf.image.decode_jpeg(bits, channels=3)

    image = tf.cast(image, tf.float32) / 255.0

    image = tf.image.resize(image, image_size)

    if label is None:

        return image

    else:

        return image, label

    

def data_augment(image, label=None, seed=2020):

    image = tf.image.random_flip_left_right(image, seed=seed)

    image = tf.image.random_flip_up_down(image, seed=seed)

           

    if label is None:

        return image

    else:

        return image, label
train_dataset = (

    tf.data.Dataset

    .from_tensor_slices((train_paths, train_labels))

    .map(decode_image, num_parallel_calls=AUTO)

    .map(data_augment, num_parallel_calls=AUTO)

    .repeat()

    .shuffle(512)

    .batch(BATCH_SIZE)

    .prefetch(AUTO)

    )
test_dataset = (

    tf.data.Dataset

    .from_tensor_slices(test_paths)

    .map(decode_image, num_parallel_calls=AUTO)

    .batch(BATCH_SIZE)

)
LR_START = 0.00001

LR_MAX = 0.0001 * strategy.num_replicas_in_sync

LR_MIN = 0.00001

LR_RAMPUP_EPOCHS = 15

LR_SUSTAIN_EPOCHS = 3

LR_EXP_DECAY = .8



def lrfn(epoch):

    if epoch < LR_RAMPUP_EPOCHS:

        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START

    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:

        lr = LR_MAX

    else:

        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN

    return lr

    

lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)



rng = [i for i in range(EPOCHS)]

y = [lrfn(x) for x in rng]

plt.plot(rng, y)

print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))
def get_model():

    base_model =  efn.EfficientNetB7(weights='imagenet', include_top=False, pooling='avg', input_shape=(img_size, img_size, 3))

    x = base_model.output

    predictions = Dense(nb_classes, activation="softmax")(x)

    return Model(inputs=base_model.input, outputs=predictions)
with strategy.scope():

    model = get_model()

    

model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
%%time

model.fit(

    train_dataset, 

    steps_per_epoch=train_labels.shape[0] // BATCH_SIZE,

    callbacks=[lr_callback],

    epochs=EPOCHS

)
SVG(tf.keras.utils.model_to_dot(

    model, show_shapes=False, show_layer_names=True, rankdir='TB',

    expand_nested=False, dpi=72, subgraph=False).create(prog='dot', format='svg'))
model.save('/kaggle/working/efficientnet_model.hdf5')
%%time

probs = model.predict(test_dataset)
sub.loc[:, 'healthy':] = probs

sub.to_csv('efnetsubmission.csv', index=False)

sub.head()
"""from tensorflow.keras.models import load_model

#from tf.keras.models import load_model

model= load_model('/kaggle/input/efficientnetmodel/efficientnet_model.hdf5')



"""