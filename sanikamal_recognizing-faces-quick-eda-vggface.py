import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import networkx as nx 

from PIL import Image

from pathlib import Path

import cv2

from sklearn.decomposition import PCA

from mpl_toolkits.mplot3d import Axes3D

from mpl_toolkits.mplot3d import proj3d

from imageio import imread

from skimage.transform import resize

from scipy.spatial import distance

from keras.models import load_model

from tqdm import tqdm



import os

print(os.listdir("../input"))
train_df = pd.read_csv('../input/recognizing-faces-in-the-wild/train_relationships.csv')

train_df.head()
def add_image_path(x):

    image_path = '../input/recognizing-faces-in-the-wild/train/' + x

    if os.path.exists(image_path):

        path = os.path.join(image_path, os.listdir(image_path)[0])

        return path
train_df['p1_path'] = train_df.p1.apply(lambda x: add_image_path(x))

train_df['p2_path'] = train_df.p2.apply(lambda x: add_image_path(x))
train_df.head()
fam = os.listdir("../input/recognizing-faces-in-the-wild/train")

print('We have',len(fam),'families')

ind = []

num = []

pic = []

tot = 0

totpic = 0

for i in fam:

    path = "../input/recognizing-faces-in-the-wild/train/"+str(i)

    temp = os.listdir(path)

    ind.append(temp)

    num.append(len(temp))

    tot+=len(temp)

    for j in temp:

        newpath = path+"/"+str(j)

        temp = os.listdir(newpath)

        pic.append(temp)

        totpic+=len(temp)

print('And',tot,'individuals with',totpic,'pictures.')

print('On average, we see',tot/len(fam),'members per family.')

print('With an average of',totpic/tot,'per individual.')
img_path = Path('../input/recognizing-faces-in-the-wild/train/')
img_list = os.listdir(img_path / train_df.p1[0])
fig,ax = plt.subplots(2,5, figsize=(50,20))



for i in range(len(img_list)):

    with open(img_path / train_df.p1[0] / img_list[i] ,'rb') as f:

        img = Image.open(f)

        ax[i%2][i//2].imshow(img)

fig.show()
img_list = os.listdir(img_path / train_df.p2[0])
fig,ax = plt.subplots(2,5, figsize=(50,20))



for i in range(len(img_list)):

    with open(img_path / train_df.p2[0] / img_list[i] ,'rb') as f:

        img = Image.open(f)

        ax[i%2][i//2].imshow(img)

fig.show()
# Create graph from data 

g = nx.Graph()

color_map = []

itt = 0

for i in range(0,len(fam)): #len(names)

    g.add_node(fam[i], type = 'fam')

    for j in ind[i]:

        temp = fam[i]+j

        g.add_node(temp, type = 'ind')

        g.add_edge(fam[i], temp, color='green', weight=1)

        for k in pic[itt]:

            g.add_node(k, type = 'pic')

            g.add_edge(temp, k, color='blue', weight=1)

        itt+=1

for n1, attr in g.nodes(data=True):

    if attr['type'] == 'fam':

        color_map.append('lime')

    else: 

        if attr['type'] == 'ind':

            color_map.append('cyan')

        else:

            color_map.append('red')
# Plot the graph

plt.figure(3,figsize=(90,90))  

edges = g.edges()

colors = [g[u][v]['color'] for u,v in edges]

nx.draw(g,node_color = color_map, edge_color = colors, with_labels = True)

plt.show()
# Extract reference graph facts & metrics 

print('Reference Graph')

print('Do we have a fully connected graph? ',nx.is_connected(g))

d = list(nx.connected_component_subgraphs(g))

print('The graph contains',len(d), 'sub-graph')

nx.isolates(g)

h = g.to_directed()

N, K = h.order(), h.size()

avg_deg= float(K) / N

print ("# Nodes: ", N)

print ("# Edges: ", K)

print ("Average Degree: ", avg_deg)

# Extract reference graph facts & metrics 

in_degrees= h.in_degree() # dictionary node:degree
!pip install git+https://github.com/rcmalli/keras-vggface.git
from keras_applications.imagenet_utils import _obtain_input_shape

from keras_vggface.vggface import VGGFace



# Convolution Features

vgg_features = VGGFace(include_top=False, input_shape=(160, 160, 3), pooling='avg')

model = vgg_features
def prewhiten(x):

    if x.ndim == 4:

        axis = (1, 2, 3)

        size = x[0].size

    elif x.ndim == 3:

        axis = (0, 1, 2)

        size = x.size

    else:

        raise ValueError('Dimension should be 3 or 4')



    mean = np.mean(x, axis=axis, keepdims=True)

    std = np.std(x, axis=axis, keepdims=True)

    std_adj = np.maximum(std, 1.0/np.sqrt(size))

    y = (x - mean) / std_adj

    return y



def l2_normalize(x, axis=-1, epsilon=1e-10):

    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))

    return output



def load_and_align_images(filepaths, margin,image_size = 160):

    

    aligned_images = []

    for filepath in filepaths:

        img = imread(filepath)

        aligned = resize(img, (image_size, image_size), mode='reflect')

        aligned_images.append(aligned)

            

    return np.array(aligned_images)
def calc_embs(filepaths, margin=10, batch_size=512):

    pd = []

    for start in tqdm(range(0, len(filepaths), batch_size)):

        aligned_images = prewhiten(load_and_align_images(filepaths[start:start+batch_size], margin))

        pd.append(model.predict_on_batch(aligned_images))

    embs = l2_normalize(np.concatenate(pd))



    return embs
test_images = os.listdir("../input/recognizing-faces-in-the-wild/test/")

test_embs = calc_embs([os.path.join("../input/recognizing-faces-in-the-wild/test/", f) for f in test_images])

np.save("test_embs_vgg.npy", test_embs)
test_embs.shape
model_path = '../input/facenet-keras/facenet_keras.h5'

model = load_model(model_path)
test_embs_vgg = calc_embs([os.path.join("../input/recognizing-faces-in-the-wild/test/", f) for f in test_images])

np.save("test_embs_fnet.npy", test_embs_vgg)
test_embs_vgg.shape
df_submit = pd.read_csv('../input/recognizing-faces-in-the-wild/sample_submission.csv')
df_submit["distance"] = 0

img2idx = dict()

for idx, img in enumerate(test_images):

    img2idx[img] = idx
for idx, row in tqdm(df_submit.iterrows(), total=len(df_submit)):

    imgs = [test_embs[img2idx[img]] for img in row.img_pair.split("-")]

    df_submit.loc[idx, "distance1"] = distance.euclidean(*imgs)

    

    # For vggface

    imgs_2 = [test_embs_vgg[img2idx[img]] for img in row.img_pair.split("-")]

    df_submit.loc[idx, "distance2"] = distance.euclidean(*imgs_2)
df_submit['distance'] = df_submit[['distance1','distance2']].mean(axis=1)

df_submit.head()

all_distances = df_submit.distance.values

sum_dist = np.sum(all_distances)
probs = []

for dist in tqdm(all_distances):

    prob = np.sum(all_distances[np.where(all_distances <= dist)[0]])/sum_dist

    probs.append(1 - prob)
sub_df = pd.read_csv("../input/recognizing-faces-in-the-wild/sample_submission.csv")

sub_df.is_related = probs

sub_df.to_csv("submission.csv", index=False)
sub_df.head(10)