import tensorflow as tf

import numpy as np

import time

from collections import defaultdict

from tensorflow.keras import datasets, layers, models

import matplotlib.pyplot as plt

from keras.preprocessing import image

from keras.applications import resnet50

from keras.applications.resnet50 import ResNet50

from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

import skimage

from skimage import transform

from PIL import Image

from matplotlib import cm

import cv2

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split

import pandas as pd

from matplotlib import pyplot as plt

from sklearn.datasets.samples_generator import make_blobs

from sklearn.cluster import KMeans

from sklearn.cluster import DBSCAN

from matplotlib import pyplot as plt

import random

from tqdm import tqdm
!ls ../input/gbt-energy-detection-outputs-fine-resolution
%matplotlib inline

np_images = np.load('../input/gbt-energy-detection-outputs-fine-resolution/GBT_58202_44201_KEPLER738B_fine_filtered.npy')

plt.imshow(np_images[0], cmap='viridis')

plt.figure(figsize=(12,16))

np_images = np.take(np_images,np.random.permutation(np_images.shape[0]),axis=0,out=np_images)

print(np_images.shape)
conv_only_model = ResNet50(include_top=False,

                 weights='imagenet',

                 input_shape=(32, 256, 3),

                 pooling="max")

# conv_only_model.summary()
def resize_and_rgb(img, shape=(224, 224)):

  np_images_resized = skimage.transform.resize(image=img, output_shape = shape)

  np_images_resized -= np.min(np_images_resized)

  np_images_resized = np_images_resized / np.max(np_images_resized)

  im = cv2.cvtColor(np.float32(np_images_resized),cv2.COLOR_GRAY2RGB)

  return im



converted_img = np.zeros((5000, 32, 256, 3))

for k in tqdm(range(converted_img.shape[0])):

  converted_img[k,:,:,:] = resize_and_rgb(np_images[k,:,:], (32, 256))



plt.imshow(converted_img[0])
# # Scale the input image to the range used in the trained network

x = resnet50.preprocess_input(converted_img)

# # Run the image through the deep neural network to make a prediction

predictions = conv_only_model.predict(x)

def k_means_clustering_fit(inputdata, clusters):

  kmeans = KMeans(n_clusters=clusters, init='k-means++', max_iter=3000, n_init=100, random_state=2)

  kmeans.fit(inputdata)

  generated = np.zeros((clusters))

  prediction =  kmeans.predict(inputdata)

  for i in range(0,inputdata.shape[0]):

    for k in range(0,clusters):

      if prediction[i]==k:

        generated[k]+=1

  print(generated)



  names = np.zeros((clusters))

  for t in range(0,clusters):

    names[t]=t

  plt.title('Distribution In Classes')

  plt.xlabel("Num of Samples")

  plt.ylabel("Classes")

  plt.bar(names, generated)

  return prediction, kmeans
hold =[]

clusters = 15

print("Predicted classes are ....")

hold, kmeans = k_means_clustering_fit(predictions, clusters)
k_means_labels =  kmeans.predict(predictions)

k_means_cluster_members = defaultdict(list)

for i in range(k_means_labels.shape[0]):

    k_means_cluster_members[k_means_labels[i]].append(i)
for clu in range(clusters):

  index_pick = k_means_cluster_members[clu][0]

  plt.figure(figsize=(12,16))

  plt.title('Spectrogram Sample for Cluster '+str(clu))

  plt.xlabel("Fchans")

  plt.ylabel("Time")

  plt.imshow(np_images[index_pick,:,:], interpolation='nearest')

  plt.show()
cluster = 5

for index in k_means_cluster_members[cluster][:10]:

    plt.figure(figsize=(12,16))

    plt.title('Spectrogram Sample for Cluster '+str(cluster))

    plt.xlabel("Fchans")

    plt.ylabel("Time")

    plt.imshow(np_images[index,:,:], interpolation='nearest')
cluster = 12

for index in k_means_cluster_members[cluster][:10]:

    plt.figure(figsize=(12,16))

    plt.title('Spectrogram Sample for Cluster '+str(cluster))

    plt.xlabel("Fchans")

    plt.ylabel("Time")

    plt.imshow(np_images[index,:,:], interpolation='nearest')
def DBSCAN_clustering_fit(inputdata):

  dbscan = DBSCAN(eps=0.5, min_samples=3, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30, p=None, n_jobs=-1)

  prediction = dbscan.fit_predict(inputdata)

  print(np.max(prediction))

  clusters = np.max(prediction)

  print(prediction)

  generated = np.zeros((clusters))

  for i in range(0,inputdata.shape[0]):

    for k in range(0,clusters):

      if prediction[i]==k:

        generated[k]+=1

  print(generated)



  names = np.zeros((clusters))

  for t in range(0,clusters):

    names[t]=t

  plt.bar(names, generated)

  return prediction, dbscan



dbscan_labels, dbscan = DBSCAN_clustering_fit(predictions)



dbscan_cluster_members = defaultdict(list)

for i in range(dbscan_labels.shape[0]):

    dbscan_cluster_members[dbscan_labels[i]].append(i)
np.count_nonzero(dbscan_labels == -1)
for clu in range(10):

  index_pick = dbscan_cluster_members[clu][0]

  plt.figure(figsize=(12,16))

  plt.title('Spectrogram Sample for Cluster '+str(clu))

  plt.xlabel("Fchans")

  plt.ylabel("Time")

  plt.imshow(np_images[index_pick,:,:], interpolation='nearest')

  plt.show()
cluster = 5

for index in dbscan_cluster_members[cluster][:10]:

    plt.figure(figsize=(12,16))

    plt.title('Spectrogram Sample for Cluster '+str(cluster))

    plt.xlabel("Fchans")

    plt.ylabel("Time")

    plt.imshow(np_images[index,:,:], interpolation='nearest')
cluster = 12

for index in dbscan_cluster_members[cluster][:10]:

    plt.figure(figsize=(12,16))

    plt.title('Spectrogram Sample for Cluster '+str(cluster))

    plt.xlabel("Fchans")

    plt.ylabel("Time")

    plt.imshow(np_images[index,:,:], interpolation='nearest')