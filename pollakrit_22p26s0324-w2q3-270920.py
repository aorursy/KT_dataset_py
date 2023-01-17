import tensorflow as tf

import keras



import sklearn

from sklearn.model_selection import train_test_split, KFold

from sklearn.metrics import jaccard_score



from scipy import stats



import seaborn as sns



import skimage

from skimage.transform import rotate



from tensorflow.keras.utils import to_categorical

from tensorflow.keras import Sequential

from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPool2D, UpSampling2D, GlobalMaxPool2D, GlobalAveragePooling2D, Conv2DTranspose, concatenate

from tensorflow.keras.layers import Dense, Dropout, Activation, Reshape, Flatten, Input

from tensorflow.keras.models import Model, load_model



from tensorflow.keras.applications.resnet50 import ResNet50

from tensorflow.keras.applications import NASNetMobile, Xception, DenseNet121, MobileNetV2, InceptionV3, InceptionResNetV2, vgg16, resnet50, inception_v3, xception, DenseNet201

from tensorflow.keras.applications.vgg16 import VGG16





from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.callbacks import CSVLogger

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.optimizers import Adam

from tensorflow.keras import metrics

from tensorflow.keras.preprocessing.image import ImageDataGenerator



from datetime import datetime



import numpy as np

import os

import cv2

import pandas as pd

# import imutils

import random

from PIL import Image

import matplotlib.pyplot as plt
df = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

df = df.dropna(0)

df = df.sample(1000)

df.sample(100)
len(df)
from sklearn import preprocessing
Room_type = ['Entire home/apt','Private room','Shared room']

df['room_type'] = np.array(df['room_type'].replace(Room_type,['0','1','2']))



columns = ['latitude', 'longitude', 'price', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'room_type']
PTF = preprocessing.PowerTransformer(method='yeo-johnson', standardize=True)

mat = PTF.fit_transform(df[columns])

mat[:5]
X = pd.DataFrame(mat, columns=columns)

X.sample(10)
from sklearn.cluster import AgglomerativeClustering
HC = AgglomerativeClustering(n_clusters=5, linkage='ward')

HC.fit(X)

df['cluster'] = HC.labels_

plt.figure(figsize=(20,10))

sns.scatterplot(x='longitude', y='latitude', hue='cluster',s=40, data=df)
import scipy.cluster.hierarchy as sch

Showing_Data = 80

fig, ax = plt.subplots(figsize=(20, 7))

Dendrogram = sch.dendrogram(sch.linkage(X[:Showing_Data], method='ward'), ax=ax, labels=df['name'][:Showing_Data].values)
sns.clustermap(X, col_cluster=False, metric="correlation")