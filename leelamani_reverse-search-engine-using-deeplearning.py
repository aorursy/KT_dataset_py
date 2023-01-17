#!pip install tqdm

#!pip install pillow

#!pip install keras --upgrade

import requests

import os

import numpy as np

from numpy.linalg import norm

import joblib as pickle

from tqdm import tqdm

import os

import PIL

import time

import tensorflow as tf

from tensorflow.keras.preprocessing import image

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import gc

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

#from tensorflow.keras.applications.MobileNet import MobileNetV2,preprocess_input

#from tensorflow.keras.applications.mobilenet import MobileNet,preprocess_input

import math

from sklearn.neighbors import NearestNeighbors

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from sklearn.decomposition import PCA

!pip install scikit-learn --upgrade
# If you are using tensorflow 2.x and run into memory issues uncomment below lines.

#config = tf.compat.v1.ConfigProto()

#config.gpu_options.allow_growth = True

#session = tf.compat.v1.InteractiveSession(config=config)
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
img_size =224



model = ResNet50(weights='imagenet', include_top=False,input_shape=(img_size, img_size, 3),pooling='max')
batch_size = 64

root_dir = '/kaggle/input/caltech101/101_ObjectCategories'



img_gen = ImageDataGenerator(preprocessing_function=preprocess_input)



datagen = img_gen.flow_from_directory(root_dir,

                                        target_size=(img_size, img_size),

                                        batch_size=batch_size,

                                        class_mode=None,

                                        shuffle=False)



num_images = len(datagen.filenames)

num_epochs = int(math.ceil(num_images / batch_size))



feature_list = model.predict_generator(datagen, num_epochs)
print("Num images   = ", len(datagen.classes))

print("Shape of feature_list = ", feature_list.shape)
# Get full path for all the images in our dataset



filenames = [root_dir + '/' + s for s in datagen.filenames]
neighbors = NearestNeighbors(n_neighbors=5,

                             algorithm='ball_tree',

                             metric='euclidean')

neighbors.fit(feature_list)
img_path = '/kaggle/input/antimage/ant.jpg'

# ref https://datascience.stackexchange.com/questions/31167/how-to-predict-an-image-using-saved-model

input_shape = (img_size, img_size, 3)

img = image.load_img(img_path, target_size=(input_shape[0], input_shape[1]))

img_array = image.img_to_array(img)

expanded_img_array = np.expand_dims(img_array, axis=0)

preprocessed_img = preprocess_input(expanded_img_array)

test_img_features = model.predict(preprocessed_img, batch_size=1)



_, indices = neighbors.kneighbors(test_img_features)
def similar_images(indices):

    plt.figure(figsize=(15,10), facecolor='white')

    plotnumber = 1    

    for index in indices:

        if plotnumber<=len(indices) :

            ax = plt.subplot(2,4,plotnumber)

            plt.imshow(mpimg.imread(filenames[index]), interpolation='lanczos')            

            plotnumber+=1

    plt.tight_layout()



print(indices.shape)



plt.imshow(mpimg.imread(img_path), interpolation='lanczos')

plt.xlabel(img_path.split('.')[0] + '_Original Image',fontsize=20)

plt.show()

print('********* Predictions ***********')

similar_images(indices[0])
pca = PCA(n_components=100)

pca.fit(feature_list)

compressed_features = pca.transform(feature_list)
neighbors_pca_features = NearestNeighbors(n_neighbors=5,

                             algorithm='ball_tree',

                             metric='euclidean').fit(compressed_features)
test_img_compressed = pca.transform(test_img_features)

distances, indices = neighbors_pca_features.kneighbors(test_img_compressed)

print(indices.shape)

plt.imshow(mpimg.imread(img_path), interpolation='lanczos')

plt.xlabel(img_path.split('.')[0] + '_Original Image',fontsize=20)

plt.show()

print('********* Predictions ***********')

similar_images(indices[0])