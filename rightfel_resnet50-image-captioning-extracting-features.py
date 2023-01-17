# Scikit-learn includes many helpful utilities

from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle



import re

import time

import json

from glob import glob

from PIL import Image



import tensorflow as tf

import numpy as np

from keras.preprocessing import image

from tensorflow.keras.applications.resnet50 import ResNet50

from tensorflow.keras.applications.resnet50 import preprocess_input

import os

import pickle
folder = "../input/flickr8k/Flickr_Data/Flickr_Data/Images/"

images = os.listdir("../input/flickr8k/Flickr_Data/Flickr_Data/Images")
image_model = ResNet50(weights='imagenet')

model_new = tf.keras.Model(image_model.input, image_model.layers[-2].output)
img_features = dict()

import time

start_time = time.time()

for img in images:

    img1 = image.load_img(folder + img, target_size=(224, 224, 3))

    x = image.img_to_array(img1)

    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    

    fea_x = model_new.predict(x)

    fea_x1 = np.reshape(fea_x , fea_x.shape[1])

    img_features[img] = fea_x1

print("--- %s seconds for feature extraction ---" % (time.time() - start_time))
pickle.dump(img_features,open("img_extract_rn50.pkl","wb"))