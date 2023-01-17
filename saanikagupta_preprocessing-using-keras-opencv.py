# Importing the required libraries



import cv2

import numpy as np

import pandas as pd

import os

import glob

import pickle

import time

from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input

from tensorflow.keras.preprocessing import image
# Version



import tensorflow as tf

print('Tensorflow version:', tf.__version__)

print('OpenCV version:', cv2.__version__)
# Initializing the paths



input_path = '/kaggle/input/deepfake-detection-challenge/'

train_dir = glob.glob(input_path + 'train_sample_videos/*.mp4')
# Reading the labels of training data



df_train = pd.read_json(input_path + 'train_sample_videos/metadata.json').transpose()

df_train.head()
# Plotting the count of labels



# Fake class is in majority

df_train.label.value_counts().plot.bar()
df_train.head()
# Undersampling



i = 0

for ind in df_train.index: 

    if(i > 243):

        break

    if(df_train['label'][ind] == 'FAKE'):

        i += 1

        train_dir.remove(input_path + 'train_sample_videos/' + ind)
len(train_dir)
# Taking the base model as Inception V3 and initializing its weight with imagenet



# Enable internet on kernel settings

input_tensor = Input(shape = (229, 229, 3))

cnn_model = InceptionV3(input_tensor = input_tensor, weights = 'imagenet', include_top = False, pooling = 'avg')

cnn_model.summary()
# Number of frames per video



fpv = 5
# Creating 40 frames per video, resizing it to (229, 229, 3) and feeding it to Pretrained Inception-V3 to extract features

# Extracted features are stored in cnn_output



t = time.time()

cnn_output = {}

# count1 = 0

for v in train_dir:

    t1 = time.time()

    folder_name = v.split('/')[5]

    cap = cv2.VideoCapture(v)

    count = 0

    while count < fpv:

        cap.set(cv2.CAP_PROP_POS_MSEC,((count * 30) + 5000))   

        ret, frame = cap.read()

        frame = cv2.resize(frame, (229, 229))

        x = image.img_to_array(frame)

        x = np.expand_dims(x, axis = 0)

        x = preprocess_input(x)

        result = cnn_model.predict(x)

        if folder_name not in cnn_output.keys():

            cnn_output[folder_name] = []

        cnn_output[folder_name].append(list(result))

        count = count + 1

#     count1 += 1

#     print('Elapsed: ', time.time() - t1, ' | ', count1, '/', len(train_dir), ' | ', v)

print('Total elapsed: ', time.time() - t)
# Saving the cnn_output



os.mknod('cnn_output.txt')

with open('cnn_output.txt', 'wb') as f:

    pickle.dump(cnn_output, f)
# # Retrieving the cnn_output



# with open('cnn_output.txt', 'rb') as f:

#     cnn_output = pickle.load(f)