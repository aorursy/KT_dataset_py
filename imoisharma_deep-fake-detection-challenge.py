# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Importing the required libraries



import cv2

import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import glob

import sys

%matplotlib inline

import pickle

import shutil

import time

from keras.applications.inception_v3 import InceptionV3

from keras.models import Model

from keras.layers import Input

from keras.preprocessing import image as im

from keras.applications.inception_v3 import preprocess_input,decode_predictions

from keras.models import load_model
import keras

print('Keras version:', keras.__version__)

print('OpenCV version:', cv2.__version__)
# Initializing the paths

input_path = '/kaggle/input/deepfake-detection-challenge/'

output_path = '/kaggle/working/'

train_dir = glob.glob(input_path + 'train_sample_videos/*.mp4')

test_dir = glob.glob(input_path + 'test_videos/*.mp4')
# Reading the labels of training data



df_train = pd.read_json(input_path + 'train_sample_videos/metadata.json').transpose()

df_train.head()
# Plotting the count of labels



df_train.label.value_counts().plot.bar()
# Creating the dicrectory which would contain the video frames



# shutil.rmtree(output_path + 'train_frames')

os.mkdir(output_path + 'train_frames')
# Creating frames of all the videos

train_dir = [train_dir[0]]

t = time.time()

count1 = 0

for v in train_dir:

    t1 = time.time()

    v_name = v.split("/")[-1]

    if not os.path.exists(output_path + "train_frames/" + v_name.split(".")[0]):

        os.mkdir(output_path + "train_frames/" + v_name.split(".")[0])

    count = 0

    cap = cv2.VideoCapture(v)

    while count < 100:

        cap.set(cv2.CAP_PROP_POS_MSEC,(count * 100))   

        ret,frame = cap.read()

        image = frame

        count = count + 1

        cv2.imwrite("train_frames/" + v_name.split(".")[0] + "/frame" + str(count) + ".jpg",image)

    count1 += 1

    print('Elapsed: ', time.time() - t1, ' | ', count1, '/', len(train_dir), ' | ', v)

print('Total elapsed: ', time.time() - t)
print(type(image))
# Taking the base model as Inception V3 and initializing it's weight with imagenet



# Enable internet on kernel settings

input_tensor = Input(shape = (229, 229, 3))

cnn_model = InceptionV3(input_tensor = input_tensor, weights = 'imagenet', include_top = False, pooling = 'avg')

cnn_model.summary()
# Defining the frame files



frame_files = glob.glob("train_frames/*/*.jpg")

print('Total frames captured: ', len(frame_files))
# Finding out the feature for each and every frame of all the videos



cnn_output = {}

t = time.time()

for name in frame_files:

    t1 = time.time()

    img = im.load_img(name, target_size = (229, 229, 3))

    x = im.img_to_array(img)

    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    folder_name = name.split("/")[1]

    if folder_name not in cnn_output.keys():

        cnn_output[folder_name] = []

    result = cnn_model.predict(x)

    cnn_output[folder_name].append(result)

print('Elapsed: ', time.time() - t)
# # Example of CNN output

# print(len(cnn_output['drsakwyvqv']), 'frames captured from eczrseixwq.mp4')

# cnn_output['drsakwyvqv']
# Saving the cnn_output



os.mknod(output_path + 'cnn_output.txt')

with open('cnn_output.txt', 'wb') as f:

    pickle.dump(cnn_output, f)
# Retrieving the cnn_output



with open(output_path + 'cnn_output.txt', 'rb') as f:

    cnn_output = pickle.load(f)