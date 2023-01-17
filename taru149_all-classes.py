data = pd.read_csv('/kaggle/input/data/Data_Entry_2017.csv')
new_df = data.copy()
new_df['Finding Labels'] = data['Finding Labels'].str.split('|')
new_df['Finding Labels']
len(new_df.loc[1,'Finding Labels'])
len(new_df)
temp_list = []
for i in range(len(new_df)):
    for j in range(len(new_df.loc[i, 'Finding Labels'])):
        if  new_df.loc[i, 'Finding Labels'][j] not in temp_list:
            temp_list.append(new_df.loc[i, 'Finding Labels'][j])
temp_list
for i in range(len(temp_list)):
    new_df[temp_list[i]] = ""
new_df.head()
for i in range(len(new_df)):
    for j in range(len(new_df.loc[i, 'Finding Labels'])):
        new_df.loc[i,new_df.loc[i, 'Finding Labels'][j]] = 1
    for col_name in (temp_list):
        if new_df.loc[i, col_name]=="":
            new_df.loc[i, col_name] = 0
for col_name in temp_list:
    new_df[col_name] = np.where(new_df["Finding Labels"])
new_df.to_csv('new_data.csv')
from __future__ import division, print_function
import tensorflow as tf
import theano
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, LSTM, Embedding, Activation, Lambda, Bidirectional
from keras.engine import Input, Model, InputSpec
from keras.preprocessing.sequence import pad_sequences
from keras.utils import plot_model
from keras.utils.data_utils import get_file
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.utils import class_weight
from keras import backend as K
from keras.preprocessing import sequence
from keras.models import model_from_json
import os
import pydot
import graphviz

from gensim import models
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Reshape, Flatten, concatenate, Input, Conv1D, GlobalMaxPooling1D, Embedding,AveragePooling2D
#from keras.layers import Dense, Dropout, Reshape, Flatten, concatenate, Input, Conv1D, GlobalMaxPooling1D, Embedding,AveragePooling2D

from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from sklearn.model_selection import train_test_split
import numpy as np
from keras.layers import GlobalMaxPooling2D,AveragePooling2D

import pandas as pd
import os
import collections
import re
import string

import sys
import os
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Convolution3D, MaxPooling3D
from keras import callbacks

# set the matplotlib backend so figures can be saved in the background
import matplotlib
#matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
#from pyimagesearch.lenet import LeNet
#from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os
from imutils import paths
nb_filters1 = 32
nb_filters2 = 64
conv1_size = 3
conv2_size = 2
pool_size = 2
classes_num = 3
lr = 0.0004
epochs = 20
img_width, img_height = 256, 256


# initialize the number of epochs to train for, initia learning rate,
# and batch size
EPOCHS = 500
INIT_LR = 1e-3
BS = 32

# initialize the data and labels
data = []
labels = []
dir_labels = ()
num_class = 0
imagePaths = sorted(list(paths.list_images('/kaggle/input/data/images_002')))
#path_list = os.listdir('/kaggle/input/data/images_002/images')
new_df = pd.read_csv('/kaggle/input/allclassesdata/new_data.csv')
data = []
labels = []
i = 0


for img_name in new_df["Image Index"]:
    for path in imagePaths[0:5000]:
        if img_name==path[-16:]:
            image = cv2.imread(path)
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (256, 256))
            image = img_to_array(image)
            data.append(image)
            labels.append(new_df.loc[i, 'Cardiomegaly':'Consolidation'])
        else:
            pass
    i = i+1
import pickle
with open('data2.txt', 'wb') as fp:
    pickle.dump(data, fp)
with open('labels2.txt', 'wb') as fp:
    pickle.dump(labels, fp)