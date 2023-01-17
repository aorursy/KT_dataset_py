# # This Python 3 environment comes with many helpful analytics libraries installed

# # It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# # For example, here's several helpful packages to load in 



# import numpy as np # linear algebra

# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# # Input data files are available in the "../input/" directory.

# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# # Any results you write to the current directory are saved as output.

import pandas as pd

import os

import json

import pylab

import imageio

import cv2

import time

import numpy as np
Label_Path = "../input/label/label/"

Label_File = "metadata.json"

face_root = '../input/train_frame_face_9/train_frame_face_9/'
File = open(Label_Path+Label_File)

Label_data = json.load(File)

Label_df = pd.DataFrame(Label_data)
Label_df
Label_df = pd.DataFrame(Label_df.values.T, index=Label_df.columns, columns=Label_df.index)

Label_df = Label_df.reset_index()

Label_df.rename(columns={'index':'file'}, inplace = True)
Label_df
Label_df_REAL = Label_df[Label_df['label']=='REAL']

Label_df_FAKE = Label_df[Label_df['label']=='FAKE']
Label_df_REAL
Label_df_FAKE
def read_frame(video_path):

    capture = cv2.VideoCapture(video_path)

    ret, frame = capture.read()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame = cv2.resize(frame, (384, 384))

    capture.release()

#     frame = Jet(frame)

    return frame



def read_frame_ori(video_path):

    capture = cv2.VideoCapture(video_path)

    ret, frame = capture.read()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    capture.release()

    return frame
# 指定训练集

train_nums = int(len(Label_df_REAL)* 0.8)

train_index = range(0,1)

# train_real_path_all = []

train_frame = []

train_label = []
for i in range(0,len(train_index)):

    train_sample_real = Label_df_REAL.iloc[train_index[i]]['file']

    print(train_sample_real)

    Label_df_FAKE_Train = Label_df_FAKE[Label_df_FAKE['original']==train_sample_real]

    train_real_path = face_root + train_sample_real + '_face/'

    train_image_names_real = os.listdir(train_real_path)

    for j in range(0,len(train_image_names_real)):

        frame = read_frame(train_real_path + train_image_names_real[j])

        train_frame.append(frame)

        train_label.append(1)

    train_fake_path = []

    for k in range(0,len(Label_df_FAKE_Train)):

        train_fake_path.append(face_root + Label_df_FAKE_Train.iloc[k]['file'] + '_face/')   

    for m in range(0,len(train_fake_path)):

        train_image_names_fake = os.listdir(train_fake_path[m])

        for n in range(0,len(train_image_names_fake)):

            frame = read_frame(train_fake_path[m] + train_image_names_fake[n])

            train_frame.append(frame)

            train_label.append(0) 
train_frame = np.array(train_frame)

train_frame.shape
train_label = np.array(train_label)

train_label.reshape((-1,1)).shape
# 指定test集

test_index = range(10,12)

# train_real_path_all = []

test_frame = []

test_label = []
for i in range(0,len(test_index)):

    test_sample_real = Label_df_REAL.iloc[test_index[i]]['file']

    print(test_sample_real)

    Label_df_FAKE_Test = Label_df_FAKE[Label_df_FAKE['original']==test_sample_real]

    test_real_path = face_root + test_sample_real + '_face/'

    test_image_names_real = os.listdir(test_real_path)

    



    for j in range(0,len(test_image_names_real)):

        frame = read_frame(test_real_path + test_image_names_real[j])

        test_frame.append(frame)

        test_label.append(1)

    test_fake_path = []

    for k in range(0,len(Label_df_FAKE_Test)):

        test_fake_path.append(face_root + Label_df_FAKE_Test.iloc[k]['file'] + '_face/')

    

     

    for m in range(0,len(test_fake_path)):

        test_image_names_fake = os.listdir(test_fake_path[m])

        for n in range(0,len(test_image_names_fake)):

            frame = read_frame(test_fake_path[m] + test_image_names_fake[n])

            test_frame.append(frame)

            test_label.append(0)
test_frame = np.array(test_frame)

test_frame.shape
test_label = np.array(test_label)

test_label.reshape((-1,1)).shape
from keras.models import Model

from keras import layers

from keras.layers import Dense, Input, BatchNormalization, Activation

from keras.layers import Conv2D, SeparableConv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D

#from keras.applications.imagenet_utils import _obtain_input_shape

from keras.utils.data_utils import get_file



from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler

from keras.utils import np_utils

from keras import regularizers, optimizers

from keras.optimizers import SGD

import os

import h5py

import matplotlib

import matplotlib.pyplot as plt

import random

#import cv2

from sklearn.preprocessing import LabelEncoder

from skimage import io, transform

from sklearn.metrics import accuracy_score

from scipy import misc

import numpy as np
# WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels.h5'



def Xception(nb_classes):



    # Determine proper input shape

#     input_shape = _obtain_input_shape(None, default_size=299, min_size=71, data_format='channels_last', include_top=False)



#     img_input = Input(shape=input_shape)

#     img_input = Input(shape=(227,227,3))

    img_input = Input(shape=(384,384,3))



    # Block 1

    x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False)(img_input)

    x = BatchNormalization()(x)

    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), use_bias=False)(x)

    x = BatchNormalization()(x)

    x = Activation('relu')(x)



    residual = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)

    residual = BatchNormalization()(residual)



    # Block 2

    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)

    x = BatchNormalization()(x)

    x = Activation('relu')(x)

    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)

    x = BatchNormalization()(x)



    # Block 2 Pool

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = layers.add([x, residual])



    residual = Conv2D(256, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)

    residual = BatchNormalization()(residual)



    # Block 3

    x = Activation('relu')(x)

    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)

    x = BatchNormalization()(x)

    x = Activation('relu')(x)

    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)

    x = BatchNormalization()(x)



    # Block 3 Pool

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = layers.add([x, residual])



    residual = Conv2D(728, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)

    residual = BatchNormalization()(residual)



    # Block 4

    x = Activation('relu')(x)

    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)

    x = BatchNormalization()(x)

    x = Activation('relu')(x)

    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)

    x = BatchNormalization()(x)



    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = layers.add([x, residual])



    # Block 5 - 12

    for i in range(8):

        residual = x



        x = Activation('relu')(x)

        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)

        x = BatchNormalization()(x)

        x = Activation('relu')(x)

        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)

        x = BatchNormalization()(x)

        x = Activation('relu')(x)

        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)

        x = BatchNormalization()(x)



        x = layers.add([x, residual])



    residual = Conv2D(1024, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)

    residual = BatchNormalization()(residual)



    # Block 13

    x = Activation('relu')(x)

    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)

    x = BatchNormalization()(x)

    x = Activation('relu')(x)

    x = SeparableConv2D(1024, (3, 3), padding='same', use_bias=False)(x)

    x = BatchNormalization()(x)



    # Block 13 Pool

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = layers.add([x, residual])



    # Block 14

    x = SeparableConv2D(1536, (3, 3), padding='same', use_bias=False)(x)

    x = BatchNormalization()(x)

    x = Activation('relu')(x)



    # Block 14 part 2

    x = SeparableConv2D(2048, (3, 3), padding='same', use_bias=False)(x)

    x = BatchNormalization()(x)

    x = Activation('relu')(x)



    # Fully Connected Layer

    x = GlobalAveragePooling2D()(x)

    x = Dense(1000, activation='relu')(x)

    x = Dense(nb_classes, activation='softmax')(x)



    inputs = img_input



    # Create model

    model = Model(inputs, x, name='xception')



    # Download and cache the Xception weights file

    #weights_path = get_file('xception_weights.h5', WEIGHTS_PATH, cache_subdir='models')



    # load weights

    #model.load_weights(weights_path)



    return model





# """

#     Instantiate the model by using the following line of code



#     model = Xception()



# """



train_data = train_frame.astype(np.float16)/255

test_data = test_frame.astype(np.float16)/255
#将标签量进行转化

train_labels = np_utils.to_categorical(train_label)

test_labels = np_utils.to_categorical(test_label)
#设置模型

num_classes=2

model=Xception(num_classes)

#编译模型

epochs = 50

learning_rate = 0.01

decay_rate = learning_rate / epochs

momentum = 0.9

sgd = SGD(lr=learning_rate, momentum=momentum,  decay=decay_rate, nesterov=False)

model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
history=model.fit(train_data, train_labels,validation_split=0.0, nb_epoch=epochs,batch_size=8)
#测试模型

preds = np.argmax(model.predict(test_data), axis=1)

test_labels = np.argmax(test_labels, axis=1)

print (accuracy_score(test_labels, preds))
print(preds)