# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import time

import math

import tensorflow as tf

from tensorflow.keras.layers import Conv2D, Dropout,Dense,Flatten

from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.layers import Input, Dense, BatchNormalization

import functools

from sklearn.model_selection import train_test_split

import datetime





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



        

        

# Any results you write to the current directory are saved as output.
IdLookupTable = pd.read_csv("../input/facial-keypoints-detection/IdLookupTable.csv")

SampleSubmission = pd.read_csv("../input/facial-keypoints-detection/SampleSubmission.csv")

trainData = pd.read_csv("../input/facial-keypoints-detection/training.zip")  

testData = pd.read_csv("../input/facial-keypoints-detection/test.zip")  
def plot_sample(X,y,axs):

    '''

    kaggle picture is 96 by 96

    y is rescaled to range between -1 and 1

    '''

    

    axs.imshow(X.reshape(96,96),cmap="gray")

    axs.scatter(48*y[0::2]+ 48,48*y[1::2]+ 48)

trainData.head().T
imageList = []

for i in range(0,len(trainData)):

    image = trainData['Image'][i].split(' ')    

    imageList.append(image)

    

imageList = np.array(imageList,dtype = 'float')



labelList = (trainData.T[0:-1])

labelList.T.isnull().sum()
averagedFeatures = labelList.T.mean(skipna = True)

for index, i in enumerate(imageList):

    if labelList[index].isnull().any():

        plt.imshow(i.reshape(96,96),cmap='gray')

        for pos, feature in enumerate(labelList[index][::2]):

            if math.isnan(feature):

                plt.scatter(averagedFeatures[pos], averagedFeatures[pos+1], c='blue', marker='+')

#         plt.scatter(labelList[index][0::2], labelList[index][1::2], c='red', marker='x')

#         plt.show()

# #         time.sleep(1)
# remove images without a position for every feature

training = trainData.T.dropna(axis='columns')

y=[]

X=training.T['Image']

training.drop('Image')

for i in range(len(training)):

    y.append(training.iloc[i,:])

    
def prepare_model():

    input = Input(shape=(96, 96, 1,))

    

    input = Input(shape=(96, 96, 1,))

    conv_1 = Conv2d(16, (2, 2))(input)

    batch_norm_1 = BatchNormalization()(conv_1)

    conv_2 = Conv2d(32, (3, 3))(batch_norm_1)

    batch_norm_2 = BatchNormalization()(conv_2)

    conv_3 = Conv2d(64, (4, 4))(batch_norm_2)

    avg_pool_1 = AveragePooling2D((2,2))(conv_3)

    batch_norm_3 = BatchNormalization()(avg_pool_1)

    conv_128 = Conv2d(128, (4, 4))(batch_norm_3)

    avg_pool_128 = AveragePooling2D((2,2))(conv_128)

    batch_norm_128 = BatchNormalization()(avg_pool_128)

    conv_256 = Conv2d(256, (5, 5))(batch_norm_128)

    avg_pool_256 = AveragePooling2D((2,2))(conv_256)

    batch_norm_256 = BatchNormalization()(avg_pool_256)

    conv_4 = Conv2d(64, (7, 7))(batch_norm_256)

    avg_pool_4 = AveragePooling2D((2, 2))(conv_4)

    batch_norm_4 = BatchNormalization()(avg_pool_4)

    conv_5 = Conv2d(32, (7, 7))(batch_norm_4)

    flat_1 = Flatten()(conv_5)

    dense_1 = Dense(30)(flat_1)

    outputs = Dense(30)(dense_1)

    model = tf.keras.Model(input, dense_1)

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    return model







def data_pipe(X, y, shuf_size):

    dataset = (

    tf.data.Dataset.from_tensor_slices((X, y))

    .shuffle(shuf_size)

    .batch(32)

    .prefetch(1)

    .repeat())

    ite = tf.compat.v1.data.make_one_shot_iterator(dataset)

    return dataset
Activation='elu'

Input = tf.keras.layers.Input

Conv2d = functools.partial(

            tf.keras.layers.Conv2D,

            activation=Activation,

            padding='same'

        )

BatchNormalization = tf.keras.layers.BatchNormalization

AveragePooling2D = tf.keras.layers.AveragePooling2D

MaxPooling2D = tf.keras.layers.MaxPool2D

Dense = functools.partial(

            tf.keras.layers.Dense,

            activation=Activation

        )

Flatten = tf.keras.layers.Flatten

model = prepare_model()
y = []

X = []

for index, row in training.T.iterrows():

    y.append(row.iloc[:-1].values)

    X.append(np.array(row['Image'].split(' '), dtype=np.float32))



X = np.asarray(X, dtype=np.float32).reshape(len(X), 96, 96, 1)

y = np.asarray(y, dtype=np.float32)



X = [x / 255.0 for x in X]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)



train_data = data_pipe(X_train, y_train, len(X_train))

test_data = data_pipe(X_test, y_test, len(X_test))
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(filepath='./weights.hdf5', verbose=1, save_best_only=True),

    tensorboard_callback,

    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

]
model.fit(train_data, steps_per_epoch=24, epochs=250, validation_data=test_data, validation_steps=6)
len(X_train)
len(X_test)
averagedData = trainData.iloc[:,:-1].apply(lambda x: x.fillna(x.mean())) 

averagedData['Image'] = trainData['Image']
y = []

X = []

for index, row in averagedData.iterrows():

    y.append(row.iloc[:-1].values)

    X.append(np.array(row['Image'].split(' '), dtype=np.float32))



X = np.asarray(X, dtype=np.float32).reshape(len(X), 96, 96, 1)

y = np.asarray(y, dtype=np.float32)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)



train_data = data_pipe(X_train, y_train, len(X_train))

test_data = data_pipe(X_test, y_test, len(X_test))
model.fit(train_data, steps_per_epoch=26, epochs=500, validation_data=test_data, validation_steps=6)
y_test_guess = model.predict(X_test)



for i in range(1,50):

    fig, ax = plt.subplots()

    ax.imshow(X_test[i*5].reshape(96,96),cmap="gray")

    ax.scatter(y_test_guess[i*3][0::2], y_test_guess[i*3][1::2])

    plt.show()

    time.sleep(1)