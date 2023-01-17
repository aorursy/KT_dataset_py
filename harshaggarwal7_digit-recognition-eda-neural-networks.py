'''import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))'''
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import tensorflow as tf
df_train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv") #../input/digit-recognizer/train.csv

df_test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

df_submission = pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv")
df_submission
display("Train File", df_train.head())

display("Test File", df_test.head())
print("Train File", df_train.shape)

print("Test File", df_test.shape)
print("Train File", df_train.isnull().any().sum())

print("Test File", df_test.isnull().any().sum())
display("Train File", df_train.describe())

display("Test File", df_test.describe())
sns.countplot(df_train['label'])
y_train = df_train['label'].astype('float32')

X_train = df_train.drop(['label'], axis=1).astype('int32')

X_test = df_test.astype('float32')

X_train.shape, y_train.shape, X_test.shape
X_train = X_train/255

X_test = X_test/255
X_train = X_train.values.reshape(-1,28,28,1)

X_test = X_test.values.reshape(-1,28,28,1)

X_train.shape , X_test.shape
from keras.utils.np_utils import to_categorical

y_train = to_categorical(y_train, num_classes = 10)

y_train.shape
from sklearn.model_selection import train_test_split

X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, test_size = 0.1, random_state=42)
from keras.layers import Input,InputLayer, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D

from keras.layers import AveragePooling2D, MaxPooling2D, Dropout

from keras.models import Sequential,Model

from keras.optimizers import SGD

from keras.callbacks import ModelCheckpoint,LearningRateScheduler

import keras

from keras import backend as K
# Building a CNN model

input_shape = (28,28,1)

X_input = Input(input_shape)



# layer 1

x = Conv2D(64,(3,3),strides=(1,1),name='layer_conv1',padding='same')(X_input)

x = BatchNormalization()(x)

x = Activation('relu')(x)

x = MaxPooling2D((2,2),name='maxPool1')(x)

# layer 2

x = Conv2D(32,(3,3),strides=(1,1),name='layer_conv2',padding='same')(x)

x = BatchNormalization()(x)

x = Activation('relu')(x)

x = MaxPooling2D((2,2),name='maxPool2')(x)

# layer 3

x = Conv2D(32,(3,3),strides=(1,1),name='conv3',padding='same')(x)

x = BatchNormalization()(x)

x = Activation('relu')(x)

x = MaxPooling2D((2,2), name='maxPool3')(x)

# fc

x = Flatten()(x)

x = Dense(64,activation ='relu',name='fc0')(x)

x = Dropout(0.25)(x)

x = Dense(32,activation ='relu',name='fc1')(x)

x = Dropout(0.25)(x)

x = Dense(10,activation ='softmax',name='fc2')(x)



conv_model = Model(inputs=X_input, outputs=x, name='Predict')

conv_model.summary()
# Adam optimizer

conv_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

conv_model.fit(X_train, y_train, epochs=10, batch_size=100, validation_data=(X_cv,y_cv))
y_pred = conv_model.predict(X_test)

y_pred = np.argmax(y_pred,axis=1)

my_submission = pd.DataFrame({'ImageId': list(range(1, len(y_pred)+1)), 'Label': y_pred})

my_submission.to_csv('submission.csv', index=False)