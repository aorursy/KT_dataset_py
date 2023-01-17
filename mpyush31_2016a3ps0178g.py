import numpy as np 

import pandas as pd 

import os

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

from keras.layers import Conv2D , MaxPooling2D , Dense, Flatten , Input , Dropout

from keras.models import Sequential , Model

import keras

import tensorflow as tf

from PIL import Image

from keras.models import model_from_json

import os

from keras.preprocessing.image import ImageDataGenerator

from keras import utils as np_utils  

from matplotlib import image



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('/kaggle/input/bitsf312-lab1/train.csv',error_bad_lines=False)

to_encode = df_train['Size']

encoded = [0 for i in range(len(to_encode))]

for i in range(len(to_encode)):

    if(to_encode[i]=='Medium'):

        encoded[i]=1

    elif(to_encode[i]=='Small'):

        encoded[i]=2

    elif(to_encode[i]=='Big'):

        encoded[i]=3

    else:

        to_encode[i]=-1

df_train['Size'] = encoded

df_train=df_train.apply(pd.to_numeric, errors='coerce').dropna()

df_train_ = df_train.drop(['ID', 'Class'], axis=1)

Y = df_train['Class']

Y = np.asarray(Y)

Y = np_utils.to_categorical(Y)

X = np.asarray(df_train_)

X=np.reshape(X, (358,11,1))

model = Sequential()

model.add(Dense(63, input_shape=(11,1), activation='relu'))

model.add(Dense(63, activation='relu'))

model.add(Dense(63, activation='relu'))

model.add(Flatten())

model.add(Dense(6, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X , Y , batch_size = 20 , epochs = 300)
df_test = pd.read_csv("/kaggle/input/bitsf312-lab1/test.csv", sep=',')

to_encode = df_test['Size']

encoded = [0 for i in range(len(to_encode))]

for i in range(len(to_encode)):

    if(to_encode[i]=='Medium'):

        encoded[i]=1

    elif(to_encode[i]=='Small'):

        encoded[i]=2

    elif(to_encode[i]=='Big'):

        encoded[i]=3

    else:

        encoded[i]=-1

df_test['Size'] = encoded

Q = df_test['ID']

df_test=df_test.drop({'ID'},axis=1)

X = np.asarray(df_test)

X = np.reshape(X,(159,11,1))

Y_t = model.predict(X)

to_df = {'ID':Q, 'Class':np.argmax(Y_t,axis=1)}

answer = pd.DataFrame(data=to_df)

answer
answer.to_csv('output.csv', index=False)