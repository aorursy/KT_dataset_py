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

convert = df_train['Size']

converted = [0 for i in range(len(convert))]

for i in range(len(convert)):

    if(convert[i]=='Medium'):

        converted[i]=1

    elif(convert[i]=='Small'):

        converted[i]=2

    elif(convert[i]=='Big'):

        converted[i]=3

    else:

        convert[i]=-1



df_train['Size'] = converted

df_train = df_train.apply(pd.to_numeric, errors='coerce').dropna()

dropped_train = df_train.drop(['ID', 'Class'], axis=1)

Y = df_train['Class']
Y = np.asarray(Y)

Y = np_utils.to_categorical(Y)

X = np.asarray(dropped_train)

X=np.reshape(X, (358,11,1))

# as first layer in a sequential model:

model = Sequential()

model.add(Dense(63, input_shape=(11,1), activation='relu'))

# now the model will take as input arrays of shape (*, 11)

# and output arrays of shape (*, 63)

model.add(Dense(63, activation='relu'))

# after the first layer, you don't need to specify

# the size of the input anymore:

model.add(Dense(63, activation='relu'))

model.add(Flatten())

model.add(Dense(6, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X , Y , batch_size = 20 , epochs = 500)
df_test = pd.read_csv("/kaggle/input/bitsf312-lab1/test.csv", sep=',')

convert = df_test['Size']

converted = [0 for i in range(len(convert))]

for i in range(len(to_encode)):

    if(convert[i]=='Medium'):

        converted[i]=1

    elif(convert[i]=='Small'):

        converted[i]=2

    elif(convert[i]=='Big'):

        converted[i]=3

    else:

        converted[i]=-1

df_test['Size'] = converted

Q = df_test['ID']

df_test=df_test.drop({'ID'},axis=1)

X = np.asarray(df_test)

X = np.reshape(X,(159,11,1))

Y_t = model.predict(X)

to_df = {'ID':Q, 'Class':np.argmax(Y_t,axis=1)}

final = pd.DataFrame(data=to_df)

final
final.to_csv('final.csv', index=False)