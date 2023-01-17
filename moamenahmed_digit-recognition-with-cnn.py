import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from keras import layers

from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D

from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D

from keras.models import Model,Sequential

from keras.preprocessing import image

from keras.utils import layer_utils,to_categorical

from keras.utils.data_utils import get_file

from keras.applications.imagenet_utils import preprocess_input

import keras.backend as K

K.set_image_data_format('channels_last')

import os

np.random.seed(2)
df_train_dev = pd.read_csv('../input/train.csv')

df_test=pd.read_csv('../input/test.csv')
print(df_train_dev.info())

print(df_test.info())
df_X_train_dev=df_train_dev.drop('label',axis=1)

print(type(df_X_train_dev))

print(df_X_train_dev.info())

X_train_dev=df_X_train_dev.values
X_test=df_test.values
df_Y_train_dev=df_train_dev.get('label')

print(type(df_Y_train_dev))

Y_train_dev=df_Y_train_dev.values
print(X_train_dev.shape)

print(Y_train_dev.shape)

print(X_test.shape)
X_train_dev_2D=X_train_dev.reshape(42000,28,28,1)

X_test_2D=X_test.reshape(28000,28,28,1)

print(X_train_dev_2D.shape)

print(X_test_2D.shape)
plt.imshow(X_train_dev_2D[10,:,:,0])
Y_train_dev_hot=to_categorical(Y_train_dev)

print(Y_train_dev_hot.shape)
print(Y_train_dev_hot[10,:])
model_cnn = Sequential()

model_cnn.add(Conv2D(filters=32, kernel_size=(5, 5),activation='relu', padding='same',input_shape=(28,28,1)))

model_cnn.add(Conv2D(32, (3, 3),activation='relu'))

model_cnn.add(MaxPooling2D(pool_size=(2, 2)))

model_cnn.add(Dropout(0.25))



model_cnn.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model_cnn.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model_cnn.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model_cnn.add(Dropout(0.25))



model_cnn.add(Flatten())

model_cnn.add(Dense(512,activation='relu'))

model_cnn.add(Dropout(0.5))

model_cnn.add(Dense(10,activation='softmax'))
model_cnn.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model_cnn.fit(X_train_dev_2D,Y_train_dev_hot, batch_size=None, epochs=10)
prediction_hot=model_cnn.predict(X_test_2D)

prediction=np.argmax(prediction_hot,axis=1)

print(prediction.shape)
ImageId=np.arange(1,28001)

dic={'ImageId': ImageId, 'Label':prediction }

ouput_df=pd.DataFrame(data=dic)

ouput_df.to_csv('output_submission.csv',index=False )