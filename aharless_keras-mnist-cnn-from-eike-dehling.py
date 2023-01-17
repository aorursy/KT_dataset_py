import numpy as np

import pandas as pd
from keras.utils.np_utils import to_categorical

from keras import backend as K
K.set_image_dim_ordering('th') #input shape: (channels, height, width)



train_df = pd.read_csv("../input/train.csv")

valid_df = pd.read_csv("../input/test.csv")



x_train = train_df.drop(['label'], axis=1).values.astype('float32')

Y_train = train_df['label'].values

x_valid = valid_df.values.astype('float32')



img_width, img_height = 28, 28



n_train = x_train.shape[0]

n_valid = x_valid.shape[0]



x_train = x_train.reshape(n_train,1,img_width,img_height)

x_valid = x_valid.reshape(n_valid,1,img_width,img_height)



x_train = x_train/255 #normalize from [0,255] to [0,1]

x_valid = x_valid/255



y_train = to_categorical(Y_train)
from keras.models import Sequential

from keras.layers.convolutional import *

from keras.layers.core import Dropout, Dense, Flatten
model = Sequential()

model.add(Conv2D(filters=40, kernel_size=(5,5), activation='relu', batch_input_shape=(None, 1, img_width, img_height)))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=40, kernel_size=(5,5), activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(1000, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(1000, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax', activity_regularizer='l1_l2'))
model.compile(optimizer='adam',

              loss='categorical_crossentropy',

              metrics=['accuracy'])
model.fit(x_train,

          y_train,

          batch_size=128,

          epochs=1,  # For running on your own laptop, ~ 15 epochs produces much better results but takes longer

          verbose=2,

          validation_split=.2)
yPred = model.predict_classes(x_valid,batch_size=32,verbose=1)

np.savetxt('predictions.csv', np.c_[range(1,len(yPred)+1),yPred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')