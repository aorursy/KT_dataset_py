import numpy as np # linear algebra

import pandas as pd # data processing

from keras.layers import Conv2D, Dense,Flatten,Dropout

from keras.models import Sequential

from tensorflow.keras import utils

#import tensorflow as tf

from sklearn.model_selection import train_test_split





from keras.layers.convolutional import MaxPooling2D

train_dataset = pd.read_csv('../input/digit-recognizer/train.csv')

test_data = pd.read_csv('../input/digit-recognizer/test.csv')

x_train = train_dataset.drop('label',axis = 1)

y_train = train_dataset['label']

del train_dataset

x_train = x_train / 255

test_data = test_data / 255

x_train = x_train.values.reshape(-1, 28, 28, 1)

test_data = test_data.values.reshape(-1, 28, 28, 1)

y_train = utils.to_categorical(y_train, 10)

model = Sequential()

model.add(Conv2D(filters = 128,kernel_size = (3,3),input_shape = (28,28, 1),activation = 'relu',padding = 'same'))

model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Dropout(0.1))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.1))

model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(120, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(84, activation='relu'))

model.add(Dense(10, activation='softmax'))

# Compile model

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



model.fit(x_train, y_train, batch_size=50, epochs=50, validation_split=0.1,verbose=1)

predictions = model.predict(test_data)

results = np.argmax(predictions,axis = 1)



Label = pd.Series(results,name = 'Label')

ImageId = pd.Series(range(1,28001),name = 'ImageId')

submission = pd.concat([ImageId,Label],axis = 1)

submission.to_csv('submission.csv',index = False)