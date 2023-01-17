import numpy as np # linear Algebra

import pandas as pd

import tensorflow as tf

import matplotlib.pyplot as plt

%matplotlib inline





import itertools

from sklearn.model_selection import train_test_split



from tensorflow.keras.utils import to_categorical #(one-hot-encoding)

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, MaxPooling2D

from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import ReduceLROnPlateau
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
train.shape, test.shape 
train.head() 
X_train = train.iloc[:,1:].values.astype('float32')

y_train = train['label'].values.astype('int32')

test = test.values.astype('float32')



del train
# Normalizing the data

X_train = X_train / 255.0 

test = test / 255.0 
#Convert train dataset to (num_images, img_rows, img_cols) format

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)

test = test.reshape(test.shape[0], 28, 28, 1)
y_train
y_train = to_categorical(y_train)

y_train
np.random.seed(2) 
# Splits data between training and testing

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.1, random_state = 2)
model = Sequential()



model.add(Conv2D(filters = 64, kernel_size = (5,5), padding = 'Same', activation = 'relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 64, kernel_size = (5,5), padding = 'Same', activation = 'relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.2))



model.add(Conv2D(filters = 128, kernel_size = (3,3), padding = 'Same', activation = 'relu'))

model.add(Conv2D(filters = 128, kernel_size = (3,3), padding = 'Same', activation = 'relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.2))





model.add(Flatten())

model.add(Dense(512, activation = 'relu'))

model.add(Dropout(0.2))

model.add(Dense(10, activation = 'softmax'))
model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
epochs = 50 

batch_size = 64 
datagen = ImageDataGenerator(featurewise_center=False,

                            samplewise_center=False,

                            featurewise_std_normalization=False,

                            samplewise_std_normalization=False,

                            zca_whitening=False,

                            rotation_range=10,

                            zoom_range=0.1,

                            width_shift_range=0.1,

                            height_shift_range=0.1,

                            horizontal_flip=False,

                            vertical_flip=False)

datagen.fit(X_train)
# Ajust and train the model

history = model.fit_generator(datagen.flow(X_train,y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (X_test,y_test),

                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size)
# Predict the test database

# OBS: That return a probability of results

results = model.predict(test)

# Takes the highest probability

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")
# Makes a dataframe with the answers

output = pd.concat([pd.Series(range(1,28001), name = 'ImageId'), results], axis = 1)

output
output.to_csv('submission.csv', index=False)