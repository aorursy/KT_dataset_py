import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns



import os

print(os.listdir("../input"))
# Load the data

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



print(train.shape, test.shape)
Y_train = train["label"]



# Drop 'label' column

X_train = train.drop(labels = ["label"],axis = 1) 



# free some space

del train
counts = Y_train.value_counts().to_frame().sort_index()

counts.plot.bar(figsize=(8, 3), color='navy')

counts.T
# Normalize the data

X_train = X_train / 255.0

test = test / 255.0
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding



X_train = X_train.values.reshape(X_train.shape[0], 28, 28, 1)

X_test = test.values.reshape(test.shape[0], 28, 28, 1)



# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])

Y_train = to_categorical(Y_train, num_classes = 10)

Y_train
from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization

from keras.utils import np_utils
model = Sequential()

dim = 28

nclasses = 10



model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', input_shape=(dim,dim,1)))

model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu'))

#model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.2))



model.add(Conv2D(filters=64, kernel_size=(5,5), padding='same', activation='relu'))

model.add(Conv2D(filters=64, kernel_size=(5,5), padding='same', activation='relu'))

#model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.2))



model.add(Flatten())

model.add(Dense(120, activation='relu'))

model.add(Dense(84, activation='relu'))

model.add(Dense(nclasses, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau



# Set a learning rate annealer

learning_rate_reduction = ReduceLROnPlateau(monitor='acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=1e-7)



datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        #shear_range=np.pi / 12,   # shear intensity (degrees)

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range=0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images

datagen.fit(X_train)



# Fit the model

history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=512),

                              epochs=50, 

                              verbose=1, 

                              steps_per_epoch=X_train.shape[0]/512, 

                              callbacks=[learning_rate_reduction])
loss_and_metrics = model.evaluate(X_train, Y_train, batch_size=128)

print(loss_and_metrics)
pred = model.predict_classes(X_test)



submission = pd.DataFrame({

    "ImageId": list(range(1,len(pred) + 1)),

    "Label": pred})

submission.to_csv("submission.csv", index=False, header=True)