# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras import layers

from keras import optimizers

from keras import callbacks

from keras.preprocessing.image import ImageDataGenerator

from keras.utils.np_utils import to_categorical

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
Y_train = train["label"]

X_train = train.drop(labels=["label"], axis=1) 

# free some space

del train 
# Normalize the data

X_train = X_train / 255.0

test = test / 255.0
# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)

X_train = X_train.values.reshape(-1, 28, 28, 1)

test = test.values.reshape(-1, 28, 28, 1)

Y_train = to_categorical(Y_train, num_classes=10)
def predict(model, test):

    results = model.predict(test)

    # select the indix with the maximum probability

    results = np.argmax(results, axis = 1)

    results = pd.Series(results, name="Label")

    return results
random_seed = 2

# Split the train and the validation set for the fitting

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)

epochs = 30

batch_size = 86

optimizer = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

learning_rate_reduction = callbacks.ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
simple_model = Sequential()

simple_model.add(layers.Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu',

        input_shape = (28, 28, 1)))

simple_model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

simple_model.add(layers.Conv2D(filters=32, kernel_size=(5,5),padding = 'Same', activation ='relu'))

simple_model.add(layers.Conv2D(filters=32, kernel_size = (5,5),padding = 'Same', activation ='relu'))

simple_model.add(layers.MaxPool2D(pool_size=(2, 2)))

simple_model.add(layers.Conv2D(filters=32, kernel_size = (5,5),padding = 'Same', activation ='relu'))

simple_model.add(layers.Conv2D(filters=32, kernel_size = (5,5),padding = 'Same', activation ='relu'))

simple_model.add(layers.MaxPool2D(pool_size=(2, 2)))

simple_model.add(layers.Dropout(0.25))

simple_model.add(layers.Flatten())

simple_model.add(layers.Dense(500, activation="relu"))

simple_model.add(layers.Dropout(0.5))

simple_model.add(layers.Dense(10, activation="softmax"))
model = Sequential()

model.add(layers.Conv2D(filters=32, kernel_size=(5, 5),padding='Same', activation ='relu',

        input_shape=(28, 28, 1)))

model.add(layers.Conv2D(filters=32, kernel_size=(5, 5),padding = 'Same', activation ='relu'))

model.add(layers.MaxPool2D(pool_size=(2, 2)))

model.add(layers.Dropout(0.25))



model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))

model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))

model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

model.add(layers.Dropout(0.25))



model.add(layers.Flatten())

model.add(layers.Dense(256, activation="relu"))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(10, activation="softmax"))
lenet = Sequential()

lenet.add(layers.Conv2D(filters=6, kernel_size=(5, 5), padding='Same', activation ='tanh',

        input_shape=(28,28,1), strides=(1, 1)))

lenet.add(layers.AveragePooling2D(pool_size=(2, 2), padding='Valid', strides=(1, 1)))

lenet.add(layers.Conv2D(filters=16, kernel_size=(5, 5), padding='Valid', strides=(1, 1), activation='tanh'))

lenet.add(layers.AveragePooling2D(pool_size=(2, 2), padding='Valid', strides=(2, 2)))

lenet.add(layers.Conv2D(filters=120, kernel_size = (5, 5),padding = 'Valid', activation ='tanh', strides=(1, 1)))

lenet.add(layers.Flatten())

lenet.add(layers.Dense(84, activation = "tanh"))

lenet.add(layers.Dense(10, activation = "softmax"))
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

simple_model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

lenet.compile(optimizer='SGD', loss="categorical_crossentropy", metrics=["accuracy"])
datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images





datagen.fit(X_train)
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (X_val,Y_val),

                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size

                              , callbacks=[learning_rate_reduction])

sm_history = simple_model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (X_val,Y_val),

                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size

                              , callbacks=[learning_rate_reduction])

ln_history = lenet.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (X_val,Y_val),

                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size

                              , callbacks=[learning_rate_reduction])
simple_submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"), predict(simple_model, test)],axis=1)

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"), predict(model, test)], axis=1)

lenet_submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"), predict(lenet, test)], axis=1)
submission.to_csv("cnn_digit.csv",index=False)

simple_submission.to_csv("cnn_digit_simple.csv",index=False)

lenet_submission.to_csv("cnn_digit_lenet.csv",index=False)