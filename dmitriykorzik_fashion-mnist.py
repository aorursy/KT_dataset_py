# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from tensorflow import keras

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, MaxPool2D

from tensorflow.python.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report, accuracy_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
FM_train = pd.read_csv("../input/fashionmnist/fashion-mnist_train.csv")

FM_test = pd.read_csv("../input/fashionmnist/fashion-mnist_test.csv")
img_rows = 28

img_cols = 28

num_img = FM_train.shape[0]

x_train = FM_train.values[:,1:]

y_train = keras.utils.to_categorical(FM_train.label, 10)

x_train = x_train.reshape(num_img, img_rows, img_cols, 1)/255.0



X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size = 0.2, random_state=2)



X_test = FM_test.values[:,1:]

test_num_img = FM_test.shape[0]

y_test = keras.utils.to_categorical(FM_test.label, 10)

X_test = X_test.reshape(test_num_img, img_rows, img_cols, 1)/255.0
model = Sequential()



model.add(Conv2D(512, kernel_size=(5, 5),

                 activation='relu',

                 input_shape=(img_rows, img_cols, 1)))

model.add(BatchNormalization())



model.add(Conv2D(128, kernel_size=(4,4), activation='relu'))

model.add(BatchNormalization())



model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))

model.add(BatchNormalization())



model.add(MaxPool2D(pool_size=(2,2)))



model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))

model.add(BatchNormalization())



model.add(Flatten())



model.add(Dense(256,activation='relu'))

model.add(BatchNormalization())



model.add(Dense(256,activation='relu'))

model.add(BatchNormalization())



model.add(Dense(128,activation='relu'))

model.add(BatchNormalization())



model.add(Dense(10, activation='softmax'))
datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images





datagen.fit(X_train)
optimizer = keras.optimizers.Adam(learning_rate = 0.025)



reduce_lr = ReduceLROnPlateau(monitor='val_accuracy',factor = 0.5, patience = 3, min_lr = 1e-6, verbose=1)

earlystop = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1)

checkpoint = ModelCheckpoint("",monitor='val_accuracy', verbose=1, save_best_only=True)



model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=optimizer,

              metrics=['accuracy'])

batch_size = 128

epochs = 500
# model.fit(X_train, Y_train,

#           batch_size=batch_size,

#           epochs=epochs,

#           validation_data= (X_val,Y_val),

#           shuffle = True,

#           callbacks=[reduce_lr, earlystop, checkpoint])

model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),

                    epochs = epochs, validation_data = (X_val,Y_val),

                    steps_per_epoch=X_train.shape[0] // batch_size,

                    callbacks = [reduce_lr, earlystop, checkpoint])
model = keras.models.load_model("")

predictions = model.predict_classes(X_test)

print(predictions)

Y_test = np.argmax(y_test, axis=1)

print(classification_report(Y_test, predictions))

print(accuracy_score(Y_test, predictions))