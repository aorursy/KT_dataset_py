# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from keras.models import Sequential, load_model

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import Adam, RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau





def model_cnn():



    ###  Model Definition

    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Valid', activation='relu', input_shape=(28, 28, 1)))

    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='Same', activation='relu'))

    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Dropout(0.2))



    model.add(Conv2D(filters=64, kernel_size=(5, 5), padding='Valid', activation='relu'))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))

    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Dropout(0.2))



    model.add(Flatten())

    model.add(Dense(519, activation="relu"))  

    model.add(Dropout(0.5))

    model.add(Dense(10, activation="softmax"))



    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-3), metrics=["accuracy"])



    annealer = ReduceLROnPlateau(monitor='val_acc', patience=1, verbose=2, factor=0.5, min_lr=0.0000001) #patience=2



    datagen = ImageDataGenerator(

        featurewise_center=False,

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



    return model, annealer, datagen
import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

from sklearn.model_selection import ShuffleSplit

from sklearn.metrics import accuracy_score, confusion_matrix

from keras.utils.np_utils import to_categorical





train = pd.read_csv("../input/train.csv")

print(train.shape)



y = train["label"]

X = train.drop("label", axis = 1)

print(y.value_counts().to_dict())

y = to_categorical(y, num_classes = 10)

del train



X = X / 255.0

X = X.values.reshape(-1,28,28,1)



# Shuffle Split Train and Test from original dataset

seed=2

train_index, valid_index = ShuffleSplit(n_splits=1,

                                        train_size=0.9,

                                        test_size=None,

                                        random_state=seed).split(X).__next__()

x_train = X[train_index]

Y_train = y[train_index]

x_test = X[valid_index]

Y_test = y[valid_index]



# Parameters

epochs = 30

batch_size = 64

validation_steps = 10000



# initialize Model, Annealer and Datagen

model, annealer, datagen = model_cnn()



# Start training

train_generator = datagen.flow(x_train, Y_train, batch_size=batch_size)

test_generator = datagen.flow(x_test, Y_test, batch_size=batch_size)



history = model.fit_generator(train_generator,

                    steps_per_epoch=x_train.shape[0]//batch_size,

                    epochs=epochs,

                    validation_data=test_generator,

                    validation_steps=validation_steps//batch_size,

                    callbacks=[annealer])



# Saving Model for future API

model.save('keras_cnn.h5')



plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='lower right')

plt.show()



plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper right')

plt.show()







## Model predict

test = pd.read_csv("../input/test.csv")

print(test.shape)

test = test / 255

test = test.values.reshape(-1, 28, 28, 1)

p = np.argmax(model.predict(test), axis=1)

valid_loss, valid_acc = model.evaluate(x_test, Y_test, verbose=0)

valid_p = np.argmax(model.predict(x_test), axis=1)

target = np.argmax(Y_test, axis=1)

cm = confusion_matrix(target, valid_p)

print(cm)



submission = pd.DataFrame(pd.Series(range(1, p.shape[0] + 1), name='ImageId'))

submission['Label'] = p

filename="keras-cnn.csv"

submission.to_csv(filename, index=False)