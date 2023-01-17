# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import keras

import os

from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data_train = pd.read_csv("../input/train.csv")
Y = data_train["label"]

nb_data = len(Y)

X = data_train.drop(labels = ["label"],axis = 1).values

        

X = np.reshape(X, (nb_data, 28,28,1))

Y = data_train["label"]

Y = keras.utils.to_categorical(Y)
image_nb = int(np.random.random() * nb_data)

image = np.reshape(X[image_nb][:], (28,28))

print(Y[image_nb])

plt.imshow(image, cmap='gray')

plt.show()
CNN = keras.models.Sequential()



CNN.add(keras.layers.Conv2D(12, (3,3), activation = 'elu', input_shape = (28,28,1), padding='same'))

CNN.add(keras.layers.MaxPooling2D((2,2)))

CNN.add(keras.layers.Conv2D(24, (3,3), activation = 'elu', padding='same'))

CNN.add(keras.layers.MaxPooling2D((2,2)))

CNN.add(keras.layers.Conv2D(36, (3,3), activation = 'elu', padding='same'))

CNN.add(keras.layers.MaxPooling2D((2,2)))

CNN.add(keras.layers.Conv2D(48, (3,3), activation = 'elu', padding='same'))

CNN.add(keras.layers.Dropout(0.1))



CNN.add(keras.layers.Flatten())

CNN.add(keras.layers.Dense(100, activation = 'elu'))

CNN.add(keras.layers.Dense(10, activation = 'softmax'))



CNN.summary()
CNN.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
checkpointer = keras.callbacks.ModelCheckpoint(filepath='weights', save_best_only=True)



train_gen = ImageDataGenerator(rescale=1.0/255,

                              width_shift_range=0.2,

                              height_shift_range=0.2)



val_gen = ImageDataGenerator(rescale=1.0/255)







X_train, X_val, Y_train, Y_val = train_test_split(X,Y,random_state=42, train_size = 0.85)



train_generator = train_gen.flow(X_train,Y_train, batch_size = 64)

val_generator = val_gen.flow(X_val,Y_val, batch_size = 64)



history = CNN.fit_generator(train_generator, steps_per_epoch = len(Y_train)/64, validation_data = val_generator, validation_steps = 654, epochs = 50, callbacks = [checkpointer])
CNN.load_weights('weights')
data_test = pd.read_csv("../input/test.csv")

nb_data = len(data_test)



X_test = data_test.values    

X_test = np.reshape(X_test, (nb_data, 28,28,1))/255.0

predictions = CNN.predict(X_test)

pd.DataFrame({"ImageId": list(range(1,nb_data+1)), "Label": np.argmax(predictions,axis=1)}).to_csv('output.csv', index=False, header=True)