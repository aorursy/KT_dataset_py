import numpy as np



np.random.seed(800)
from sklearn.datasets import make_moons



x,y = make_moons(n_samples=100,noise=0.2,random_state=1)
#plot the graph

import matplotlib.pyplot as plt

plt.scatter(x[:,0],x[:,1],c=y,s=100)

plt.show()
#importing libraries

import tensorflow as tf

import warnings

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=42)
model = Sequential()

model.add(Dense(500, input_dim=2, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', 

              optimizer='adam', 

              metrics=['accuracy'])



history = model.fit(x_train, y_train, 

                    validation_data=(x_test, y_test), 

                    epochs=4000, verbose=0)
plt.plot(history.history['loss'], label='train')

plt.plot(history.history['val_loss'], label='test')

plt.legend()

plt.show()


model = Sequential()

model.add(Dense(500, input_dim=2, activation='relu',kernel_regularizer='l2'))

model.add(Dense(1, activation='sigmoid',kernel_regularizer='l2'))

model.compile(loss='binary_crossentropy', 

              optimizer='adam', 

              metrics=['accuracy'])



history = model.fit(x_train, y_train, 

                    validation_data=(x_test, y_test), 

                    epochs=4000, verbose=0)
plt.plot(history.history['loss'], label='train')

plt.plot(history.history['val_loss'], label='test')

plt.legend()

plt.show()
from tensorflow.keras.layers import Dropout

model = Sequential()

model.add(Dense(500, input_dim=2, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', 

              optimizer='adam', 

              metrics=['accuracy'])



history = model.fit(x_train, y_train, 

                    validation_data=(x_test, y_test), 

                    epochs=500, verbose=0)
plt.plot(history.history['loss'], label='train')

plt.plot(history.history['val_loss'], label='test')

plt.legend()

plt.show()
from keras.preprocessing.image import ImageDataGenerator



aug = ImageDataGenerator(

    rotation_range=20,

    zoom_range=0.15,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.15,

    horizontal_flip=True,

    fill_mode="nearest")
from tensorflow.keras.callbacks import EarlyStopping



model = Sequential()

model.add(Dense(128, input_dim=2, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', 

              optimizer='adam', 

              metrics=['accuracy'])



callback= EarlyStopping(monitor='val_loss')

history = model.fit(x_train, y_train, 

                    validation_data=(x_test, y_test), 

                    epochs=2000,callbacks=[callback])
plt.plot(history.history['loss'], label='train')

plt.plot(history.history['val_loss'], label='test')

plt.legend()

plt.show()
