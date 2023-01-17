import tensorflow as tf



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

from keras.layers import BatchNormalization,Activation

from keras.backend import clear_session
(x_train,y_train),(x_test,y_test) = mnist.load_data()

img_index=88

print('Label :',y_train[img_index])

plt.imshow(x_train[img_index],cmap='gray')

plt.show()
x_train = x_train.reshape(x_train.shape + (1,))

x_test = x_test.reshape(x_test.shape + (1,))
x_train = x_train.astype(np.float32)

x_test = x_test.astype(np.float32)



x_train /= 255

x_test /= 255
mod = Sequential()

mod.add(Conv2D(16,3,strides=2,padding='same',input_shape=(28,28,1)))

mod.add(BatchNormalization(momentum=0.8))

mod.add(Activation('relu'))

mod.add(Conv2D(32,3,strides=2,padding='same'))

mod.add(BatchNormalization(momentum=0.8))

mod.add(Activation('relu'))

mod.add(Flatten())

mod.add(Dense(256,activation='relu'))

mod.add(Dense(64,activation='relu'))

mod.add(Dense(10,activation='softmax'))
mod.summary()
mod.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
#clear_session()

Hist = mod.fit(x_train,y_train,epochs=50,verbose=1,validation_split=0.25,batch_size=128)
mod.evaluate(x_test,y_test)
mod.save('MNIST_23072019.h5')
accuracy = Hist.history['acc']

val_accuracy = Hist.history['val_acc']

loss = Hist.history['loss']

val_loss = Hist.history['val_loss']

plt.plot(accuracy,label='Max Train Acc.: %.4f'%max(accuracy))

plt.plot(val_accuracy,label='Max Val Acc.: %.4f'%max(val_accuracy))

plt.grid(True)

plt.legend()

plt.show()
plt.plot(loss,label='Min Train loss: %.4f'%min(loss))

plt.plot(val_loss,label='Min Val loss: %.4f'%min(val_loss))

plt.grid(True)

plt.legend()

plt.show()