# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 



np.random.seed(1987)



from keras.datasets import mnist  #data set

from keras.models import Sequential

from keras.layers.core import Dense,Dropout,Activation

from keras.optimizers import SGD,Adam,RMSprop

from keras.utils import np_utils



# X_train shape (60000,28,28)

# X_test shape (10000,28,28)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# plot 4 images as gray scale

plt.subplot(221)

plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))

plt.subplot(222)

plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))

plt.subplot(223)

plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))

plt.subplot(224)

plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))

# show some samples 

plt.show()
# reshape  input output

X_train = X_train.reshape(60000, 784)

X_test = X_test.reshape(10000, 784)

X_train = X_train.astype('float32')

X_test = X_test.astype('float32')



#  normalization

X_train /= 255

X_test /= 255

print(X_train.shape[0], 'train samples')

print(X_test.shape[0], 'test samples')
#model parameters

batch_size=64

nb_classes=10

nb_epoch=20



# convert class vectors to binary class matrices

Y_train = np_utils.to_categorical(y_train, nb_classes)

Y_test = np_utils.to_categorical(y_test, nb_classes)
#build model

model = Sequential() 

model.add(Dense(512, input_shape=(784,))) 

model.add(Activation('relu')) 

model.add(Dropout(0.2)) 

model.add(Dense(512)) 

model.add(Activation('relu')) 

model.add(Dropout(0.2)) 

model.add(Dense(10)) 

model.add(Activation('softmax'))



model.summary()

#compile

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
#fit model

history = model.fit(X_train, Y_train,

                    batch_size=batch_size, nb_epoch=nb_epoch,

                    verbose=1, validation_data=(X_test, Y_test))
#评估模型

score = model.evaluate(X_test, Y_test, verbose=0)

print('Test score:', score[0])

print('Test accuracy:', score[1])
model.save('mnist-mpl.h5')