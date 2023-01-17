import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split



import matplotlib.pyplot as plt

%matplotlib inline



# let's keep our keras backend tensorflow quiet

import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'



import keras

from keras.models import Sequential

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Dense, Dropout, Lambda, Flatten, Conv2D, MaxPool2D

from keras.layers.normalization import BatchNormalization

from keras.utils.np_utils import to_categorical



from keras.datasets import mnist
import os



(X_train, y_train), (X_test, y_test) = mnist.load_data(path=os.path.abspath('../input/mnist-numpy/mnist.npz'))



# building the input vector from the 28x28 pixels

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)

X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train = X_train.astype('float32')

X_test = X_test.astype('float32')



# normalizing the data to help with the training

X_train /= 255

X_test /= 255



# print the final input shape ready for training

print("Train matrix shape", X_train.shape)

print("Test matrix shape", X_test.shape)
Y_train = y_train.astype('int32') 

Y_test = y_test.astype('int32') 



# one-hot encoding

n_classes = 10

Y_train = to_categorical(Y_train, n_classes)

Y_test = to_categorical(Y_test, n_classes)
model= Sequential()



model.add(Conv2D(filters = 128, kernel_size = (5,5),padding = 'Same', activation ='relu', kernel_initializer='he_normal', input_shape = (28,28,1)))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(BatchNormalization())

model.add(Conv2D(filters = 128, kernel_size = (5,5),padding = 'Same', activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(BatchNormalization())

model.add(Dropout(0.25))



model.add(Conv2D(filters = 256, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(BatchNormalization())

model.add(Conv2D(filters = 256, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(BatchNormalization())

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.33))

model.add(Dense(10, activation='softmax'))



model.compile(

    optimizer = 'rmsprop', # 'rmsprop', 'adam'

    loss = 'categorical_crossentropy',

    metrics=['accuracy'])
from keras.callbacks import EarlyStopping, ModelCheckpoint



# Set callback functions to early stop training and save the best model so far

callbacks = [EarlyStopping(monitor='val_loss', patience=20),

         ModelCheckpoint(filepath=os.path.abspath('best_model.h5'), monitor='val_loss', save_best_only=True)]



h = model.fit(

        X_train,

        Y_train, 

        callbacks=callbacks,

        batch_size=256,

        epochs = 50, 

        validation_data=(X_test, Y_test),

        verbose = 1)
model.load_weights('best_model.h5')

%rm best_model.h5



final_loss, final_acc = model.evaluate(X_test, Y_test, verbose=0)

print("Final loss: {0:.6f}, final accuracy: {1:.6f}".format(final_loss, final_acc))
loss = h.history['loss']

val_loss = h.history['val_loss']



epochs = range(len(loss))

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()
predicted_classes = model.predict_classes(X_test)



submissions = pd.DataFrame({"ImageId": list(range(1, len(predicted_classes) + 1)), "Label": predicted_classes})

submissions.to_csv("submission.csv", index=False, header=True)