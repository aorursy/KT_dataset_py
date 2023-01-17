# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
train_data = pd.read_csv('../input/sign_mnist_train.csv') 

train_data.head()
test_data = pd.read_csv('../input/sign_mnist_test.csv') 

test_data.head()
X_train = train_data.drop(columns='label').values

y_train = train_data['label'].values
X_test = test_data.drop(columns='label').values

y_test = test_data['label'].values
X_train = X_train.reshape(len(X_train),28,28,1)

X_test = X_test.reshape(len(X_test),28,28,1)
# data normalization



X_train = X_train / 255.

X_test = X_test / 255.
%matplotlib inline

import matplotlib.pyplot as plt

plt.imshow(X_train[0].reshape(28,28))
from tensorflow import keras

from keras.utils import to_categorical
# one-hot encode target column

y_train = to_categorical(y_train)

y_test = to_categorical(y_test)
# define some callbacks



from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau



class my_max_acc_callback(keras.callbacks.Callback):

  def on_epoch_end(self, epoch, logs={}):

    if(logs.get('acc')>0.99):

      print("\nReached 99% accuracy so cancelling training!")

      self.model.stop_training = True



# my max accuracy callback

my_mxaccb = my_max_acc_callback()



# checkpoint for best weights

filepath='model-best-weights.hdf5'

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')



# early stopping

earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto', restore_best_weights=True)



# reducing_Learning_Rate

reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, min_delta=1e-4, mode='auto')



#collecting all callbacks

callbacks_list = [my_mxaccb, checkpoint, earlyStopping, reduce_lr_loss]
from keras.models import Sequential

from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout



# create model

model = Sequential()



#add model layers

model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))

model.add(MaxPooling2D(2,2))

#

model.add(Conv2D(64, kernel_size=3, activation='relu'))

model.add(MaxPooling2D(2,2))

#

model.add(Dense(256,activation='relu'))

#

model.add(Conv2D(64, kernel_size=3, activation='relu'))

model.add(MaxPooling2D(2,2))

#

model.add(Flatten())

model.add(Dropout(0.2))

#

model.add(Dense(256,activation='relu'))

model.add(Dense(25, activation='softmax'))



model.summary()



model.compile(

    loss = 'categorical_crossentropy', 

    optimizer='adam', 

    metrics=['accuracy'])
history = model.fit(

    X_train, 

    y_train,

    validation_split=0.3,

    epochs=50,

    verbose=1,

    callbacks=callbacks_list)
acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'ro-', label='Training accuracy')

plt.plot(epochs, val_acc, 'bo-', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend(loc=0)

plt.figure()
plt.plot(epochs, loss, 'go-', label='Training loss')

plt.plot(epochs, val_loss, 'yo-', label='Validation loss')

plt.title('Training and validation loss')

plt.legend(loc=0)

plt.figure()
# testing model on test set

test_loss = model.evaluate(X_test, y_test)

print(model.metrics_names)

print(test_loss)