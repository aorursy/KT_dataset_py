import numpy as np
import pandas as pd
df_train = pd.read_csv('../input/fashion-mnist_train.csv')
df_test = pd.read_csv('../input/fashion-mnist_test.csv')
df_train.head()
df_train.shape
df_test.shape
df_train.label.unique()
x_train = df_train.drop('label',1)
# print (x_train.head)
y_train = df_train['label']
print (y_train.head(10))
print (type(df_train))
print (type(df_train))
x_train = x_train.as_matrix()
print (type(x_train))
print (x_train.shape)
y_train = y_train.as_matrix()
print (type(y_train))
print (y_train.shape)
x_train_partial = x_train[:40000]
x_train_val = x_train[40000:]
y_train_partial = y_train[:40000]
y_train_val = y_train[40000:]
n_classes = len(np.unique(y_train))
n_classes
batch_size = int(x_train_partial.shape[0]*0.01)
batch_size
from keras.utils.np_utils import to_categorical
y_train_partial = to_categorical(y_train_partial)
y_train_val = to_categorical(y_train_val)
from keras import models
from keras import layers
model = models.Sequential()
model.add(layers.Dense(32,activation='relu',input_shape=(784,)))
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(n_classes,activation='softmax'))
model.summary()
from keras.callbacks import EarlyStopping
earlystop = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=5, \
                          verbose=1, mode='auto')
callbacks_list = [earlystop]

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc'])

history = model.fit(x_train_partial,
                    y_train_partial,
                    epochs=50,
                    batch_size=batch_size,
                    callbacks=callbacks_list,
                    validation_data=(x_train_val, y_train_val))
import matplotlib.pyplot as plt
%matplotlib inline

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1,len(history.epoch)+1) # creating a range object for plotting on x axis

plt.plot(epochs, loss_values, 'ro', label='Training loss')           
plt.plot(epochs, val_loss_values, 'b--', label='Validation loss')      
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
###### Let us plot the training and validation accuracy 
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc_values, 'ro', label='Training acc')
plt.plot(epochs, val_acc_values, 'b--', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
df_test.head()
df_test.shape
x_test = df_test.drop('label',1)
y_test = df_test['label']
x_test = x_test.as_matrix()
y_test = y_test.as_matrix()
print (type(x_test))
print (type(y_test))
print (x_test.shape)
print (y_test.shape)
y_test = to_categorical(y_test)
results = model.evaluate(x_test,y_test)
results
from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,initializers
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras import backend as K
batch_size = 512
num_classes = 10
# input image dimensions
im_rows, im_cols = 28, 28
x_train_partial = x_train[:40000]
x_train_val = x_train[40000:]
y_train_partial = y_train[:40000]
y_train_val = y_train[40000:]
# theano and tensorflow backends take the images differently , the code below is to handle that
if K.image_data_format() == 'channels_first':
    x_train_partial = x_train_partial.reshape(x_train_partial.shape[0], 1, im_rows, im_cols)
    x_train_val = x_train_val.reshape(x_train_val.shape[0], 1, im_rows, im_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, im_rows, im_cols)
    input_shape = (1, im_rows, im_cols)
else:
    x_train_partial = x_train_partial.reshape(x_train_partial.shape[0],  im_rows, im_cols,1)
    x_train_val = x_train_val.reshape(x_train_val.shape[0], im_rows, im_cols,1)
    x_test = x_test.reshape(x_test.shape[0], im_rows, im_cols,1)
    input_shape = (im_rows, im_cols,1)
x_train_partial = x_train_partial.astype('float32')
x_train_val = x_train_val.astype('float32')
x_test = x_test.astype('float32')

# x_train_partial /= 255 # normalizing 
# x_train_partial /= 255 # normalizing 
# x_test /= 255 # normalizing

print('x_train shape:', x_train_partial.shape)
print(x_train_partial.shape[0], 'train samples')

print('x_train_partial shape:', x_train_val.shape)
print(x_train_val.shape[0], 'train validation samples')

print('x_test shape:', x_test.shape)
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train_partial = keras.utils.to_categorical(y_train_partial, num_classes)
y_train_val = keras.utils.to_categorical(y_train_val, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
import gc
gc.collect()
model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
# model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Dropout(0.2))
model.add(Flatten())

model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.1))
model.add(Dense(num_classes, activation='softmax'))
earlystop_acc = EarlyStopping('val_acc', min_delta=0.001, patience=10, \
                          verbose=1, mode='auto')
earlystop_val = EarlyStopping('val_loss', min_delta=0.001, patience=10, \
                          verbose=1, mode='auto')
filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [earlystop_acc,earlystop_val,checkpoint]
model.summary()
epochs =50
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc'])

history = model.fit(x_train_partial,
                    y_train_partial,
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=callbacks_list,
                    validation_data=(x_train_val, y_train_val))
import matplotlib.pyplot as plt

%matplotlib inline

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1,len(history.epoch)+1) # creating a range object for plotting on x axis

plt.plot(epochs, loss_values, 'ro', label='Training loss')           
plt.plot(epochs, val_loss_values, 'b--', label='Validation loss')      
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc_values, 'ro', label='Training acc')
plt.plot(epochs, val_acc_values, 'b--', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
y_test = df_test['label']
y_test = y_test.as_matrix()
y_test = keras.utils.to_categorical(y_test, num_classes)
results = model.evaluate(x_test,y_test)
results
