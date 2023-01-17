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
import keras
from keras.layers import Input, Conv2D, Dense, Dropout, MaxPool2D, Flatten
from keras.models import Model, Sequential
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
train_df = pd.read_csv('../input/train.csv')
train_df.head()
test_df = pd.read_csv('../input/test.csv')
test_df.head()
train_df.info()
x_train = train_df.iloc[:,1:].values
y_train = train_df.iloc[:,0].values
print(x_train.shape, y_train.shape)
x_test = test_df.iloc[:,:].values
#y_test = test_df.iloc[:,0].values
print(x_test.shape)
# reshape input
x_train = x_train.reshape(x_train.shape[0], 28, 28)
x_test = x_test.reshape(x_test.shape[0], 28, 28)
print(x_train.shape, x_test.shape)
def visual(x_input, y):
    if len(x_input.shape) != 3:
        x = x_input.reshape(x_input.shape[0], 28, 28)
    else:
        x = x_input
    fig = plt.figure()
    for i in range(len(x)):
      plt.subplot(3,3,i+1)
      plt.tight_layout()
      plt.imshow(x[i], cmap='gray', interpolation='none')
      plt.title("Digit: {}".format(y[i]))
      plt.xticks([])
      plt.yticks([])
    #return fig
visual(x_train[:9], y_train[:9])
def pre_process(x):
    x = x.reshape(x.shape[0], 28, 28, 1)
    x = x.astype('float32')
    x = x/255
    
    return x
x_train = pre_process(x_train)
y_train = to_categorical(y_train, num_classes=len(np.unique(y_train)))
print(x_train.shape, y_train.shape)
# split to validation
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=91)
print('Train:', x_train.shape, y_train.shape)
print('Val:', x_val.shape, y_val.shape)
def get_model(input_shape, nb_class):
    model = Sequential()
    model.add(Conv2D(64, (3,3), input_shape=input_shape, activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_class, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    model.summary()
    
    return model    
input_shape = (x_train.shape[1], x_train.shape[2], 1)
nb_class = 10
model = get_model(input_shape, nb_class)
# train
batch_size = 128
epochs = 15
model_log = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_val, y_val))
score = model.evaluate(x_val, y_val, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
def visual_hist():
    fig = plt.figure()
    plt.subplot(2,1,1)
    plt.plot(model_log.history['acc'])
    plt.plot(model_log.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')
    plt.subplot(2,1,2)
    plt.plot(model_log.history['loss'])
    plt.plot(model_log.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.tight_layout()
visual_hist()
import datetime
x_test = pre_process(x_test)
sub_df = pd.read_csv('../input/sample_submission.csv')
sub_df.info()
y_test = model.predict(x_test)
y_test.shape
y_result = np.argmax(y_test, axis=1)
print(y_test[91], y_result[91])
sub_df.Label = y_result
sub_df.head()
#visual(x_test[91:101], y_result[91:101])
sub_df.head()
current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%s')
file_name = 'submission_' + current_time +'.csv'
sub_df.to_csv(file_name, index=False)
!ls
df = pd.read_csv(file_name)
df.head()
