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
from subprocess import check_output
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.callbacks import LearningRateScheduler
num_classes = 10
random_seed = 10
train_df = pd.read_csv('/kaggle/input/fashion-mnist_train.csv',sep=',')
test_df = pd.read_csv('/kaggle/input/fashion-mnist_test.csv', sep = ',')
train_df.head()
test_df.head()
train_data = np.array(train_df, dtype = 'float32')
test_data = np.array(test_df,dtype='float32')
x_train = train_data[:,1:]/255
y_train = train_data[:,0]
x_test = test_data[:,1:]/255
y_test = test_data[:,0]
image_rows = 28

image_cols = 28

batch_size = 512

image_shape = (1,image_rows,image_cols)
x_train = x_train.reshape(x_train.shape[0],*image_shape)
x_test = x_test.reshape(x_test.shape[0],*image_shape)

x_test.shape
nn = 5
model = [0]*nn

for j in range(nn):
    model[j] = Sequential()
    model[j].add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(1,28,28), activation='relu',data_format='channels_first'))
    model[j].add(BatchNormalization())
    model[j].add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Dropout(0.4))
    
    model[j].add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Dropout(0.4))
    
    model[j].add(Conv2D(128, kernel_size = 4, activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Flatten())
    model[j].add(Dropout(0.4))
    model[j].add(Dense(10, activation='softmax'))
    
    # Compile model
    model[j].compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# With data augmentation to prevent overfitting
datagen = ImageDataGenerator(
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        vertical_flip=False)  # randomly flip images


datagen.fit(x_train)
history = [0] * nn
epochs = 50

# Fit the model
for j in range(nn):
    x_train2, x_val, y_train2, y_val = train_test_split(x_train, y_train, test_size = 0.10, random_state=random_seed)
    history[j] = model[j].fit_generator(datagen.flow(x_train2,y_train2, batch_size=64),
                              epochs = epochs, validation_data = (x_val,y_val),
                              verbose = 0, steps_per_epoch=(len(x_train)//64),validation_steps=(len(x_val)//64))
    print("CNN {0:d}: Epochs={1:d}, Train accuracy={2:.2f}, Validation accuracy={3:.2f}".format(
        j+1,epochs,max(history[j].history['acc']),max(history[j].history['val_acc']) ))
score = [0]*nn

for j in range(nn):
    score[j] = model[j].evaluate(x_test,y_test)
np_score = np.array(score)
# axis = 0 means calculating across columns
np_sum = np.sum(np_score,axis =0)
avg_loss = (np_sum[0]/len(np_score))
avg_accuracy = (np_sum[1]/len(np_score))
print('Avg test loss: ', avg_loss)
print('Avg test accuracy: ',avg_accuracy)
