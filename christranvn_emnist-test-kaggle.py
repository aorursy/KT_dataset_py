# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


def import_data(file):
    """create a dataframe and optimize its memory usage"""
    df = pd.read_csv(file, parse_dates=True, keep_date_col=True)
    df = reduce_mem_usage(df)
    return df
# Import modules
import tensorflow as tf
import matplotlib.pyplot as plt
import math

from sklearn.model_selection import train_test_split

#keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
import sklearn.metrics as metrics


train = import_data(r'../input/emnist/emnist-bymerge-train.csv')
test = import_data(r'../input/emnist/emnist-bymerge-test.csv')

#mapp = pd.read_csv(
 #   r'../input/emnist/emnist-bymerge-mapping.txt',
 #   delimiter=' ',
 #   index_col=0,
 #   header=None,
 #   squeeze=True
#)

print("Train: %s, Test: %s" %(train.shape, test.shape))
# Constants
HEIGHT = 28
WIDTH = 28
# Split x and y
train_x = train.iloc[:,1:] # Get the images
train_y = train.iloc[:,0] # Get the label
del train # free up some memory

test_x = test.iloc[:,1:]
test_y = test.iloc[:,0]
del test
# Reshape and rotate EMNIST images
def rotate(image):
    image = image.reshape(HEIGHT, WIDTH)
    image = np.fliplr(image)
    image = np.rot90(image)
    return image 
# Flip and rotate image
train_x = np.asarray(train_x)
train_x = np.apply_along_axis(rotate, 1, train_x)
print ("train_x:",train_x.shape)

test_x = np.asarray(test_x)
test_x = np.apply_along_axis(rotate, 1, test_x)
print ("test_x:",test_x.shape)
# Normalize
train_x = train_x / 255.0
test_x = test_x / 255.0
print(type(train_x[0,0,0]))
print(type(test_x[0,0,0]))
# Plot image
'''
for i in range(100,109):
  plt.subplot(330 + (i+1))
  plt.subplots_adjust(hspace=0.5, top=1)
  plt.imshow(train_x[i], cmap=plt.get_cmap('gray'))
  plt.title(chr(mapp[train_y[i]]))
'''
# Number of classes
num_classes = train_y.nunique() # .nunique() returns the number of unique objects
print(num_classes) 
# One hot encoding
train_y = to_categorical(train_y, num_classes)
test_y = to_categorical(test_y, num_classes)
print("train_y: ", train_y.shape)
print("test_y: ", test_y.shape)
# partition to train and val
train_x, val_x, train_y, val_y = train_test_split(train_x, 
                                                  train_y, 
                                                  test_size=0.10, 
                                                  random_state=7)

print(train_x.shape, val_x.shape, train_y.shape, val_y.shape)
# Reshape
train_x = train_x.reshape(-1, HEIGHT, WIDTH, 1)
test_x = test_x.reshape(-1, HEIGHT, WIDTH, 1)
val_x = val_x.reshape(-1, HEIGHT, WIDTH, 1)
# Create more images via data augmentation
datagen = ImageDataGenerator(
    rotation_range = 10,
    zoom_range = 0.10,
    width_shift_range=0.1,
    height_shift_range=0.1
)
# Building model
# ((Si - Fi + 2P)/S) + 1

model = Sequential()

model.add(Conv2D(32, kernel_size=3, 
                 activation='relu', input_shape=(HEIGHT, WIDTH, 1)))
#model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=3,activation='relu'))
#model.add(AveragePooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
#model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32, kernel_size=5, strides=2, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=5, strides=2, padding='same', activation='relu'))
#model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(128, kernel_size=4, activation='relu'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(units=num_classes, activation='softmax'))

input_shape = (None, HEIGHT, WIDTH, 1)
model.build(input_shape)
model.summary()

my_callbacks = [
    # Decrease learning rate
    LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x),
    # Training will stop there is no improvement in val_loss after 3 epochs
    EarlyStopping(monitor="val_acc", 
                  patience=3, 
                  mode='max', 
                  restore_best_weights=True)
]

# TRAIN NETWORKS
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(train_x, train_y, 
                    epochs=40,
                    verbose=1, validation_data=(val_x, val_y), 
                    callbacks=my_callbacks)
# plot accuracy and loss
'''
def plotgraph(epochs, acc, val_acc):
    # Plot training & validation accuracy values
    plt.plot(epochs, acc, 'b')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()
'''

#%%
'''
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(acc)+1)
'''
# Accuracy curve
#plotgraph(epochs, acc, val_acc)

# loss curve
#plotgraph(epochs, loss, val_loss)
score = model.evaluate(test_x, test_y, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
model.save("emnist_model.h5")
model.save_weights("emnist_model_weights.h5")
y_pred = model.predict(test_x)
y_pred = (y_pred > 0.5)

cm = metrics.confusion_matrix(test_y.argmax(axis=1), y_pred.argmax(axis=1))
print(cm)

