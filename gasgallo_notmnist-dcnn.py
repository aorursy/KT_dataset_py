import numpy as np
import os
import sys
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
from PIL import Image

import keras
from keras.preprocessing.image import load_img
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Activation, BatchNormalization
from keras.layers import Reshape, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.utils import np_utils
from keras.regularizers import l1_l2
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

%matplotlib inline
%env JOBLIB_TEMP_FOLDER=/tmp
dataset = 'notMNIST_large'
DATA_PATH = '../input/' + dataset + '/' + dataset

test = 'notMNIST_small'
TEST_PATH = '../input/' + dataset + '/' + dataset
max_images = 100
grid_width = 10
grid_height = int(max_images / grid_width)
fig, axs = plt.subplots(grid_height, grid_width, figsize=(grid_width, grid_height))
classes = os.listdir(DATA_PATH)
for j, cls in enumerate(classes):
    figs = os.listdir(DATA_PATH + '/' + cls)
    for i, fig in enumerate(figs[:grid_width]):
        ax = axs[j, i]
        ax.imshow(np.array(load_img(DATA_PATH + '/' + cls + '/' + fig)))
        ax.set_yticklabels([])
        ax.set_xticklabels([])
X = []
labels = []
# for each folder (holding a different set of letters)
for directory in os.listdir(DATA_PATH):
    # for each image
    for image in os.listdir(DATA_PATH + '/' + directory):
        # open image and load array data
        try:
            file_path = DATA_PATH + '/' + directory + '/' + image
            img = Image.open(file_path)
            img.load()
            img_data = np.asarray(img, dtype=np.int16)
            # add image to dataset
            X.append(img_data)
            # add label to labels
            labels.append(directory)
        except:
            None # do nothing if couldn't load file
N = len(X) # number of images
img_size = len(X[0]) # width of image
X = np.asarray(X).reshape(N, img_size, img_size,1) # add our single channel for processing purposes
labels_cat = to_categorical(list(map(lambda x: ord(x)-ord('A'), labels)), 10) # convert to one-hot
labels = np.asarray(list(map(lambda x: ord(x)-ord('A'), labels)))

X_test = []
y_test = []
# for each folder (holding a different set of letters)
for directory in os.listdir(TEST_PATH):
    # for each image
    for image in os.listdir(TEST_PATH + '/' + directory):
        # open image and load array data
        try:
            file_path = DATA_PATH + '/' + directory + '/' + image
            img = Image.open(file_path)
            img.load()
            img_data = np.asarray(img, dtype=np.int16)
            # add image to dataset
            X_test.append(img_data)
            # add label to labels
            y_test.append(directory)
        except:
            None # do nothing if couldn't load file
N = len(X_test) # number of images
img_size = len(X_test[0]) # width of image
X_test = np.asarray(X_test).reshape(N, img_size, img_size,1) # add our single channel for processing purposes
y_test_cat = to_categorical(list(map(lambda x: ord(x)-ord('A'), y_test)), 10) # convert to one-hot
y_test = np.asarray(list(map(lambda x: ord(x)-ord('A'), y_test)))
cls_s = np.sum(labels,axis=0)

fig, ax = plt.subplots()
plt.bar(np.arange(10), cls_s)
plt.ylabel('No of pics')
plt.xticks(np.arange(10), np.sort(classes))
plt.title('Checking balance for data set..')
plt.show()
from sklearn.cross_validation import train_test_split
X_train,X_valid,y_train,y_valid=train_test_split(X,labels,test_size=0.2)
X_train_cat,X_valid_cat,y_train_cat,y_valid_cat=train_test_split(X,labels_cat,test_size=0.2)

print('Training:', X_train.shape, y_train.shape)
print('Validation:', X_valid.shape, y_valid.shape)
print('Test:', X_test.shape, y_test.shape)
fig, axs = plt.subplots(grid_height, grid_width, figsize=(grid_width, grid_height))
for j in range(max_images):
    ax = axs[int(j/grid_width), j%grid_width]
    ax.imshow(X_train[j,:,:,0])
    ax.set_yticklabels([])
    ax.set_xticklabels([])
# helper functions
def plot_training_curves(history):
    """
    Plot accuracy and loss curves for training and validation sets.
    Args:
        history: a Keras History.history dictionary
    Returns:
        mpl figure.
    """
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(8,2))
    if 'acc' in history:
        ax_acc.plot(history['acc'], label='acc')
        if 'val_acc' in history:
            ax_acc.plot(history['val_acc'], label='Val acc')
        ax_acc.set_xlabel('epoch')
        ax_acc.set_ylabel('accuracy')
        ax_acc.legend(loc='upper left')
        ax_acc.set_title('Accuracy')

    ax_loss.plot(history['loss'], label='loss')
    if 'val_loss' in history:
        ax_loss.plot(history['val_loss'], label='Val loss')
    ax_loss.set_xlabel('epoch')
    ax_loss.set_ylabel('loss')
    ax_loss.legend(loc='upper right')
    ax_loss.set_title('Loss')

    sns.despine(fig)
    return

# parameters
batch_size = 128
nb_classes = 10
nb_epoch = 200
input_dim = 784
resolution = 28
reg = l1_l2(l1=0, l2=0.02)
# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 64
# size of pooling area for max pooling
pool_size = [2, 2]
# convolution kernel size
kernel_size = [3, 3]

input_shape = [img_rows, img_cols, 1]
# define path to save model
model_path = './cnn_notMNIST.h5'
# prepare callbacks
callbacks = [
    EarlyStopping(
        monitor='val_acc', 
        patience=20,
        mode='max',
        verbose=1),
    ModelCheckpoint(model_path,
        monitor='val_acc', 
        save_best_only=True, 
        mode='max',
        verbose=1),
    ReduceLROnPlateau(
        factor=0.1, 
        patience=5, 
        min_lr=0.00001, 
        verbose=1)
]

# model layers
cnn_ad = Sequential()
cnn_ad.add(Conv2D(nb_filters, kernel_size, padding='same', input_shape=input_shape))
cnn_ad.add(Activation('relu'))
cnn_ad.add(BatchNormalization())
cnn_ad.add(Conv2D(nb_filters, kernel_size, padding='same'))
cnn_ad.add(Activation('relu'))
cnn_ad.add(BatchNormalization())
cnn_ad.add(MaxPooling2D(pool_size=pool_size))
cnn_ad.add(Dropout(0.25))
cnn_ad.add(Flatten())
cnn_ad.add(Dense(128, kernel_regularizer=reg))
cnn_ad.add(Activation('relu'))
cnn_ad.add(BatchNormalization())
cnn_ad.add(Dropout(0.5))
cnn_ad.add(Dense(nb_classes, kernel_regularizer=reg))
cnn_ad.add(Activation('softmax'))

cnn_ad.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

history = cnn_ad.fit(X_train_cat, y_train_cat,
                 batch_size=batch_size, epochs=nb_epoch,
                 verbose=0, validation_data=(X_valid_cat, y_valid_cat),
                 shuffle=True, callbacks=callbacks)
score = cnn_ad.evaluate(X_test, y_test_cat, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

plot_training_curves(history.history)