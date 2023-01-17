import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
#import and explore data
path = '../input/card_dataset'
path_train = os.path.join(path,'train')
path_test = os.path.join(path,'test')
print(path_train)

train = pd.read_csv(os.path.join(path,'train_labels.csv'))
test = pd.read_csv(os.path.join(path,'test_labels.csv'))
print(train.shape)
print(test.shape)
print('_' * 49)
print(train.head())
print('_' * 100)
print(test.head())
print(train['class'].unique())
#display crop image
for i in range(10):
    row = train.iloc[i]
#     print(path_train)
    img = cv2.imread(os.path.join(path_train, row['filename']))
#     print(img.shape)
    xmin = row['xmin']
    xmax = row['xmax']
    ymin = row['ymin']
    ymax = row['ymax']
    crop_img = img[ymin:ymax, xmin:xmax]
    print(crop_img.shape)
    plt.imshow(crop_img)
    plt.show()

height_crop_image = 250
weight_crop_image = 200
def create_data():
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    classes = ['queen', 'ten', 'nine', 'king', 'jack', 'ace']
    
    m_train = train.shape[0]
    m_test = test.shape[0]
    
    for i in range(m_train):
        row = train.iloc[i]
        img = cv2.imread(os.path.join(path_train, row['filename']))
        xmin = row['xmin']
        xmax = row['xmax']
        ymin = row['ymin']
        ymax = row['ymax']
        c = row['class']
        
        y_train.append(classes.index(c))
        crop_img = img[ymin:ymax, xmin:xmax]
        crop_img = cv2.resize(crop_img, (weight_crop_image, height_crop_image))
        x_train.append(crop_img)
    
    for i in range(m_test):
        row = test.iloc[i]
        img = cv2.imread(os.path.join(path_test, row['filename']))
        xmin = row['xmin']
        xmax = row['xmax']
        ymin = row['ymin']
        ymax = row['ymax']
        c = row['class']
        
        y_test.append(classes.index(c))
        crop_img = img[ymin:ymax, xmin:xmax]
        crop_img = cv2.resize(crop_img, (weight_crop_image, height_crop_image))
        x_test.append(crop_img)
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    return x_train, y_train, x_test, y_test, classes
 
x_train_org, y_train_org, x_test_org, y_test_org, classes = create_data()
print(x_train_org.shape)
print(y_train_org.shape)
print(x_test_org.shape)
print(y_test_org.shape)
print(classes)
from tensorflow.keras.utils import to_categorical

x_train = x_train_org / 255.0
x_test = x_test_org / 255.0
y_train = to_categorical(y_train_org, 6)
y_test = to_categorical(y_test_org, 6)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras import regularizers
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras import regularizers
from keras.callbacks import LearningRateScheduler
from keras import backend as K
import keras
import matplotlib.pyplot as plt
weight_decay = 1e-4
model = Sequential()
model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
 
model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))
 
model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))
 
model.add(Flatten())
model.add(Dense(6, activation='softmax'))
opt_rms = keras.optimizers.rmsprop(lr=0.001,decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt_rms, metrics=['accuracy'])
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

# Split the train and the validation set for the fitting
# X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1)
datagen.fit(x_train)
model.fit_generator(
    datagen.flow(x_train, y_train),
    epochs=30,
    validation_data = (x_test, y_test), 
    steps_per_epoch=x_train.shape[0]
)
