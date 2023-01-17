import gc

import glob

import os

import cv2

import random

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import imageio as im

from keras import models

from keras.models import Sequential

from keras.layers import Activation, Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.optimizers import adam

from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint

from keras.utils import np_utils

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

import seaborn as sns

import matplotlib

from matplotlib import pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

print(os.listdir("../input"))
# load images dataset

def loadImagesData(glob_path):

    images = []

    names = []

    for img_path in glob.glob(glob_path):

        # load/resize images with cv2

        names.append(os.path.basename(img_path))

        images.append(cv2.resize(cv2.imread(img_path, cv2.IMREAD_COLOR), 

                   (100,100), interpolation=cv2.INTER_CUBIC))

    return (images,names)

# map of training label to list of images

trainData = {}

for label in os.listdir('../input/train/'):

    (images,names) = loadImagesData(f"../input/train/{label}/*.png")

    trainData[label] = images

print("train labels:", ",".join(trainData.keys()))

# show some data

plt.figure(figsize=(5,5))

columns = 5

for i, label in enumerate(trainData.keys()):

    plt.subplot(len(trainData.keys()) / columns + 1, columns, i + 1)

    plt.imshow(trainData[label][0])

plt.show()
# build x/y dataset

trainList = []

for label in trainData.keys():

    for image in trainData[label]:

        trainList.append({

            'label': label,

            'data': image

        })

# shuffle dataset

random.shuffle(trainList)

# dataframe and display

train_df = pd.DataFrame(trainList)

gc.collect()

train_df.head()
# encode training data

data_stack = np.stack(train_df['data'].values)

dfloats = data_stack.astype(np.float32)

all_x = np.multiply(dfloats, 1.0 / 255.0)

all_x.shape
# encode labels

le = LabelEncoder()

le.fit(list(trainData.keys()))

le_y = le.transform(train_df['label'])

# convert to keras categorical one-hot

all_y = np_utils.to_categorical(le_y)

all_y[0:2]
# split test/training data

train_x,test_x,train_y,test_y=train_test_split(all_x,all_y,test_size=0.2,random_state=7)

print(train_x.shape,test_x.shape)
# create the network

num_filters = 8

kernel_size = (10, 10)

input_shape = train_x.shape[1:]

clf = Sequential()

# some models

def simplerNet(clf):

    # a wide-filter cnn of my own design!

    # its not very good, but I like it

    clf.add(Conv2D(num_filters, kernel_size, padding='same', input_shape=input_shape, activation = 'relu'))

    clf.add(MaxPooling2D(pool_size=(2, 2)))

    clf.add(Flatten())

    clf.add(Dense(units = 12, activation = 'softmax'))

def tdsNet(clf):

    # from towards data science keras cnn tutorial

    # performs much better (70% accuracy after 20 epochs)

    clf.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=input_shape))

    clf.add(Conv2D(32, kernel_size=3, activation='relu'))

    clf.add(Flatten())

    clf.add(Dense(units = 12, activation = 'softmax'))

simplerNet(clf)

# show summary

clf.summary()
# compile with same parameters as vanilla cnn

opt = adam(lr=0.0001, decay=1e-6)

clf.compile(optimizer = opt,

            loss = 'categorical_crossentropy', 

            metrics = ['accuracy'])
# data augmenter

# this dataset has many varied sizes and poor centering,

# so resizing and shifting training data helps network

datagen = ImageDataGenerator(

    featurewise_center=False,  # set input mean to 0 over the dataset

    samplewise_center=False,  # set each sample mean to 0

    featurewise_std_normalization=False,  # divide inputs by std of the dataset

    samplewise_std_normalization=False,  # divide each input by its std

    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)

    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

    horizontal_flip=True,  # randomly flip images

    vertical_flip=False)  # randomly flip images

datagen.fit(train_x)
# train model

batch_size = 32

history = clf.fit_generator(datagen.flow(train_x, train_y,

                            batch_size=batch_size),

                            steps_per_epoch= (train_x.shape[0] // batch_size),

                            epochs = 32,

                            validation_data=(test_x, test_y),

                            workers=4)
# plot model metrics from

#  https://stackoverflow.com/questions/51006505/how-training-and-test-data-is-split-keras-on-tensorflow

print(history.history.keys())

# list all data in history

print(history.history.keys())

# summarize history for accuracy

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
# confusion matrix of labels

pre_cls=clf.predict_classes(all_x)    

cm1 = confusion_matrix(le.transform(train_df['label']),pre_cls)

# from https://gist.github.com/shaypal5/94c53d765083101efc0240d776a23823

def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):

    df_cm = pd.DataFrame(

        confusion_matrix, index=class_names, columns=class_names, 

    )

    fig = plt.figure(figsize=figsize)

    try:

        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")

    except ValueError:

        raise ValueError("Confusion matrix values must be integers.")

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)

    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    return fig

class_names = list(le.classes_)

print_confusion_matrix(cm1, class_names)

None
# evaluate data accuracy against split test set

score, acc = clf.evaluate(test_x,test_y)

print('Test score:', score)

print('Test accuracy:', acc)
# load test image datas

(test_images, test_names) = loadImagesData(f"../input/test/*.png")

data_stack = np.stack(test_images)

dfloats = data_stack.astype(np.float32)

unknown_x = np.multiply(dfloats, 1.0 / 255.0)

# predict

predicted = np.argmax(clf.predict(unknown_x), axis=1)

predicted_labels = le.inverse_transform(predicted)

submission_df = pd.DataFrame({'file':test_names,'species':predicted_labels})

submission_df.to_csv('submission.csv', index=False)

len(submission_df)