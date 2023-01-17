# IMPORTING NECESSARY MODULES FOR DATA ANALYSIS AND PREDICTIVE MODELLING

import numpy as np

import pandas as pd

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score,confusion_matrix

from sklearn.model_selection import train_test_split

import re

import gc

import os

import cv2

import argparse

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Flatten

from keras.layers import Dropout

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.utils import np_utils

from keras import optimizers

import psutil

import humanize

from sklearn.model_selection import KFold, StratifiedKFold

from tqdm import tqdm

import matplotlib.pyplot as plt

from IPython.display import HTML, display, clear_output

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)
print(os.listdir("../input"))
TrainDataPath = '../input/train.csv'

TestDataPath = '../input/test.csv' 

SubDataPath = '../input/sample_submission.csv'



# Loading the Training and Test Dataset and Submission File

TrainData = pd.read_csv(TrainDataPath)

TestData = pd.read_csv(TestDataPath)

SubData = pd.read_csv(SubDataPath)
print("Training Dataset Shape:")

print(TrainData.shape)

TrainData.head()
print("Test Dataset Shape:")

print(TestData.shape)

TestData.head()
print("Submission Dataset Shape:")

print(SubData.shape)

print("\n")

print("Submission Dataset Columns/Features:")

print(SubData.dtypes)

SubData.head()
# checking missing data percentage in train data

total = TrainData.isnull().sum().sort_values(ascending = False)

percent = (TrainData.isnull().sum()/TrainData.isnull().count()*100).sort_values(ascending = False)

missing_TrainData  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_TrainData.head(10) # We are just doing a sanity check that each and every pixel value is present or not!!!
# checking missing data percentage in test data

total = TestData.isnull().sum().sort_values(ascending = False)

percent = (TestData.isnull().sum()/TestData.isnull().count()*100).sort_values(ascending = False)

missing_TrainData  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_TrainData.head(10) # Doing the same thing again for test data
def printmemusage():

 process = psutil.Process(os.getpid())

 print("Gen RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ), " | Proc size: " + humanize.naturalsize( process.memory_info().rss))



printmemusage()
def plot_bar_counts_categorical(data_se, title, figsize, sort_by_counts=False):

    info = data_se.value_counts()

    info_norm = data_se.value_counts(normalize=True)

    categories = info.index.values

    counts = info.values

    counts_norm = info_norm.values

    fig, ax = plt.subplots(figsize=figsize)

    if data_se.dtype in ['object']:

        if sort_by_counts == False:

            inds = categories.argsort()

            counts = counts[inds]

            counts_norm = counts_norm[inds]

            categories = categories[inds]

        ax = sns.barplot(counts, categories, orient = "h", ax=ax)

        ax.set(xlabel="count", ylabel=data_se.name)

        ax.set_title("Distribution of " + title)

        for n, da in enumerate(counts):

            ax.text(da, n, str(da)+ ",  " + str(round(counts_norm[n]*100,2)) + " %", fontsize=10, va='center')

    else:

        inds = categories.argsort()

        counts_sorted = counts[inds]

        counts_norm_sorted = counts_norm[inds]

        ax = sns.barplot(categories, counts, orient = "v", ax=ax)

        ax.set(xlabel=data_se.name, ylabel='count')

        ax.set_title("Distribution of " + title)

        for n, da in enumerate(counts_sorted):

            ax.text(n, da, str(da)+ ",  " + str(round(counts_norm_sorted[n]*100,2)) + " %", fontsize=10, ha='center')
def count_plot_by_hue(data_se, hue_se, title, figsize, sort_by_counts=False):

    if sort_by_counts == False:

        order = data_se.unique()

        order.sort()

    else:

        order = data_se.value_counts().index.values

    off_hue = hue_se.nunique()

    off = len(order)

    fig, ax = plt.subplots(figsize=figsize)

    ax = sns.countplot(y=data_se, hue=hue_se, order=order, ax=ax)

    ax.set_title(title)

    patches = ax.patches

    for i, p in enumerate(ax.patches):

        x=p.get_bbox().get_points()[1,0]

        y=p.get_bbox().get_points()[:,1]

        total = x

        p = i

        q = i

        while(q < (off_hue*off)):

            p = p - off

            if p >= 0:

                total = total + (patches[p].get_bbox().get_points()[1,0] if not np.isnan(patches[p].get_bbox().get_points()[1,0]) else 0)

            else:

                q = q + off

                if q < (off*off_hue):

                    total = total + (patches[q].get_bbox().get_points()[1,0] if not np.isnan(patches[q].get_bbox().get_points()[1,0]) else 0)

       

        perc = str(round(100*(x/total), 2)) + " %"

        

        if not np.isnan(x):

            ax.text(x, y.mean(), str(int(x)) + ",  " + perc, va='center')

    plt.show()
def show_unique(data_se):

    display(HTML('<h5><font color="green"> Shape Of Dataset Is: ' + str(data_se.shape) + '</font></h5>'))

    for i in data_se.columns:

        if data_se[i].nunique() == data_se.shape[0]:

            display(HTML('<font color="red"> ATTENTION!!! ' + str(i+' --> '+str(data_se[i].nunique())) + '</font>'))

        elif (data_se[i].nunique() == 1):

            display(HTML('<font color="Blue"> ATTENTION!!! ' + str(i+' --> '+str(data_se[i].nunique())) + '</font>'))

        else:

            print(i+' -->', data_se[i].nunique())
def show_countplot(data_se):

    display(HTML('<h2><font color="blue"> Dataset CountPlot Visualization: </font></h2>'))

    for i in data_se.columns:

        if (data_se[i].nunique() <= 10):

            plot_bar_counts_categorical(data_se[i].astype(str), 'Dataset Column: '+ i, (15,7))

        elif (data_se[i].nunique() > 10 and data_se[i].nunique() <= 20):

            plot_bar_counts_categorical(data_se[i].astype(str), 'Dataset Column: '+ i, (15,12))

        else:

            print('Columns do not fit in display '+i+' -->', data_se[i].nunique())
gc.collect() # Python garbage collection module for dereferencing the memory pointers and making memory available for better usage
TrainData.head()
TestData.head()
print(TrainData.shape) 

print(TestData.shape)
plot_bar_counts_categorical(TrainData['label'], 'Train Dataset "Target Variable" Distribution Plot', figsize=(18,5), sort_by_counts=False)
train_img = np.array(TrainData.drop(['label'], axis=1))

train_labels = np.array(TrainData['label'])



test_img = np.array(TestData)
train_img
train_labels
test_img
img_rows, img_cols = 28, 28 # Because we have 784 columns i.e. it is an image with size = 28x28

total_classes = 10

train_labels = np_utils.to_categorical(train_labels, 10) # Converting each label into one hot encoded label
train_labels
X_train, X_val, y_train, y_val = train_test_split(train_img, train_labels, test_size=0.1)  # Splitting the dataset into train and validation set
X_train_new = X_train.reshape(X_train.shape[0],28,28,1)

X_val_new = X_val.reshape(X_val.shape[0],28,28,1)
def CNN(height, width, depth, total_classes):

    # Model Initialize

    model = Sequential()

    # First layer Convultion2D ---> Relu Activation ---> MaxPooling2D

    # The border_mode = "same", you get an output that is the "same" size as the input.

    model.add(Conv2D(20, (3,3), activation='relu', input_shape=(width, height, depth), padding="same"))

    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    # Second layer Convultion2D ---> Relu Activation ---> MaxPooling2D

    model.add(Conv2D(50, (3,3), activation='relu', padding="same"))

    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    # Third layer includes flattening the image into a columm of pixel values

    model.add(Flatten())

    model.add(Dense(100, activation='relu'))

    # Fourth layer the output layer

    model.add(Dense(total_classes, activation='softmax')) 

    model.summary()

    return model
print('Now Setting the Optimizer....\n')

sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

model = CNN(img_rows, img_cols, 1, total_classes)

model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
print(X_train_new.shape)

print(X_val_new.shape)
print(y_train.shape)

print(y_val.shape)
history = model.fit(X_train_new / 255.0 , y_train, batch_size=128, epochs=5, verbose=1)
loss, accuracy = model.evaluate(X_val_new / 255.0, y_val, batch_size=128, verbose=1)

print('Accuracy of Model on Validation Set: {:.2f}%'.format(accuracy * 100))
# list all data in history

print(history.history.keys())
# summarize history for accuracy

plt.plot(history.history['acc'])

plt.plot(history.history['loss'])

plt.title('Model Accuracy VS Loss')

plt.ylabel('Score')

plt.xlabel('epoch')

plt.legend(['Accuracy', 'Loss'])

plt.show()
test_img.shape
X_test = test_img.reshape(test_img.shape[0],28,28,1)

test_labels_pred = model.predict(X_test)