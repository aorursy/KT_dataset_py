# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import keras

from keras.models import Sequential

from keras.layers import Dense

from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import OneHotEncoder

from sklearn import preprocessing

from sklearn.metrics import accuracy_score

import math

import itertools

import sklearn

import seaborn as sns

from IPython import display

from matplotlib import cm

from matplotlib import gridspec

from matplotlib import pyplot as plt

import time
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
print (train.describe())					#Gives statitics of the data



X_train = train.loc[:, train.columns != 'label']

y_labels_train = train["label"]

# print (X_train)

# print (y_labels_train)
print (test.describe())					#Gives statitics of the data



X_test = test.loc[:, test.columns != 'label']

# print (X_test)
X_train = X_train.values

X_test = X_test.values



y_labels_train = y_labels_train.values

lb = preprocessing.LabelBinarizer()

lb.fit(y_labels_train)

output_classes = lb.classes_

print ("No.of Output Classes = ",output_classes)

y_train = lb.transform(y_labels_train)
Data_y = {}



for i in output_classes:

    Data_y[i] = 0

for j in y_labels_train:

    Data_y[j] += 1

    

print (Data_y)

plt_ = sns.barplot(list(Data_y.keys()), list(Data_y.values()))

plt_.set_xticklabels(plt_.get_xticklabels(), rotation=90)

plt.show()
print ("Shape of Training Set is",X_train.shape)

print ("Shape of Test Set is",X_test.shape)



print ("Shape of Training Set is",y_train.shape)
X_train = X_train.reshape(-1,28,28,1)

X_test = X_test.reshape(-1,28,28,1)



X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
keras.backend.clear_session()

from tensorflow.keras import layers

from tensorflow.keras import Model

from tensorflow.keras.optimizers import RMSprop,Adam



# Our input feature map is 28x28

img_input = layers.Input(shape=(28, 28, 1))



# First convolution extracts 96 filters that are 3x3

# Convolution is followed by max-pooling layer with a 2x2 window

x = layers.Conv2D(96, 3, activation='relu')(img_input)

x = layers.MaxPooling2D(2)(x)



# Second convolution extracts 32 filters that are 3x3

# Convolution is followed by max-pooling layer with a 2x2 window

x = layers.Conv2D(48, 3, activation='relu')(x)

x = layers.MaxPooling2D(2)(x)



# Flatten feature map to a 1-dim tensor

x = layers.Flatten()(x)



# Create a fully connected layer with ReLU activation and 512 hidden units

x = layers.Dense(1024, activation='relu')(x)



# Add a dropout rate of 0.5

x = layers.Dropout(0.5)(x)



# Create output layer with a single node and sigmoid activation

output = layers.Dense(10, activation='softmax')(x)



# Configure and compile the model

model = Model(img_input, output)

model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.003),metrics=['acc'])
model.summary()
datagen = ImageDataGenerator(

    rescale=1./255,

    zoom_range=0.2,

)

datagen.fit(X_train)

results = model.fit_generator(datagen.flow(X_train,y_train, batch_size=512),

                              epochs = 10, validation_data = datagen.flow(X_val,y_val),

                              verbose = 2, steps_per_epoch=X_train.shape[0] / 512)

def plot_confusion_matrix(cm, classes,normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



# Predict the values from the validation dataset

y_pred = model.predict(X_val)

# Convert predictions classes to one hot vectors 

y_pred_classes = np.argmax(y_pred,axis = 1) 

# Convert validation observations to one hot vectors

y_true = np.argmax(y_val,axis = 1) 

# compute the confusion matrix

confusion_mtx = confusion_matrix(y_true, y_pred_classes) 

# plot the confusion matrix

plot_confusion_matrix(confusion_mtx, classes = range(10)) 
results = model.predict(X_test)

predictions = np.argmax(results,axis = 1)



output_test_data = pd.DataFrame() 

output_test_data['Label'] = predictions

rows = predictions.shape[0]

print (rows)

output_test_data['ImageId'] = list(np.arange(1,rows+1))

submission = output_test_data[['ImageId','Label']]

submission.to_csv("submission.csv", index=False)

submission.tail()