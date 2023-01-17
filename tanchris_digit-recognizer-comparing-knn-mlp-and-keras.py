# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



%matplotlib inline

import numpy as np # linear algebra

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import matplotlib

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import os as os

import sys

import warnings

import time

warnings.filterwarnings('ignore')



sns.set(style='white', context='notebook', palette='deep')

np.random.seed(2)

from IPython.display import Image



# From Matplotlib

from matplotlib.colors import ListedColormap



# From Scikit Learn

from sklearn import preprocessing, decomposition, tree

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split, cross_val_score

from astropy.table import Table, Column

from sklearn.preprocessing import LabelEncoder

import itertools



# Set DEBUG = True to produce debug results

DEBUG = False
print("The Python version is %s.%s.%s." % sys.version_info[:3])
%pwd
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

%time digit_train = pd.read_csv("../input/train.csv", header=0, sep=",")

%time digit_test = pd.read_csv("../input/test.csv", header=0, sep=",")
digit_train.describe()
digit_test.describe()
# Source: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.dropna.html

digit_train = digit_train.dropna(axis = 0, how = 'all')

digit_test = digit_test.dropna(axis = 0, how = 'all')

if DEBUG:

    #Dimensions of dataset

    print("Shape of Data", digit_train.shape)

    print("Shape of Data", digit_test.shape)

    #Colum names

    print("Colums Names", digit_train.columns)

    print("Colums Names", digit_test.columns)

    #See bottol few rows of dataset

    print(digit_train.tail())
# designate target variable name

targetName = 'label'

targetSeries = digit_train[targetName]

#remove target from current location and insert in collum 0

del digit_train[targetName]

digit_train.insert(0, targetName, targetSeries)

#reprint dataframe and see target is in position 0

digit_train.head()
digit_train.info()
sns.countplot(digit_train['label'])
print(digit_train['label'].describe())
if DEBUG:

    print(digit_train.dtypes)
digit_train['label'] = digit_train['label'].astype(str)

if DEBUG:

    print(digit_train['label'].describe())
#digit_train.fillna(digit_train.median(), inplace=True)

#print(digit_train.describe())
#if DEBUG:

#    print(digit_train.shape)

#    print(digit_train.info())

#    print(digit_train.head())
features_train = digit_train.iloc[:,1:]

target_train = digit_train.iloc[:,0]

features_test = digit_test.iloc[:,0:]
# pixel values are gray scale between 0 and 255

# normalize inputs from 0-255 to 0-1

features_train = features_train/255.0

features_test = features_test/255.0
if DEBUG:

    print(features_train)
start_time = time.perf_counter()

train_results = []

test_results = []

# search for an optimal value of k for KNN MOdel

k_range = list(range(1,5))

k_scores = []

for k in k_range:

    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1, algorithm='ball_tree', leaf_size=40, weights='uniform')

    scores = cross_val_score(knn, features_train, target_train, cv=10, scoring='accuracy', n_jobs=-1)

    k_scores.append(scores.mean())

if DEBUG:

    print(k_scores) 

print(time.perf_counter() - start_time, "seconds")
if DEBUG:

    scores = pd.DataFrame(k_scores)

    print(scores)
# plot the value of K (x-axis) versus the cross-validated accuracy (y-axis)

plt.plot(k_range, k_scores)

plt.xlabel('K Value for KNN')

plt.ylabel('Cross-Validated Accuracy')

plt.title('KNN Model for Accuracy')
# changing to misclassification error

MSE = [1 - x for x in k_scores]



# determining best k

optimal_k = k_range[MSE.index(min(MSE))]



# plot misclassification error vs k

plt.plot(k_range, MSE)

plt.xlabel('Number of Neighbors K')

plt.ylabel('Misclassification Error')

plt.title('KNN Model for Misclassification Error')

plt.show()
print("The optimal number of neighbors is %d." % optimal_k)
start_time = time.perf_counter()

#KNN train model. Call up my model and name it clf_knn

clf_knn = KNeighborsClassifier(n_neighbors=3, n_jobs=-1, algorithm='ball_tree', leaf_size=40, weights='uniform')

#Call up the model to see the parameters you can tune (and their default setting)

print(clf_knn)

#Fit clf to the training data

clf_knn = clf_knn.fit(features_train, target_train)

#Predict clf_knn model again test data

target_predicted_knn = clf_knn.predict(features_test)

print(time.perf_counter() - start_time, "seconds")
start_time = time.perf_counter()

#verify KNN with Cross Validation

scores_knn = cross_val_score(clf_knn, features_train, target_train, cv=10, scoring='accuracy', n_jobs=-1)

print("Cross Validation Score for each K",scores_knn)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores_knn.mean(), scores_knn.std() * 2))

print(time.perf_counter() - start_time, "seconds")
digit_test['Label'] = pd.Series(target_predicted_knn)

digit_test['ImageId'] = pd.Series(range(1,28001))
digit_test.to_csv('submission_knn.csv', columns=["ImageId","Label"], index=False)
start_time = time.perf_counter()

from sklearn.neural_network import MLPClassifier

# Multi-layer Perceptron train model. Call up my model and name it clf_mlp

clf_mlp = MLPClassifier(hidden_layer_sizes=(784,), warm_start=True)

#Call up the model to see the parameters you can tune (and their default setting)

print(clf_mlp)

#Fit clf_NN to the training data

clf_mlp = clf_mlp.fit(features_train, target_train)

#Predict clf_NN model again test data

target_predicted_mlp = clf_mlp.predict(features_test)

print(time.perf_counter() - start_time, "seconds")
start_time = time.perf_counter()

#verify RF with Cross Validation

scores_mlp = cross_val_score(clf_mlp, features_train, target_train, cv=10, n_jobs=-1, scoring='accuracy')

print("Cross Validation Score for MLP",scores_mlp)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores_mlp.mean(), scores_mlp.std() * 2))

print(time.perf_counter() - start_time, "seconds")
digit_test['Label'] = pd.Series(target_predicted_mlp)

digit_test['ImageId'] = pd.Series(range(1,28001))
digit_test.to_csv('submission_mlp.csv', columns=["ImageId","Label"], index=False)
from IPython.lib.display import YouTubeVideo

vid = YouTubeVideo('j_pJmXJwMLA', autoplay=0)

display(vid)
# From Keras for TensorFlow Model

import keras as keras

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau
# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)

X_train = features_train.values.reshape(-1,28,28,1)

test = features_test.values.reshape(-1,28,28,1)
# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])

Y_train = to_categorical(target_train, num_classes = 10)
# Split the train and the validation set for the fitting

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=0)
g = plt.imshow(X_train[10][:,:,0])
clf_keras = Sequential()
clf_keras.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'same', strides=1, activation ='relu', 

                     input_shape = (28,28,1)))

clf_keras.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'same', activation ='relu'))

clf_keras.add(keras.layers.BatchNormalization())

clf_keras.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
clf_keras.add(Conv2D(filters = 64, kernel_size = (3,3), strides=2, padding = 'same', activation ='relu'))

clf_keras.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'same', activation ='relu'))

clf_keras.add(keras.layers.BatchNormalization())

clf_keras.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
clf_keras.add(Dropout(rate = 0.5))
clf_keras.add(Flatten())
clf_keras.add(Dense(10, activation = "softmax"))
# Define the optimizer

optimizer = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
# Compile the model

clf_keras.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics=["accuracy"])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=3, min_lr=0.00001, verbose=1)
datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.10, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images



datagen.fit(X_train)
epochs = 30
batch_size = 86
start_time = time.perf_counter()

# Fit the model 

history = clf_keras.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size), epochs = epochs, 

                                  validation_data = (X_val,Y_val), verbose = 2, 

                                  steps_per_epoch=features_train.shape[0] // batch_size, 

                                  callbacks=[learning_rate_reduction], 

                                  use_multiprocessing=True)

print(time.perf_counter() - start_time, "seconds")
# Plot training & validation accuracy values

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
# predict results

results = clf_keras.predict(test)



# select the indix with the maximum probability

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("submission_keras.csv",index=False)