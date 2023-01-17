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
from numpy import mean

from numpy import std

import pandas as pd 

import numpy as np

from matplotlib import pyplot

from sklearn.model_selection import KFold

from keras.datasets import mnist

from keras.utils import to_categorical

from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Dense

from keras.layers import Flatten

from keras.optimizers import SGD

import pandas as pd 





train=  pd.read_csv('../input/Kannada-MNIST/train.csv')

test=  pd.read_csv('../input/Kannada-MNIST/test.csv')











# example of loading the mnist dataset

from keras.datasets import mnist

from matplotlib import pyplot









def load_dataset():

	# load dataset

	#(trainX, trainY), (testX, testY) = df.load_data()

	# reshape dataset to have a single channel

	trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))

	testX = testX.reshape((testX.shape[0], 28, 28, 1))

	# one hot encode target values

	trainY = to_categorical(trainY)

	testY = to_categorical(testY)

	return trainX, trainY, testX, testY



# scale pixels

def prep_pixels(train, test):

	# convert from integers to floats

	train_norm = train.astype('float32')

	test_norm = test.astype('float32')

	# normalize to range 0-1

	train_norm = train_norm / 255.0

	test_norm = test_norm / 255.0

	# return normalized images

	return train_norm, test_norm



# define cnn model

def define_model():

	model = Sequential()

	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))

	model.add(MaxPooling2D((2, 2)))

	model.add(Flatten())

	model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))

	model.add(Dense(10, activation='sigmoid'))

	# compile model

	#opt = SGD(lr=0.01, momentum=0.9) 

	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

	return model



# evaluate a model using k-fold cross-validation

def evaluate_model(dataX, dataY, n_folds=2):

	scores, histories = list(), list()

	# prepare cross validation

	kfold = KFold(n_folds, shuffle=True, random_state=1)

	# enumerate splits

	for train_ix, test_ix in kfold.split(dataX):

		# define model

		model = define_model()

		# select rows for train and test

		trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]

		# fit model

		history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)

		# evaluate model

		_, acc = model.evaluate(testX, testY, verbose=0)

		print('> %.3f' % (acc * 100.0))

		# stores scores

		scores.append(acc)

		histories.append(history)

	return scores, histories



# plot diagnostic learning curves

def summarize_diagnostics(histories):

	for i in range(len(histories)):

		# plot loss

		pyplot.subplot(2, 1, 1)

		pyplot.title('Cross Entropy Loss')

		pyplot.plot(histories[i].history['loss'], color='blue', label='train')

		pyplot.plot(histories[i].history['val_loss'], color='orange', label='test')

		# plot accuracy

		pyplot.subplot(2, 1, 2)

		pyplot.title('Classification Accuracy')

		pyplot.plot(histories[i].history['accuracy'], color='blue', label='train')

		pyplot.plot(histories[i].history['val_accuracy'], color='orange', label='test')

	pyplot.show()



# summarize model performance

def summarize_performance(scores):

	# print summary

	print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))

	# box and whisker plots of results

	pyplot.boxplot(scores)

	pyplot.show()



# run the test harness for evaluating a model

def run_test_harness():

  trainX=train.iloc[:,1:].values 

  trainY=train.iloc[:,0].values 

  trainX = trainX.reshape(trainX.shape[0], 28, 28,1) 

  trainY = to_categorical(trainY, 10) 

  testX=test.drop('id', axis=1).iloc[:,:].values

  testX = testX.reshape(testX.shape[0], 28, 28,1)

  #trainY = train.loc[:,'label'].values

  #trainX = train.loc[:,'pixel0':].values

  #testY= test.loc[:,'id'].values

  #testX=test.loc[:,'pixel0':].values

	# load dataset

	#trainX, trainY, testX, testY = load_dataset()

	# prepare pixel data

  trainX, testX = prep_pixels(trainX, testX)

	# evaluate model

  scores, histories = evaluate_model(trainX, trainY)

	# learning curves

  summarize_diagnostics(histories)

	# summarize estimated performance

  summarize_performance(scores)



# entry point, run the test harness

run_test_harness()