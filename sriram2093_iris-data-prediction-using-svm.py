# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn #ML model building

from sklearn import svm



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
IrisData = pd.read_csv('../input/Iris.csv')
print(IrisData.columns)
from sklearn import svm
X = IrisData[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']].astype(float)

y  = IrisData['Species']
from sklearn.cross_validation import train_test_split

from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

encoder.fit(y)

encoded_Y = encoder.transform(y)

# convert integers to dummy variables (i.e. one hot encoded)

dummy_y = np_utils.to_categorical(encoded_Y)
from sklearn.model_selection import cross_val_score

clf = svm.SVC(kernel='linear', C=1)

scores = cross_val_score(clf, X, y, cv=10)



print(scores)

print(np.mean(scores))
from sklearn.preprocessing import LabelEncoder

from sklearn.cross_validation import train_test_split

from keras.models import Sequential

from keras.layers import Activation

from keras.optimizers import SGD

from keras.layers import Dense

from keras.utils import np_utils

import numpy as np

import argparse

import cv2

import os
from keras.models import Sequential

from keras.layers import Dense, Activation

from keras.wrappers.scikit_learn import KerasClassifier

from keras.utils import np_utils

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import Pipeline





# fix random seed for reproducibility

seed = 7

np.random.seed(seed)



# define baseline model

def baseline_model():

	# create model

	model = Sequential()

	model.add(Dense(4, input_dim=4, kernel_initializer='normal', activation='relu'))

	model.add(Dense(3, kernel_initializer='normal', activation='sigmoid'))

	# Compile model

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	return model



estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)

kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

results = cross_val_score(estimator, X, dummy_y, cv=kfold)

print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
print(X.shape)

print(dummy_y.shape)