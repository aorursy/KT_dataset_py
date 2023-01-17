import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

df = pd.read_csv('/kaggle/input/EEG_Eye_State.csv')
# visualize dataset

from pandas import read_csv

from matplotlib import pyplot

# load the dataset

data = df

# retrieve data as numpy array

values = data.values

# create a subplot for each time series

pyplot.figure()

for i in range(values.shape[1]):

	pyplot.subplot(values.shape[1], 1, i+1)

	pyplot.plot(values[:, i])

pyplot.show()
# remove outliers from the EEG data

from pandas import read_csv

from numpy import mean

from numpy import std

from numpy import delete

from numpy import savetxt

# load the dataset.

data = df

values = data.values

# step over each EEG column

for i in range(values.shape[1] - 1):

	# calculate column mean and standard deviation

	data_mean, data_std = mean(values[:,i]), std(values[:,i])

	# define outlier bounds

	cut_off = data_std * 4

	lower, upper = data_mean - cut_off, data_mean + cut_off

	# remove too small

	too_small = [j for j in range(values.shape[0]) if values[j,i] < lower]

	values = delete(values, too_small, 0)

	print('>deleted %d rows' % len(too_small))

	# remove too large

	too_large = [j for j in range(values.shape[0]) if values[j,i] > upper]

	values = delete(values, too_large, 0)

	print('>deleted %d rows' % len(too_large))
from pandas import read_csv

from matplotlib import pyplot

# load the dataset

data = df

# retrieve data as numpy array

values = data.values

# create a subplot for each time series

pyplot.figure()

for i in range(values.shape[1]):

	pyplot.subplot(values.shape[1], 1, i+1)

	pyplot.plot(values[:, i])

pyplot.show()
from pandas import read_csv

from sklearn.metrics import accuracy_score

from sklearn.model_selection import KFold

from sklearn.neighbors import KNeighborsClassifier

from numpy import mean

# load the dataset

data = df

values = data.values

# evaluate knn using 10-fold cross-validation

scores = list()

kfold = KFold(10, shuffle=True, random_state=1)

for train_ix, test_ix in kfold.split(values):

	# define train/test X/y

	trainX, trainy = values[train_ix, :-1], values[train_ix, -1]

	testX, testy = values[test_ix, :-1], values[test_ix, -1]

	# define model

	model = KNeighborsClassifier(n_neighbors=3)

	# fit model on train set

	model.fit(trainX, trainy)

	# forecast test set

	yhat = model.predict(testX)

	# evaluate predictions

	score = accuracy_score(testy, yhat)

	# store

	scores.append(score)

	print('>%.3f' % score)

# calculate mean score across each run

print('Final Score: %.3f' % (mean(scores)))
# knn for predicting eye state

from pandas import read_csv

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from numpy import array

# load the dataset

data = df

values = data.values

# split data into inputs and outputs

X, y = values[:, :-1], values[:, -1]

# split the dataset

trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.1, shuffle=False, random_state=1)

# walk-forward validation

historyX, historyy = [x for x in trainX], [x for x in trainy]

predictions = list()

for i in range(len(testy)):

    # define model

    model = KNeighborsClassifier(n_neighbors=3)

    # fit model on a small subset of the train set

    tmpX, tmpy = array(historyX)[-10:,:], array(historyy)[-10:]

    model.fit(tmpX, tmpy)

    # forecast the next time step

    yhat = model.predict([testX[i, :]])[0]

    # store prediction

    predictions.append(yhat)

    # add real observation to history

    historyX.append(testX[i, :])

    historyy.append(testy[i])

# evaluate predictions

score = accuracy_score(testy, predictions)

print(score)