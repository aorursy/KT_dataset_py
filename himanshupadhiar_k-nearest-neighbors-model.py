# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import itertools

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib.ticker import NullFormatter

import numpy as np

import pandas as pd

import matplotlib.ticker as ticker

from sklearn import preprocessing

%matplotlib inline

dataset = "../input/teleCust1000t.csv"

df_data = pd.read_csv(dataset)

df_data.head()
#Check the count of each category.

df_data["custcat"].value_counts()
X = df_data[["region","tenure","age","marital","address","income","ed","employ","retire","gender","reside"]].values

print(X[0:5])
y = df_data["custcat"].values

print(y[0:5])
# Normalize data

X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

print(X[0:5])
# Create a Train\Test Split.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 4)

print('Train set : ', X_train.shape, y_train.shape)

print('Test set : ', X_test.shape, y_test.shape)
# Import KNeighborsClassifier

from sklearn.neighbors import KNeighborsClassifier
k = 9

neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)

neigh
# Predict the values

y_hat = neigh.predict(X_test)

y_hat
# Check the accuracy.

from sklearn import metrics

train_set_accuracy = metrics.accuracy_score(y_train, neigh.predict(X_train))

test_set_accuracy = metrics.accuracy_score(y_test, y_hat)

print('Train set accuracy : ', train_set_accuracy)

print('Test set accuracy : ', test_set_accuracy)