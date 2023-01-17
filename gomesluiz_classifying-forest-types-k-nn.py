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
train_data = pd.read_csv('../input/learn-together/train.csv')
train_data.head()
train_data.describe()
train_data.columns
train_data[train_data.isnull().any(axis=1)]
predictors = ~train_data.columns.isin(['Id','Cover_Type'])

target     = "Cover_Type"

X_full_data     = train_data[train_data.columns[predictors]]

y_full_data     = train_data[target]





# Split dataset into training set and test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_full_data, y_full_data, test_size=0.3)

print("Dataset split")
X_train.describe()
X_test.head()
from sklearn.neighbors import KNeighborsClassifier



# Train model

knn = KNeighborsClassifier(n_neighbors=7, metric='minkowski', p=2)

knn.fit(X_train, y_train)



# Test model

y_hat = knn.predict(X_test)

print("Model built")
from sklearn import metrics

print("Accuracy:",metrics.accuracy_score(y_test, y_hat))
knn_on_full_data = KNeighborsClassifier(n_neighbors=7)

knn_on_full_data.fit(X_full_data, y_full_data)

test_data_path = '../input/learn-together/test.csv'

test_data = pd.read_csv(test_data_path)

test_X = test_data[test_data.columns[~test_data.columns.isin(['Id'])]]

test_preds = knn_on_full_data.predict(test_X)



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.

output = pd.DataFrame({'Id': test_data.Id,'Cover_Type': test_preds})

output.to_csv('submission.csv', index=False)