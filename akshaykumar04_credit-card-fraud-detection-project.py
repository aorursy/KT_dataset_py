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
import numpy as nd

import pandas as pd

import sys

import sklearn

import scipy

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv('../input/creditcardfraud/creditcard.csv')
data.describe()
print(data.columns)

data.shape
data.head()
data.hist(figsize = (40,40))
fraud = data[data['Class'] == 1]

valid = data[data['Class'] == 0]

outlier_fraction = len(fraud)/len(valid)

print(outlier_fraction)

print('fraud cases: {}'.format(len(fraud)))

print('valid cases: {}'.format(len(valid)))
correlation_matrix = data.corr()

fig = plt.figure(figsize = (12,9))

sns.heatmap(correlation_matrix ,vmax = 0.8 , square = True)

plt.show()
columns = data.columns.tolist()

columns = [c for c in columns if c not in['Class']]

target = 'Class'

X = data[columns]

Y = data[target]



print(X.shape)

print(Y.shape)
from sklearn.metrics import classification_report, accuracy_score

from sklearn.ensemble import IsolationForest

from sklearn.neighbors import LocalOutlierFactor
state = 1

classifiers = {'Isolation Forest' : IsolationForest(max_samples = len(X),contamination = outlier_fraction, random_state = state),

               'Local Outlier Factor' : LocalOutlierFactor(n_neighbors = 20,contamination = outlier_fraction)}
n_outliers = len(fraud)

for i, (clf_name, clf) in enumerate(classifiers.items()):

    if clf_name == 'Local Outlier Factor':

        y_pred = clf.fit_predict(X)

        scores_pred = clf.negative_outlier_factor_

    else:

        clf.fit(X)

        scores_pred = clf.decision_function(X)

        y_pred = clf.predict(X)

        

        y_pred[y_pred == 1] = 0

        y_pred[y_pred == -1] = 1

        

        n_errors = (y_pred != Y).sum()

        

        print('{}:{}'.format(clf_name, n_errors))

        print(accuracy_score(Y, y_pred))

        print(classification_report(Y, y_pred))