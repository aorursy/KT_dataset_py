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
dataset = pd.read_csv('/kaggle/input/mnist-in-csv/mnist_train.csv')
X = dataset.values[:, 1:]

y = dataset.values[:, 0]
len(X), len(y)
from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV

from sklearn import metrics
parameters = {}

model = {}

result = {}
parameters['forest'] = { 'n_estimators':[50, 100, 200] }
model['forest'] = RandomForestClassifier()
clf = GridSearchCV(model['forest'], parameters['forest'], cv=2)

result['forest'] = clf.fit(X, y)
result['forest'].best_params_, result['forest'].best_score_