# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



from sklearn import preprocessing



%matplotlib inline



df_train = pd.read_csv('../input/train.csv', index_col='id')

df_train.describe()
data = df_train.drop('species', axis=1).values

labels = df_train['species'].values



le = preprocessing.LabelEncoder()

le.fit(labels)

classes = le.classes_

targets = le.transform(labels)



print(data.shape)

print(targets.shape)

print(classes.shape)
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier



rfc = RandomForestClassifier()

parameters = {'n_estimators': (10, 50, 100), 'max_features':['sqrt'], 'min_samples_leaf':(1, 2)}

clf = GridSearchCV(rfc, parameters, cv=5)



clf.fit(data, targets)



print(clf.best_score_)

print(clf.best_params_)



clf = clf.best_estimator_
df_test = pd.read_csv('../input/test.csv', index_col='id')



test_data = df_test.values

print(test_data.shape)



proba = clf.predict_proba(test_data)



df_sub = pd.DataFrame(proba, index=df_test.index, columns=classes)

df_sub.to_csv('submission.csv', index='id')