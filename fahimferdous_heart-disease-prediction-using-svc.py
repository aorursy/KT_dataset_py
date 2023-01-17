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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df_heart = pd.read_csv('../input/heart.csv')
df_heart.head()
df_heart.keys()

sns.pairplot(df_heart, hue='target', vars = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs'])
sns.countplot(df_heart['target'])
sns.scatterplot(x = 'age', y = 'trestbps', hue='target', data = df_heart)
plt.figure(figsize=[30,15])

sns.heatmap(df_heart.corr(), annot = True)
x = df_heart.drop(['target'], axis = 1)

y = df_heart['target']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.33, random_state=42)
x_train
from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix
svc_model = SVC()

svc_model.fit(x_train,y_train)
y_predict = svc_model.predict(x_test)

y_predict
cm = confusion_matrix(y_test, y_predict)

sns.heatmap(cm, annot = True, fmt = 'd')
print(classification_report(y_test, y_predict))
min_train = x_train.min()

range_train = (x_train - min_train).max()

x_train_scaled = (x_train - min_train)/range_train
sns.scatterplot(x = 'age', y = 'trestbps', hue=y_train, data = x_train_scaled)
min_test = x_test.min()

range_test = (x_test - min_test).max()

x_test_scaled = (x_test - min_test)/range_test
svc_model.fit(x_train_scaled, y_train)
y_predict = svc_model.predict(x_test_scaled)

cm = confusion_matrix(y_test, y_predict)

sns.heatmap(cm, annot=True, fmt='d')
print(classification_report(y_test, y_predict))
param_grid = {'C' : [0.1, 1, 10,100,1000],  'gamma' : [1, 0.1, 0.01,.001,0.0001], 'kernel': ['rbf']}
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 4)

grid.fit(x_train_scaled, y_train)
grid.best_params_
grid_predictions = grid.predict(x_test_scaled)
cm = confusion_matrix(y_test, grid_predictions)

sns.heatmap(cm, annot = True, fmt = 'd')
print(classification_report(y_test, grid_predictions))