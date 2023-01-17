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
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']

# load dataset
pima = pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv", names=col_names)
pima.head()
# load dataset
pima = pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv", skiprows=1, names=col_names)
pima.head()
pima.shape
pima.columns
feature_cols = ['pregnant', 'glucose', 'bp','skin', 'insulin', 'bmi', 'pedigree','age']
X = pima[feature_cols] # Features
y = pima.label # Target variable
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
display(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
# instantiate the model (using the default parameters)
lr = LogisticRegression()

# fit the model with data
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)
y_pred
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix
cnf_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(cnf_matrix, annot=True)
accuracyScore = metrics.accuracy_score(y_test, y_pred)
print('Accuracy Score : ',accuracyScore)
print('Accuracy In Percentage : ', int(accuracyScore*100), '%')
