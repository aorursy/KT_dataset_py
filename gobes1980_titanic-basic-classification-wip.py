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
# import data
train = pd.read_csv('../input/train.csv', index_col='PassengerId')
test = pd.read_csv('../input/test.csv', index_col='PassengerId')
#inspect data
train.info()
#inspect data some more
train.describe()
#inspect some rows
train.head()
#importing some plotting tools
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
#create heatmap of feature correlations
sns.heatmap(train.corr(), annot=True)
# list of columns with features
f_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
# create tabel with features only
f_train = train[f_columns]
# inspect some rows of the feature table
f_train.head()
# transfer categorical features to numerical
f_train = pd.get_dummies(f_train, drop_first=True)
# inspect some rows again after changing the columns
f_train.head()
# preparing data for ML algorythms and creating target array to predict
X = f_train.values
y = train['Survived'].values
y = y.reshape(-1, 1)

# import algo and function to split data in train & test (not necessary in this case since we already have a train and test set.)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# creating the training and test data
# X_train, X_test, y_train, y_test = train_test_split(X, y.reshape(-1,1),  test_size=0.3)
# creating a model
reg = LogisticRegression()
# import an imputer
from sklearn.preprocessing import Imputer
# create the imputer
imp = Imputer()
from sklearn.pipeline import make_pipeline
# create pipeline with imputer and logistical regression model
pipeline = make_pipeline(imp, reg)
# fit pipeline to data
pipeline.fit(X, y)
X_test = test[f_columns]
X_test = pd.get_dummies(X_test, drop_first=True)
# score pipeline
# pipeline.score(X_test, y_test)
# predict label probabilities
# pipeline.predict_proba(X_test)
# predict labels
y_pred = pipeline.predict(X_test)
# output predictions
y_pred
submission = pd.DataFrame({'PassengerID' : X_test.index, 'Survived':y_pred})
submission.head()
submission.to_csv('submission.csv', index=False)












