# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Load training set

df_raw_train = pd.read_csv('../input/train.csv')

df_raw_train.head()
#Load testing set

df_raw_test = pd.read_csv('../input/test.csv')

df_raw_test.head()
df_raw_train.isnull().sum().sort_index()
train_data = df_raw_train.copy()

train_data["Age"].fillna(df_raw_train["Age"].median(skipna=True), inplace=True)

train_data["Embarked"].fillna(df_raw_train['Embarked'].value_counts().idxmax(), inplace=True)

train_data.drop('Cabin', axis=1, inplace=True)

train_data.isnull().sum()
data_cols = train_data[['Pclass','Sex','Age','Fare','SibSp', 'Parch','Embarked']]

X = pd.get_dummies(data_cols)

print(X.head())

y = train_data['Survived']

print(y.head())
X.isnull().sum()
#dividing the data in training and test data 



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)

logreg = LogisticRegression() #logistic regression using python

my_model = logreg.fit(X_train, y_train)
from sklearn.metrics import accuracy_score

y_pred = logreg.predict(X_val) #predicting the values

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(accuracy_score(y_val, y_pred)))
import eli5

from eli5.sklearn import PermutationImportance



perm = PermutationImportance(my_model, random_state=1).fit(X_val, y_val)

eli5.show_weights(perm, feature_names = X_val.columns.tolist())