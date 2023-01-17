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
# load dataset
x = pd.read_csv('../input/train.csv')

x.head()
numeric_variables=['PassengerId','Pclass','Age','SibSp','Parch','Fare']

x[numeric_variables].head()
x['Age'].isnull().sum()
# 177 NaN values in Age column
# Replacing NaN values with median of non NaN values in Age column

x['Age']=x['Age'].fillna(x['Age']).median()
x['Age'].isnull().sum()
y=x['Survived']
from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier(n_estimators=100)
# training the model...
model.fit(x[numeric_variables],y)
test = pd.read_csv('../input/test.csv') 
test[numeric_variables].head()
test['Age']=test['Age'].fillna(test['Age']).median()
test['Age'].isnull().sum()
test = test[numeric_variables].fillna(test.mean()).copy()
y_pred = model.predict(test[numeric_variables])
y_pred
Submission = pd.DataFrame({ 'PassengerId': test['PassengerId'],
                            'Survived': y_pred })
Submission.to_csv("Submission.csv", index=False)
Submission.head()
