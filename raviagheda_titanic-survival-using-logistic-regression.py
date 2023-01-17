# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
# replacing male with 0 and femlae with 1
train.replace({'Sex': {'male':0, 'female': 1}}, inplace=True)
test.replace({'Sex': {'male':0, 'female': 1}}, inplace=True)
# Dealing with null value in AGE column

mean_age = train['Age'].mean()
train['Age'].fillna(value= mean_age, inplace=True)
test['Age'].fillna(value=mean_age, inplace= True)
train_features = train[['Sex','Age','Pclass','SibSp']]
train_labels = train['Survived']
test_features = test[['Sex','Age','Pclass','SibSp']]
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.fit_transform(test_features)
model = LogisticRegression()

model.fit(train_features, train_labels)

predict_test_labels = model.predict(test_features)
train_score = model.score(train_features, train_labels)
train_score
given_labels = pd.read_csv('../input/titanic/gender_submission.csv')['Survived']
test_score = model.score(test_features, given_labels)
test_score
output = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': predict_test_labels.astype(int)})
output.to_csv('titanic_submission.csv', index=False)
