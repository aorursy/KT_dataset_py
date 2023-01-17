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
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

train_data.head()
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

test_data
from sklearn import svm

from sklearn import preprocessing



cols_with_missing = [col for col in train_data.columns 

                                 if train_data[col].isnull().any()]

train_data = train_data.drop(cols_with_missing, axis=1)

test_data = test_data.drop(cols_with_missing, axis=1)



y = train_data['Survived']

features = ['PassengerId','Pclass','Sex','SibSp','Parch']



X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])



scaler = preprocessing.StandardScaler()

train_X = scaler.fit_transform( X )

test_X = scaler.transform( X_test )

from sklearn.ensemble import RandomForestClassifier





model = RandomForestClassifier(n_estimators=1000, max_depth=100, random_state=1)

model.fit(train_X, y)

predictions = model.predict(test_X)



output = pd.DataFrame({'PassengerId':test_data.PassengerId,'Survived':predictions})

output.to_csv('mysubmission.csv',index=False)

print("Your submission was successfully saved")