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

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

path = '../input/'

train_data = pd.read_csv(path+'train.csv')

test = pd.read_csv(path+'test.csv')

train_data.head()
train_data.describe()
exclude_features = ['PassengerId','Name', 'SibSp','Cabin','Ticket']

train_data = train_data.drop(exclude_features, axis= 1)

test = test.drop(exclude_features, axis =1)

train_data = train_data.fillna(0, axis= 0)

test = test.fillna(0, axis=0)

encoder = LabelEncoder()

scaler = StandardScaler()



train_data['Sex'] = encoder.fit_transform(train_data['Sex'])

train_data['Embarked']= encoder.fit_transform(train_data['Embarked'].astype(str))





test['Sex'] = encoder.fit_transform(test['Sex'])

test['Embarked']= encoder.fit_transform(test['Embarked'].astype(str))



train_data_features = train_data.drop('Survived', axis = 1)

train_data_labels = train_data['Survived']



train_data = scaler.fit_transform(train_data)

test = scaler.fit_transform(test)
model = RandomForestClassifier(n_estimators= 10)



model.fit(train_data_features,train_data_labels)

predictions =model.predict(test)
output = pd.read_csv(path+'gender_submission.csv')



print(confusion_matrix(output['Survived'], predictions))

print(classification_report(output['Survived'], predictions))

print(accuracy_score(output['Survived'], predictions))
output = pd.DataFrame({'PassengerId': output['PassengerId'], 'Survived': predictions})

output.to_csv('submission.csv', index=False)
