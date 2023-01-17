import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn import svm

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix 
dataset = pd.read_csv('../input/titanic/train.csv')

print(dataset.isnull().sum())
dataset.head()
data = pd.DataFrame()

col = [['Pclass'],['Sex'],['Age'],['SibSp'],['Parch'],['Fare'],['Embarked'],['Survived']]

for x in col:

  data[x]=dataset[x]



data['Age'] = data['Age'].interpolate(method="linear",limit_direction='forward')

data['Embarked']=data['Embarked'].fillna(method='bfill')

encoder = preprocessing.LabelEncoder()

encoder.fit(data['Sex'])

data['Sex']=encoder.transform(data['Sex'])



train , test , train_target , test_target = train_test_split(data.iloc[:,:6],data.iloc[:,7],train_size = 0.8)
svm_model = svm.SVC(kernel='linear',C=1,gamma='auto').fit(train,train_target)

svm_predictor = svm_model.predict(test)

accuracy = svm_model.score(test, test_target) 

cm = confusion_matrix(test_target, svm_predictor) 

accuracy