# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
#boolCheck = train.isnull()
#boolCheck.head(10)
#sns.heatmap(data=boolCheck,cmap='coolwarm')
train['Age'].fillna(value=train['Age'].mean(),inplace=True)
#boolCheck = train.isnull()
#sns.heatmap(data=boolCheck,cmap='coolwarm',cbar=True)
train.dropna(axis=1,inplace=True)
#boolCheck = train.isnull()
#sns.heatmap(data=boolCheck,cmap='coolwarm',cbar=True)
#sns.pairplot(data=train,hue='Survived',palette='viridis')
dummy = pd.get_dummies(data=train,columns=['Sex'],dummy_na=False,drop_first=True)
X = dummy[['Pclass','Age','SibSp','Parch','Fare','Sex_male']]
y = dummy['Survived']
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X,y)
test = pd.read_csv('../input/test.csv')
#boolCheck = test.isnull()
#plt.figure(figsize=(12,7))
#sns.heatmap(data=boolCheck,cmap='coolwarm',cbar=True)
test['Age'].fillna(value=train['Age'].mean(),inplace=True)
test['Fare'].fillna(value=train['Fare'].mean(),inplace=True)
test.dropna(axis=1,inplace=True)
test_records = pd.get_dummies(data=test,columns=['Sex'],dummy_na=False,drop_first=True)
X_test = test_records[['Pclass','Age','SibSp','Parch','Fare','Sex_male']]
predictions = model.predict(X_test)
output = pd.DataFrame()
output['PassengerId'] = test['PassengerId']
output['Survived'] = predictions
output.to_csv('output.csv',index=False)
