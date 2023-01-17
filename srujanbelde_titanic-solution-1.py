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
import csv
train_data=pd.read_csv("../input/train.csv")
train_data.head(5)
test_data=pd.read_csv("../input/test.csv")
test_data.head(5)
train_data.dtypes
train_data['Cabin'] = train_data['Cabin'].astype('category')
train_data['Embarked'] = train_data['Cabin'].astype('category')
train_data['Sex'] = train_data['Sex'].astype('category')


test_data['Cabin'] = test_data['Cabin'].astype('category')
test_data['Embarked'] = test_data['Cabin'].astype('category')
test_data['Sex'] = test_data['Sex'].astype('category')
cat_columns = train_data.select_dtypes(['category']).columns
train_data[cat_columns] = train_data[cat_columns].apply(lambda x: x.cat.codes)


cat_columns = test_data.select_dtypes(['category']).columns
test_data[cat_columns] = test_data[cat_columns].apply(lambda x: x.cat.codes)
train_data=train_data.drop(['PassengerId','Name','Ticket'],axis=1)


test_data=test_data.drop(['PassengerId','Name','Ticket'],axis=1)
train_data['Age']=train_data['Age'].fillna(train_data['Age'].mean()).astype(int)
test_data['Age']=test_data['Age'].fillna(train_data['Age'].mean()).astype(int)

train_data['Fare']=train_data['Fare'].fillna(train_data['Fare'].mean()).astype(int)
test_data['Fare']=test_data['Fare'].fillna(train_data['Fare'].mean()).astype(int)
train_data['Fare']=train_data['Fare'].astype(int)
train_data['Age']=train_data['Age'].astype(int)

test_data['Fare']=test_data['Fare'].astype(int)
test_data['Age']=test_data['Age'].astype(int)
y=train_data['Survived']
X=train_data.drop(['Survived'],axis=1)
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,y)
result=clf.predict(test_data)
np.savetxt("result.csv", result, delimiter=",",header='Survived')
res=pd.DataFrame(result, columns=['Survived'])
sub=pd.read_csv("../input/gender_submission.csv")
sub['Survived']=res['Survived']
sub.to_csv('answer.csv')
from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression()
logisticRegr.fit(X,y)
result=logisticRegr.predict(test_data)

res=pd.DataFrame(result, columns=['Survived'])
sub=pd.read_csv("../input/gender_submission.csv")
sub['Survived']=res['Survived']
sub.to_csv('answer-LR.csv')