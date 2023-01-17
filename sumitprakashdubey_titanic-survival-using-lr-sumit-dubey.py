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
train_data=pd.read_csv("../input/train.csv")

test_data=pd.read_csv("../input/test.csv")
train_data.head()

test_data.head()
import seaborn as sns
sns.countplot(x='Survived',data=train_data)
sns.countplot(x='Survived',hue='Sex',data=train_data)
sns.distplot(train_data['Age'].dropna())
train_data['Age']=train_data['Age'].fillna(train_data['Age'].mean())

test_data['Age']=test_data['Age'].fillna(test_data['Age'].mean())
train_data[train_data['Age'].isnull()]

#test_data[train_data['Age'].isnull()]
sns.heatmap(train_data.isnull())

#sns.heatmap(test_data.isnull())
train_data.drop('Cabin',axis=1,inplace=True)

test_data.drop('Cabin',axis=1,inplace=True)
train_data.head()

#test_data.head()
train_data.info()
sex=pd.get_dummies(train_data['Sex'],drop_first=True)
train_data['Sex_m']=sex
train_data.drop(['Name','Sex','Ticket','Embarked','Sex'],axis=1,inplace=True)
sex=pd.get_dummies(test_data['Sex'],drop_first=True)
test_data['Sex_m']=sex
test_data.drop(['Name','Sex','Ticket','Embarked','Sex'],axis=1,inplace=True)
test_data.head()
#train_data.head()

test_data.tail()
features=['Pclass', 'Age', 'SibSp', 'Parch', 'Sex_m']

target='Survived'
#test_data[test_data.isnull()==True]

from sklearn.linear_model import LogisticRegression
l_model=LogisticRegression()
l_model.fit(train_data[features],train_data[target])
predict=l_model.predict(test_data[features])
submission = pd.DataFrame({'PassengerId':test_data['PassengerId'],'Survived':predict})
submission.head()