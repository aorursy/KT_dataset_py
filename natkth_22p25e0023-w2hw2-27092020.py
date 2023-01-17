# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
df = pd.read_csv('/kaggle/input/titanic/train.csv')

df2 = pd.read_csv('/kaggle/input/titanic/test.csv')
df.head()
df.info()
df.nunique()
df['Embarked'].unique()
df['Sex'] = df['Sex'].replace({'male':1,'female':0})

df['Embarked'] = df['Embarked'].replace({'S':1,'C':2,'Q':3})

df
df2['Sex'] = df2['Sex'].replace({'male':1,'female':0})

df2['Embarked'] = df2['Embarked'].replace({'S':1,'C':2,'Q':3})

df2
df['Age'].median()
df['Age'] = df['Age'].fillna(28.0)

df2['Age'] = df2['Age'].fillna(28.0)
df.isnull().sum()
df.drop(columns=['Cabin','Name','Ticket'],inplace=True)
df.isnull().sum()
df.dropna(inplace=True)
df.info()
df['Embarked'] = df['Embarked'].astype('int64')

df2['Embarked'] = df2['Embarked'].astype('int64')
df.describe()
df['Survived_'] = df['Survived'].astype('Bool')

df['Pclass_'] = df['Pclass'].astype('category')

df['Sex_'] = df['Sex'].astype('category')

df['Embarked_'] = df['Embarked'].astype('category')
df2['Pclass_'] = df2['Pclass'].astype('category')

df2['Sex_'] = df2['Sex'].astype('category')

df2['Embarked_'] = df2['Embarked'].astype('category')
df.head()
Age_group = [0,22,28,35,50,100]

Age_label = [1,2,3,4,5]

df['Age_'] = pd.cut(df['Age'],bins=Age_group,labels=Age_label)

df.head()
Age_group = [0,22,28,35,50,100]

Age_label = [1,2,3,4,5]

df2['Age_'] = pd.cut(df['Age'],bins=Age_group,labels=Age_label)

df2.head()
df2.drop(columns=['Cabin','Name','Ticket'],inplace=True)
df2.head()
Fare_group = [-1,8,15,30,100,200,550]

Fare_label = [1,2,3,4,5,6]

df['Fare_'] = pd.cut(df['Fare'],bins=Fare_group,labels=Fare_label)

df.info()
df.head()
df.isnull().sum()
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score

from sklearn import model_selection
#feature = ['Pclass_','Sex_','Age_','SibSp','Parch','Embarked','Fare_']

feature = ['Pclass_','Sex_','Age_','SibSp','Parch','Embarked','Fare_']

X = df[feature]

y = df['Survived']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
kfold = model_selection.KFold(n_splits=5)
model = DecisionTreeClassifier()

cvs = model_selection.cross_val_score(model,X_train,y_train,cv=kfold)

#cvs = cross_val_score(model,X_train,y_train,cv=5)

cvs
cvs.mean()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test,y_pred,target_names=['Dead','Survived']))
from sklearn.naive_bayes import GaussianNB
model2 = GaussianNB()

cvs2 = model_selection.cross_val_score(model2,X_train,y_train,cv=kfold)

#cvs2 = cross_val_score(model2,X_train,y_train,cv=5)

cvs2
cvs2.mean()
model2.fit(X_train,y_train)
y_pred2 = model2.predict(X_test)
print(classification_report(y_test,y_pred2,target_names=['Dead','Survived']))
from sklearn.linear_model import LogisticRegression
model3 = LogisticRegression()

cvs3 = model_selection.cross_val_score(model3,X_train,y_train,cv=kfold)

cvs3
cvs3.mean()
model3.fit(X_train,y_train)
y_pred3 = model3.predict(X_test)
print(classification_report(y_test,y_pred3,target_names=['Dead','Survived']))
from sklearn.neural_network import MLPClassifier
model4 = MLPClassifier(hidden_layer_sizes=(200,),max_iter=100,alpha=1e-4,solver='sgd',verbose=10,tol=1e-5,random_state=1,learning_rate_init=0.01)

cvs4 = model_selection.cross_val_score(model4,X_train,y_train,cv=kfold)

cvs4
cvs4.mean()
model4.fit(X_train,y_train)
y_pred4 = model4.predict(X_test)
print(classification_report(y_test,y_pred4,target_names=['Dead','Survived']))
print(classification_report(y_test,y_pred,target_names=['Dead','Survived']))
print(classification_report(y_test,y_pred2,target_names=['Dead','Survived']))
print(classification_report(y_test,y_pred3,target_names=['Dead','Survived']))
print(classification_report(y_test,y_pred4,target_names=['Dead','Survived']))
df2.head()
Fare_group = [-1,8,15,30,100,200,550]

Fare_label = [1,2,3,4,5,6]

df2['Fare_'] = pd.cut(df2['Fare'],bins=Fare_group,labels=Fare_label)

df2.head()
feature = ['Pclass_','Sex_','Age_','SibSp','Parch','Embarked','Fare_']

XX = df2[feature]
y_pred5 = model4.predict(XX)

y_pred5
NK_submission = pd.DataFrame({'PassengerId':df2['PassengerId'],'Survived':y_pred5})

NK_submission.to_csv('submission.csv',index=False)