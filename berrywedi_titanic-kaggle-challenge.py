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
!pwd
!pip install -q  missingno
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import missingno as ms

%matplotlib inline
!ls -l
!wget -q https://www.dropbox.com/s/8grgwn4b6y25frw/titanic.csv
!ls -l
data = pd.read_csv("../input/train.csv")
data.head(3)
#to get the last 5 entries of the data

data.tail(5)
type(data)
data.shape
data.info()
data.isnull().sum()
data.info()
data.describe()
ms.matrix(data)
data.info()
sns.set_style('whitegrid')

sns.countplot(x='Survived',data=data,palette='RdBu_r')
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Sex',data=data,palette='RdBu_r')
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Pclass',data = data,palette='rainbow')
sns.distplot(data['Fare'])

#KDE?
data['Fare'].hist(color = 'green', bins = 40, figsize = (8,3))
data.corr()
sns.heatmap(data.corr(),cmap='coolwarm')

plt.title('data.corr()')
sns.swarmplot

sns.swarmplot(x='Pclass',y='Age',data=data,palette='Set1')
data['Age'].hist(bins = 40, color = 'darkred', alpha = 0.8)
data.info()
plt.figure(figsize=(12, 7))

sns.boxplot(x='Pclass',y='Age',data=data,palette='winter')
data['Age'].fillna(28, inplace=True)
data['Age'].median()
ms.matrix(data)
data['Cabin'].value_counts()
data.info()
data.drop('Cabin',axis=1, inplace=True)
data.head()
data['Embarked'].value_counts()
data.dropna(inplace = True) # dropping missing embarked.
ms.matrix(data)
data.info()
data['Sex'].value_counts()
sex = pd.get_dummies(data['Sex'],drop_first=True)

sex.head()
data['Embarked'].value_counts()
embark = pd.get_dummies(data['Embarked'],drop_first=True)

embark.head(10)
sex.head()
old_data = data.copy()

data.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)

data.head()

data = pd.concat([data,sex,embark],axis=1)
data.dropna(inplace = True) # dropping missing embarked.data.info()

data.info()
data.describe()
X = data.drop('Survived',axis=1)

y = data['Survived']
y.shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y 

                                                    , test_size=0.20, 

                                                    random_state=42)
X_test.shape
len(y_test)
178/889
X.describe()
X_train.describe()
y_train.describe()
from sklearn.linear_model import LogisticRegression



# Build the Model.

logmodel = LogisticRegression()

logmodel.fit(X_train,y_train) # this is where training happens
logmodel.coef_
logmodel.intercept_
predict =  logmodel.predict(X_test)

predict[:5]
y_test[:5]
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,predict))
print(confusion_matrix(y_test, predict))
from sklearn.metrics import precision_score
print(precision_score(y_test,predict))
from sklearn.metrics import recall_score
print(recall_score(y_test,predict))
from sklearn.metrics import f1_score
print(f1_score(y_test,predict))
from sklearn.metrics import classification_report
print(classification_report(y_test,predict))
prod_data=pd.read_csv('../input/test.csv')
prod_data.info()
ms.matrix(prod_data)
prod_data['Age'].fillna(28, inplace=True)
ms.matrix(prod_data)
prod_data.drop('Cabin', axis = 1, inplace= True)
ms.matrix(prod_data)
prod_data.fillna(prod_data['Fare'].mean(),inplace=True)
prod_data.info()
ms.matrix(prod_data)
sex = pd.get_dummies(prod_data['Sex'], drop_first=False)

embark = pd.get_dummies(prod_data['Embarked'], drop_first=False)
sex = pd.get_dummies(prod_data['Sex'], drop_first=False)

embark = pd.get_dummies(prod_data['Embarked'], drop_first=False)







prod_data.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
prod_data = pd.concat([prod_data,sex,embark],axis=1)
prod_data.head()
prod_data.drop(["female", 'C'], axis = 1, inplace = True)
prod_data.head()
prod_data.info()
prod_data['Fare'].fillna(prod_data['Fare'].median(), inplace = True)
prod_data.info()
predict1=logmodel.predict(prod_data)

predict1
df1=pd.DataFrame(predict1,columns=['Survived'])
df2=pd.DataFrame(prod_data['PassengerId'],columns=['PassengerId'])
df2.head()
result = pd.concat([df2,df1],axis=1)

result.head()
result.to_csv('result.csv',index=False)