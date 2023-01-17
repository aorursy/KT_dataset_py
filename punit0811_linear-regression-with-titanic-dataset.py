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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
train = pd.read_csv('../input/titanic_train.csv')
train.head()
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis') #To get the columns which has null values
sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train)
sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r') #survival based on sex
sns.countplot(x='Survived',hue='Pclass',data=train) #survival based on passenger class
sns.distplot(train['Age'].dropna(),kde=False,bins=30)
sns.countplot(x='SibSp',data=train) # count based on sibling or spouse
train['Fare'].hist(bins=40,figsize=(10,4))
import cufflinks as cf
cf.go_offline()
train['Fare'].iplot(kind='hist',bins=40)
sns.countplot(x='Parch',data=train)
plt.figure(figsize=(10,7))
sns.boxplot(x='Pclass',y='Age',data=train)
def inpute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age
train['Age'] = train[['Age','Pclass']].apply(inpute_age,axis=1)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False)
train.drop('Cabin',axis=1,inplace=True)
train.dropna(inplace=True)
sex = pd.get_dummies(train['Sex'],drop_first=True)
sex.head()
embark = pd.get_dummies(train['Embarked'],drop_first=True)
embark.head()
train = pd.concat([train,sex,embark],axis=1)
train.head()
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train.head()
train.tail()
train.drop('PassengerId',axis=1,inplace=True)
train.head()
pclass = pd.get_dummies(train['Pclass'])
pclass.head()
X= train.drop('Survived',axis=1)
y= train['Survived'] #trying to predict
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)
