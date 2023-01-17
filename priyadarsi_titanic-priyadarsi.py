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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv('../input/titanic/train.csv')
train.head()
train.info()
train.describe()
sns.heatmap(train.isnull(),yticklabels=False,cbar=True,cmap='viridis')
sns.set_style('whitegrid')
sns.countplot(x='Survived',data = train, palette='RdBu_r')
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data = train,palette='rainbow')
sns.distplot(train['Age'].dropna(),kde=True,color='darkred',bins =30)
sns.countplot(x='SibSp',data=train)
train['Fare'].hist(color='green',bins=40,figsize=(8,4))
plt.figure(figsize=(12,7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')
def impute_age(cols):
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
train['Age']=train[['Age','Pclass']].apply(impute_age,axis=1)
sns.heatmap(train.isnull(),yticklabels=False,cbar= False,cmap='viridis')
train.drop('Cabin',axis=1,inplace=True)
train.head()
train.dropna(inplace=True)
sns.heatmap(train.isnull(),yticklabels=False,cbar= False,cmap='viridis')
sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace =True)
train= pd.concat([train,sex,embark],axis=1)
train.head()
from sklearn.model_selection import train_test_split
A_train, A_test, b_train, b_test = train_test_split(train.drop('Survived',axis=1),
                                                   train['Survived'], test_size=0.30,
                                                   random_state = 101)
train.info()
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(A_train, b_train)
predictions = logmodel.predict(A_test)
from sklearn.metrics import classification_report
print(classification_report(b_test,predictions))
X = train.drop('Survived',axis=1)
y = train['Survived']
X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1),
                                                   train['Survived'], test_size=0.30,
                                                   random_state = 101)
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
predictions = dtree.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=1000)
rfc.fit(X_train,y_train)
predictions = rfc.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,predictions))
print('\n')
print(confusion_matrix(y_test,predictions))
train
test = pd.read_csv('../input/titanic/test.csv')
test
sns.heatmap(test.isnull(),yticklabels=False,cbar= False,cmap='viridis')
test['Age']=test[['Age','Pclass']].apply(impute_age,axis=1)
test.drop('Cabin',axis=1,inplace=True)
test
sex1 = pd.get_dummies(test['Sex'],drop_first=True)
embark1 = pd.get_dummies(test['Embarked'],drop_first=True)
test= pd.concat([test,sex1,embark1],axis=1)
test.head()
train
test.columns
test.drop('Sex', axis=1, inplace= True)
test.drop('Embarked', axis=1, inplace= True)
test.drop('Name', axis=1, inplace= True)
test.drop('Ticket', axis=1, inplace= True)
test.columns
X_test = test
X_test
X_test = X_test.fillna(0)
from sklearn.tree import DecisionTreeClassifier 
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
sns.heatmap(test.isnull(),yticklabels=False,cbar= False,cmap='viridis')
train['Fare'] = train['Fare'].fillna(0)
predictions = dtree.predict(X_test)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=600)
rfc.fit(X_train,y_train)
predictions = rfc.predict(X_test)
X_test
predictions
len(predictions)
submission = pd.DataFrame()
submission['PassengerId']= test['PassengerId']
submission['Survived'] = predictions
submission.head()
submission.to_csv('randomforest_submission.csv',index=False)
