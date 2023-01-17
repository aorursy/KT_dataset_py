import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
train = pd.read_csv('../input/titanic_train.csv')
train.head()
plt.figure(figsize=(16,8))
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.set_style('whitegrid')
plt.figure(figsize=(7,5))
sns.countplot(x='Survived',data=train)

plt.figure(figsize=(10,5))
sns.countplot(x='Survived',data=train,hue='Sex',palette='RdBu_r')
plt.figure(figsize=(14,7))
sns.countplot(x='Survived',data=train,hue='Pclass',palette='Blues')
plt.figure(figsize=(14,7))
sns.distplot(train['Age'].dropna(),kde=False,bins=30)
plt.figure(figsize=(14,7))
sns.countplot(x='SibSp',data=train,palette='hls')
plt.figure(figsize=(25,7))
train['Fare'].hist(bins=40)
plt.figure(figsize=(25,7))
sns.boxplot(x='Pclass',y='Age',data = train)
def findAge(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        elif Pclass == 3:
            return 24
    else:
        return Age


train['Age'] = train[['Age','Pclass']].apply(findAge,axis = 1)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
train.drop('Cabin',axis=1,inplace = True)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
train.dropna(inplace=True)
train.head(10)
sex = pd.get_dummies(train['Sex'],drop_first=True)
sex.head()
embark = pd.get_dummies(train['Embarked'],drop_first=True)
embark.head()
train = pd.concat([train,sex,embark],axis=1)
train.head(10)
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train.head()
train.drop('PassengerId',axis=1,inplace=True)
train.head()
X = train.drop('Survived',axis = 1)
y = train['Survived']
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,predictions))

