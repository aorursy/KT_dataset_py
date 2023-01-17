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
#import warnings

import warnings
warnings.filterwarnings('ignore')
#import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import train data set

train = pd.read_csv('/kaggle/input/titanic/train.csv')
train.head()
#import test data set

test = pd.read_csv('/kaggle/input/titanic/test.csv')
test.head()
#describe train data set

print(train.shape)
print(train.info())
print(train.describe())
#check data type for p-class
print('Train[pclass]:',train.Pclass.dtype)
#change it to object data type
train['Pclass'] = train['Pclass'].astype('object')
print('Train[pclass]:',train.Pclass.dtype)
#decoding p-class column
train['Pclass'] = train['Pclass'].replace({1:'1st',2:'2nd',3:'3rd'}).astype('object')
#counting the values
train['Pclass'].value_counts(ascending=False)
#decoding embarked column
train['Embarked'] = train['Embarked'].replace({'C':'Cherbourge','Q':'Queenstown','S':'Southampton'}).astype('object')
#counting the values
train['Embarked'].value_counts(ascending=False)
#describe test dataset

print(test.shape)
print(test.info())
print(test.describe())
#check the datatype for pclass

print('test[Pclass]:',test['Pclass'].dtype)
#Change it to object datatype
test['Pclass'] = test['Pclass'].astype('object')
print('test[Pclass]:',test['Pclass'].dtype)
#decoding values for Pclass col
test['Pclass'] = test['Pclass'].replace({1:'1st',2:'2nd',3:'3rd'}).astype('object')
#counting the values
test['Pclass'].value_counts()
#decoding values for Embarked col
test['Embarked'] = test['Embarked'].replace({'C':'Cherbourge','Q':'Queenstown','S':'Southampton'}).astype('object')
#counting the values
test['Embarked'].value_counts()
#findout missing values
train.isnull().sum()
#for cabin
train['Cabin'].describe()
train.Cabin[train['Cabin'].notnull()].head()
#fill the missing value
train['Cabin'] = pd.Series(i[0] if not pd.isnull(i) else 'J' for i in train['Cabin'])
#check the null values again
print(train['Cabin'].isnull().sum())
train.Cabin.value_counts()
#for age
train['Age'].describe()
train['Age'] = train['Age'].fillna(train['Age'].median())
#check the null values again
print(train['Age'].isnull().sum())
train['Embarked'].value_counts()
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])
#check the null values again
train['Embarked'].isnull().sum()
#findout missing values
test.isnull().sum()
test['Age'].describe()
#fill missing values with median
test['Age'] = test['Age'].fillna(test['Age'].median())
#check the null values again if any
test['Age'].isnull().sum()
test['Cabin'].describe()
#fill the missing value
test['Cabin'] = pd.Series(i[0] if not pd.isnull(i) else 'J' for i in test['Cabin'])
#check the null values again
print(test['Cabin'].isnull().sum())
test.Cabin.value_counts()
test['Fare'] = test['Fare'].fillna(test['Fare'].median())
#check null values if any
test['Fare'].isnull().sum()
print(train.isnull().sum())
print(test.isnull().sum())
#pairplot for train set
sns.pairplot(train)
#heatmap
plt.figure(figsize=(10,8))
sns.heatmap(train.corr(),annot=True,cmap='RdYlGn')
#heatmap for testset
plt.figure(figsize=(8,6))
sns.heatmap(test.corr(),annot=True,cmap='summer')
plt.figure(figsize=(15,8))
plt.subplot(1,2,1)
colors = ['gold', 'yellowgreen', 'lightcoral']
explode = (0.1, 0, 0) #explode 1st slice
train.Pclass.value_counts().plot.pie(explode=explode,colors=colors,autopct='%1.1f%%',shadow=True,startangle=140)
plt.title('Passenger by class')
plt.subplot(1,2,2)
sns.barplot(x='Pclass',y='Survived',data=train)
plt.title('Pclass vs Survival')
plt.show()
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.countplot(x='Sex',hue='Pclass',data=train,palette='YlOrBr')
plt.title('Sex vs Pclass')
plt.subplot(1,2,2)
sns.barplot(x='Pclass',y='Survived',hue='Sex',data=train)
plt.ylabel('Survival Rate')
plt.title('Sex & Passenger Class Vs Survival rate')
plt.show()
plt.figure(figsize=(10,6))
sns.distplot(train[train['Survived']==0].Age,bins=20, kde=False, color='r',label= 'not survived')
sns.distplot(train[train['Survived']==1].Age,bins=20 ,kde=False, color='b',label= 'survived')
plt.title('Age Vs Survival rate',fontweight='bold',size=15)
plt.legend()
#plt.ylabel('Survival rate')
plt.show()
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.countplot(x='Embarked',hue='Pclass',data=train,palette='gist_heat')
plt.title('Embarked vs Pclass')
plt.subplot(1,2,2)
sns.barplot(x='Pclass',y='Survived',hue='Embarked',data=train)
plt.ylabel('Survival Rate')
plt.title('Embarked & Passenger Class Vs Survival rate')
plt.show()
plt.figure(figsize=(12,6))
sns.barplot(x='Parch',y='Survived',data=train,palette='tab20')
plt.title("Parent/Child Vs Survival Rate")
plt.ylabel("Survival rate")
plt.show()
plt.figure(figsize=(12,6))
sns.barplot(x='SibSp',y='Survived',data=train,palette='tab20')
plt.title("Sibling/Spouse Vs Survival Rate")
plt.ylabel("Survival rate")
plt.show()
train['Family'] = train['SibSp']+train['Parch']+1
test['Family'] = test['SibSp']+test['Parch']+1
plt.figure(figsize=(10,6))
sns.countplot('Family',hue='Pclass',data=train)
plt.show()
plt.figure(figsize=(10,6))
sns.barplot(x='Pclass', y='Family',hue='Survived',data=train,palette='mako')
plt.title('Family & Pclass Vs Survival rate')

plt.show()
plt.figure(figsize=(8,6))
train.groupby('Pclass')['Fare'].sum().plot(kind='bar')
plt.show()
plt.figure(figsize=(10,6))
sns.barplot('Cabin','Survived', hue='Pclass',data=train)
plt.ylabel('Survival Rate')
plt.show()
#Check columns
train.columns
#Drop above col from both train & test data.

train.drop(['Name','SibSp','Parch','Ticket'],axis=1,inplace=True)
test.drop(['Name','SibSp','Parch','Ticket'],axis=1,inplace=True)
#Convert Pclass column 
code = {'1st':1, '2nd':2, '3rd':3}
train['Pclass'] = train['Pclass'].map(code)
test['Pclass'] = test['Pclass'].map(code)
#Convert Sex
code={"male":0, "female":1}
train['Sex'] = train['Sex'].map(code)
test['Sex'] = test['Sex'].map(code)
#Convert Embarked
code={"Cherbourge":1, "Queenstown":2,"Southampton":3}
train['Embarked'] = train['Embarked'].map(code)
test['Embarked'] = test['Embarked'].map(code)
#Convert Cabin
code = {'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'T':8,'J':9}
train['Cabin'] = train['Cabin'].map(code)
#Convert Cabin
code = {'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'J':8}
test['Cabin'] = test['Cabin'].map(code)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

train[['Age','Fare']] = scaler.fit_transform(train[['Age','Fare']])
test[['Age','Fare']] = scaler.transform(test[['Age','Fare']])
#check the train data
train.head()
#Check test data
test.head()
X_train = train.drop(['PassengerId','Survived'],axis=1)
y_train = train['Survived']
X_test = test.drop(['PassengerId'],axis=1).copy()
X_train.shape, y_train.shape, X_test.shape
#import libraries
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()

#Fitting the model
LR.fit(X_train,y_train)
#making prediction
y_pred = LR.predict(X_test) 
#calculating accuracy of our model
print("Accuracy:",round(LR.score(X_train,y_train)*100,2))
LR.coef_
X_train.columns
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Cabin', 'Embarked', 'Family']
coeff = pd.DataFrame(X_train.columns)
coeff.columns = ['Feature']
coeff['Correlation'] = pd.Series(LR.coef_[0])

coeff.sort_values(by='Correlation', ascending=False)
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators=100,random_state=22)

#fitting model on trained set
RF.fit(X_train,y_train)
#making prediction
y_pred = RF.predict(X_test)
#calculating accuracy
print("Accuracy:",round(RF.score(X_train,y_train)*100,2))
feature_1 = pd.Series(RF.feature_importances_,index=features).sort_values(ascending=False)
feature_1
Submit_file = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':y_pred})
Submit_file.to_csv('Titanic_predict.csv', index=False)