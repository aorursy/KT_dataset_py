import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn import preprocessing
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data_train = pd.read_csv('../input/titanic/train.csv')

print("Shape of the training set : " ,data_train.shape)

data_test = pd.read_csv('../input/titanic/test.csv')

print("Shape of the test set : " ,data_test.shape)
train_copy = data_train

test_copy = data_test
data_train.head()
data_train.isnull().sum()
data_test.isnull().sum()
plt.figure(figsize=(15,15))

sns.kdeplot(data_train['Age'][data_train.Survived==0],shade=True)

sns.kdeplot(data_train['Age'][data_train.Survived==1],shade=True)

plt.legend(['Died','Survived'])



import warnings

warnings.filterwarnings('ignore')
plt.figure(figsize=(30,25))

sns.countplot(data=data_train,x='Age',hue='Survived')

plt.legend(['Died','Survived'])
median_age = data_train['Age'].median()

median_age
data_train['Age']=data_train['Age'].fillna(-1)

b = [-2,0,12,18,30,50,np.inf]

labels = ['U','I','T','Y','A','O']

data_train['Age']=pd.cut(data_train['Age'],b,labels=labels)
data_test['Age']=data_test['Age'].fillna(-1)

bt = [-2,0,12,18,30,50,np.inf]

labelst = ['U','I','T','Y','A','O']

data_test['Age']=pd.cut(data_test['Age'],bt,labels=labelst)
data_train.head(20)
#median_age_test = data_test['Age'].median()

#median_age_test
#data_test['Age']=data_test['Age'].fillna(-1)

#data_test['Age'] = data_test['Age'].astype(int)
plt.figure(figsize=(30,20))

sns.countplot(data=data_train,x='Age',hue='Survived')

plt.legend(['Died','Survived'])
#data_test['Fare']=data_test['Fare'].fillna(int(median_fare))

#data_test['Fare']=data_test['Fare'].astype(int)
data_train.isnull().sum()
plt.figure(figsize=(10,5))

plot= sns.countplot(data=data_train,x='Embarked')
data_train['Embarked'] = data_train['Embarked'].fillna('S')
data_test.isnull().sum()
data_test['Fare']=data_test['Fare'].fillna(int(data_test['Fare'].median()))
data_test.isnull().sum()
data_train = data_train.drop(columns=['Name','PassengerId','Ticket','Cabin'])

data_test = data_test.drop(columns=['Name','PassengerId','Ticket','Cabin'])
data_train['Family'] = data_train['SibSp'] + data_train['Parch']

data_test['Family'] = data_test['SibSp'] + data_test['Parch']
plt.figure(figsize=(20,20))

sns.countplot(data= data_train, x = 'Family',hue='Survived')

plt.legend(['Died','Survived'])
data_train.loc[data_train['Family']!= 0, 'o_fam'] = 1

data_train.loc[data_train['Family']== 0, 'o_fam'] = 0

data_train['o_fam'] = data_train['o_fam'].astype(int)
data_test.loc[data_test['Family']!= 0, 'o_fam'] = 1

data_test.loc[data_test['Family']== 0, 'o_fam'] = 0

data_test['o_fam'] = data_test['o_fam'].astype(int)
data_train.head(20)
data_test.head(20)
plt.figure(figsize=(20,15))

g = sns.catplot(x="o_fam", hue="Survived", col="Sex", data=data_train, kind="count", height=5, aspect=1)

plt.show()
data_train = pd.get_dummies(data_train,columns=['Sex','Embarked','Age'])

data_test = pd.get_dummies(data_test,columns=['Sex','Embarked','Age'])
data_train.head()
data_test.head()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

data_train['Fare'] = scaler.fit_transform(data_train[['Fare']])
data_test['Fare'] = scaler.fit_transform(data_test[['Fare']])
data_train.head()
data_test.head()
sns.kdeplot(data_train['Fare'][data_train.Survived==0],shade=True)

sns.kdeplot(data_train['Fare'][data_train.Survived==1],shade=True)

plt.legend(['Died','Survived'])
plt.figure(figsize=(15,15))

plot = sns.heatmap(data_train.corr(),annot=True)
data_train = data_train.drop(columns=['SibSp','Parch','Family'])
data_test = data_test.drop(columns=['SibSp','Parch','Family'])
data_train.head()
#converting test data

data_test.head()
X = data_train.drop('Survived',axis=1)

y = data_train['Survived']
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state = 1)
lr = LogisticRegression(max_iter=10)

lr.fit(X_train,y_train)
predict = lr.predict(X_test)
lr.score(X_train,y_train)
lr.score(X_test,y_test)
from sklearn.metrics import classification_report

print(classification_report(y_test,predict))
from sklearn import model_selection

rf = RandomForestClassifier(n_estimators=100,criterion='gini',max_depth=10, random_state=0)

rf.fit(X_train,y_train)
predrf=rf.predict(X_test)
rf.score(X_train,y_train)
rf.score(X_test,y_test)
print(classification_report(y_test,predrf))
svm = SVC(gamma='auto',C=1)

svm.fit(X_train,y_train)
predsvm=svm.predict(X_test)
svm.score(X_train,y_train)
svm.score(X_test,y_test)
print(classification_report(y_test,predsvm))
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(max_depth=6, min_samples_split=2)
dtc.fit(X_train,y_train)
preddtc = dtc.predict(X_test)
dtc.score(X_train,y_train)
dtc.score(X_test,y_test)
print(classification_report(y_test, preddtc))
pred = rf.predict(data_test)
Data_submit= test_copy[['PassengerId']]



Data_submit = pd.concat([Data_submit,pd.DataFrame(pred,columns=['Survived'])],axis=1)



Data_submit.to_csv('Submit.csv', index=False)