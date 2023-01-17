import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 

import seaborn as sns



import matplotlib.pyplot as plt



%matplotlib inline
data = pd.read_csv('../input/train.csv')   #读取数据集

data.head(5)
data.info()
data.describe()
print("survived passengers are ",data.Survived.sum())   #存活的乘客
sns.countplot(x='Survived',data=data)
sns.countplot(x='Survived',hue='Sex',data=data)
sns.countplot(x='Survived',hue='Pclass',data=data)
sns.countplot(x='SibSp',data=data)
data['Fare'].plot.hist()
data.hist(figsize=(12,8))

plt.figure()
print('Oldest Passenger was of:',data['Age'].max(),'Years')

print('Youngest Passenger was of:',data['Age'].min(),'Years')

print('Average Age on the ship:',data['Age'].mean(),'Years')
data['Age'].plot.hist()
grid = sns.FacetGrid(data, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', bins=30)
grid = sns.FacetGrid(data, col='Survived', row='Sex', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', bins=15)
f,ax=plt.subplots(1,2,figsize=(12,6))

sns.violinplot("Pclass","Age", hue="Survived", data=data,split=True,ax=ax[0])

ax[0].set_title('Pclass and Age vs Survived')

ax[0].set_yticks(range(0,110,10))

sns.violinplot("Sex","Age", hue="Survived", data=data,split=True,ax=ax[1])

ax[1].set_title('Sex and Age vs Survived')

ax[1].set_yticks(range(0,110,10))

plt.show()
sns.boxplot(x='Pclass',y='Age',data=data)
f,ax=plt.subplots(2,2,figsize=(15,8))

sns.countplot('Embarked',data=data,ax=ax[0,0])

ax[0,0].set_title('No. Of Passengers Boarded')

sns.countplot('Embarked',hue='Sex',data=data,ax=ax[0,1])

ax[0,1].set_title('Male-Female Split for Embarked')

sns.countplot('Embarked',hue='Survived',data=data,ax=ax[1,0])

ax[1,0].set_title('Embarked vs Survived')

sns.countplot('Embarked',hue='Pclass',data=data,ax=ax[1,1])

ax[1,1].set_title('Embarked vs Pclass')

plt.subplots_adjust(wspace=0.2,hspace=0.5)

plt.show()
sns.barplot(x='Pclass', y='Survived', data=data)
data.corr()
data.groupby('Pclass',as_index=False)['Survived'].mean()
data.groupby('Sex',as_index=False)['Survived'].mean()
data.groupby('Embarked',as_index=False)['Survived'].mean()
data.groupby('SibSp',as_index=False)['Survived'].mean().sort_values(by='Survived',ascending=False)
data.groupby('Parch',as_index=False)['Survived'].mean().sort_values(by='Survived',ascending=False)
fig,ax = plt.subplots(figsize=(8,7))

ax = sns.heatmap(data.corr(), annot=True,linewidths=.5,fmt='.1f')

plt.show()
gender=pd.get_dummies(data['Sex'],prefix='sx',drop_first=True)

gender.head()
embarked=pd.get_dummies(data['Embarked'],prefix='emb',drop_first=True)

embarked.head()
pcl=pd.get_dummies(data['Pclass'],prefix='pcl',drop_first=True)

pcl.head()
data=pd.concat([data,gender,embarked,pcl],axis=1)
total=data.isnull().sum()

percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(13)
sns.heatmap(data.isnull(),cmap='viridis')
data.drop('Cabin',axis=1,inplace=True)

data.head()
data.dropna(inplace=True)
data.drop(['Pclass','Sex','PassengerId','Name','Ticket'],axis=1,inplace=True)
data.drop(['Age','Fare','Embarked'],axis=1,inplace=True)
data.head()
from sklearn.linear_model import LogisticRegression #logistic regression

from sklearn import svm #support vector Machine

from sklearn.ensemble import RandomForestClassifier #Random Forest

from sklearn.neighbors import KNeighborsClassifier #KNN

from sklearn.naive_bayes import GaussianNB #Naive bayes

from sklearn.tree import DecisionTreeClassifier #Decision Tree

from sklearn.model_selection import train_test_split #training and testing data split

from sklearn import metrics #accuracy measure

from sklearn.metrics import confusion_matrix #for confusion matrix
train,test=train_test_split(data,test_size=0.3,random_state=0,stratify=data['Survived'])

train_X=train[train.columns[1:]]

train_Y=train[train.columns[:1]]

test_X=test[test.columns[1:]]

test_Y=test[test.columns[:1]]

X=data[data.columns[1:]]

Y=data['Survived']

model=svm.SVC(kernel='rbf',C=1,gamma=0.1)

model.fit(train_X,train_Y)

prediction1=model.predict(test_X)

print('Accuracy is ',metrics.accuracy_score(prediction1,test_Y))
model=svm.SVC(kernel='linear',C=0.1,gamma=0.1)

model.fit(train_X,train_Y)

prediction2=model.predict(test_X)

print('Accuracy for linear SVM is',metrics.accuracy_score(prediction2,test_Y))
from sklearn.model_selection import KFold #for K-fold cross validation

from sklearn.model_selection import cross_val_score #score evaluation

from sklearn.model_selection import cross_val_predict #prediction

from sklearn.ensemble import AdaBoostClassifier

ada=AdaBoostClassifier(n_estimators=200,random_state=0,learning_rate=0.05)

result=cross_val_predict(ada,X,Y,cv=10)

sns.heatmap(confusion_matrix(Y,result),cmap='winter',annot=True,fmt='2.0f')

plt.show()
test = pd.read_csv('../input/test.csv')
embarked=pd.get_dummies(test['Embarked'],prefix='emb',drop_first=True)

pcl=pd.get_dummies(test['Pclass'],prefix='pcl',drop_first=True)

gender=pd.get_dummies(test['Sex'],prefix='sx',drop_first=True)

test=pd.concat([test,gender,embarked,pcl],axis=1)
test.drop(['Pclass','Sex','PassengerId','Name','Ticket','Cabin','Age','Fare','Embarked'],axis=1,inplace=True)
data

train_X = data[data.columns[1:]]

train_Y = data[data.columns[0]]

test_X = test
model=svm.SVC(kernel='linear',C=0.1,gamma=0.1)

model.fit(train_X,train_Y)

prediction=model.predict(test_X)

prediction = pd.DataFrame(prediction)

prediction.columns=['Survived']
raw_test = pd.read_csv('../input/test.csv')

submission = {'PassengerId': raw_test['PassengerId'],

              'Survived':prediction['Survived']}

submission = pd.DataFrame(submission)
### submission.to_csv('location',index=False)