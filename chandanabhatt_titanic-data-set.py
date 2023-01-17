import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder 

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.metrics import cohen_kappa_score

from sklearn.metrics import confusion_matrix

from sklearn.svm import SVC

from sklearn.naive_bayes import MultinomialNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import RandomForestClassifier
data=pd.read_csv('../input/titanic.csv')
data.head()
cols=data.columns

cols
data.describe()
data.info()
data.Age.isnull().sum()
m=data.Age.mean()

m
data['Age']=data['Age'].fillna(m)
data.Age.isnull().sum()
sns.pairplot(data)
corr=data.corr()
plt.figure(figsize=(20,15))

sns.heatmap(corr,vmin=-1,vmax=1,annot=True)

plt.show() 
cols
data=data.drop('PassengerId',axis=1)
data=data.drop('Name',axis=1)

data=data.drop('Ticket',axis=1)

data.info()
data
sns.countplot(data['Sex'])

plt.show()
len(data[data['Sex']=='male']) #male are 577
len(data[data['Sex']=='female']) #femmale are 314
sns.countplot(data['Pclass'])

plt.show()
sns.countplot(data['Embarked'])

plt.show()
sns.lmplot('Age', 'Fare', hue ='Sex', data = data, fit_reg=True)
data.columns
data.Survived.isnull().sum()
data.Sex.isnull().sum()
data.SibSp.isnull().sum()
data.Parch.isnull().sum()
data.Fare.isnull().sum()
data.Cabin.isnull().sum()
data=data.drop('Cabin',axis=1)

data
y=data.Survived
y.head()
x=data.drop('Survived',axis=1)
x
x=x.drop('Embarked',axis=1)

x.head()
le=LabelEncoder()

x.Sex=le.fit_transform(x.Sex)
x.info()
x.head()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=123)
m=LogisticRegression()

model=m.fit(x_train,y_train)

y_pred=model.predict(x_test)
print('accuracy_score:',accuracy_score(y_pred,y_test))

print()

print('cohen_kappa_score',cohen_kappa_score(y_pred,y_test))

print()

print('classification_report',classification_report(y_pred,y_test))

print()

print('confusion_matrix',confusion_matrix(y_pred,y_test))
cm=confusion_matrix(y_pred,y_test)

sns.heatmap(cm, annot=True)
knn=KNeighborsClassifier(n_neighbors=10)

knn_fit=knn.fit(x_train,y_train)

y_knnpred = knn_fit.predict(x_test)
print('accuracy_score:',accuracy_score(y_pred,y_test))

print()

print('cohen_kappa_score',cohen_kappa_score(y_pred,y_test))

print()

print('classification_report',classification_report(y_pred,y_test))

print()

print('confusion_matrix',confusion_matrix(y_pred,y_test))
cm=confusion_matrix(y_pred,y_test)

sns.heatmap(cm, annot=True)
model=RandomForestClassifier()

model_fit=model.fit(x_train,y_train)

y_pred=model_fit.predict(x_test)
print('accuracy_score:',accuracy_score(y_pred,y_test))

print()

print('cohen_kappa_score',cohen_kappa_score(y_pred,y_test))

print()

print('classification_report',classification_report(y_pred,y_test))

print()

print('confusion_matrix',confusion_matrix(y_pred,y_test))
cm=confusion_matrix(y_pred,y_test)

sns.heatmap(cm, annot=True)
model=SVC()

model_fit=model.fit(x_train,y_train)

y_pred= model_fit.predict(x_test)
print('accuracy_score:',accuracy_score(y_pred,y_test))

print()

print('cohen_kappa_score',cohen_kappa_score(y_pred,y_test))

print()

print('classification_report',classification_report(y_pred,y_test))

print()

print('confusion_matrix',confusion_matrix(y_pred,y_test))
cm=confusion_matrix(y_pred,y_test)

sns.heatmap(cm, annot=True)
model=MultinomialNB()

mn_fit=model.fit(x_train,y_train)

y_pred=mn_fit.predict(x_test)
print('accuracy_score:',accuracy_score(y_pred,y_test))

print()

print('cohen_kappa_score',cohen_kappa_score(y_pred,y_test))

print()

print('classification_report',classification_report(y_pred,y_test))

print()

print('confusion_matrix',confusion_matrix(y_pred,y_test))
cm=confusion_matrix(y_pred,y_test)

sns.heatmap(cm, annot=True)
model=DecisionTreeClassifier()

mn_fit=model.fit(x_train,y_train)

y_pred=mn_fit.predict(x_test)
print('accuracy_score:',accuracy_score(y_pred,y_test))

print()

print('cohen_kappa_score',cohen_kappa_score(y_pred,y_test))

print()

print('classification_report',classification_report(y_pred,y_test))

print()

print('confusion_matrix',confusion_matrix(y_pred,y_test))
cm=confusion_matrix(y_pred,y_test)

sns.heatmap(cm, annot=True)
model=AdaBoostClassifier()

mn_fit=model.fit(x_train,y_train)

y_pred=mn_fit.predict(x_test)
print('accuracy_score:',accuracy_score(y_pred,y_test))

print()

print('cohen_kappa_score',cohen_kappa_score(y_pred,y_test))

print()

print('classification_report',classification_report(y_pred,y_test))

print()

print('confusion_matrix',confusion_matrix(y_pred,y_test))
cm=confusion_matrix(y_pred,y_test)

sns.heatmap(cm, annot=True)
model=RandomForestClassifier()

model_fit=model.fit(x_train,y_train)

y_pred=model_fit.predict(x_test)
model.feature_importances_
cm=confusion_matrix(y_pred,y_test)

sns.heatmap(cm, annot=True)