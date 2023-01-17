import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train = pd.read_csv("../input/train.csv")

train.head(5) 
train.isnull().sum()
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis');

train.drop('Cabin',axis=1,inplace=True)

train['Age'].fillna((train['Age'].median()), inplace=True)

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis');

sns.set_style('whitegrid')

sns.countplot(x='Survived',data=train,palette='RdBu_r');
sns.set_style('darkgrid')

sns.countplot(x='Survived',hue='Sex',data=train,palette='ocean');
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Pclass',data=train,palette='winter');
sns.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=20);
sns.countplot(x='SibSp',data=train,palette='ocean');
sns.countplot(x='Parch',data=train,palette='ocean');
plt.figure(figsize=(12, 7))

sns.boxplot(x='Pclass',y='Age',data=train,palette='winter');
train.info()
# create the dummy variables and drop one column as there is no need of 2 columns in order to differentiate the values.

sex = pd.get_dummies(train['Sex'],drop_first=True)

# similarly for this colimn as well. If there are n dummy columns, consider n-1

embark = pd.get_dummies(train['Embarked'],drop_first=True)
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)

train = pd.concat([train,sex,embark],axis=1)
train.head()

from sklearn.model_selection import train_test_split

X = train.drop("Survived",axis=1)

y = train['Survived']

#X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1),train['Survived'], test_size=0.20,random_state=5)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=5)
from sklearn.linear_model import LogisticRegression

# create an instance

logmodel = LogisticRegression()

# pass the values and build the model

logmodel.fit(X_train,y_train)
# preditcing the test models

predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

print(accuracy_score(y_test,predictions)*100)
from sklearn.tree import DecisionTreeClassifier

dt_model=DecisionTreeClassifier()

dt_model.fit(X_train,y_train)

dt_pred = dt_model.predict(X_test)

print(confusion_matrix(y_test,dt_pred))

print(classification_report(y_test,dt_pred))

print(accuracy_score(y_test,dt_pred)*100)
from sklearn.ensemble import RandomForestClassifier

rf= RandomForestClassifier(n_estimators=100)

rf.fit(X_train,y_train)
rf_pre=rf.predict(X_test)

print(confusion_matrix(y_test,rf_pre))

print(classification_report(y_test,rf_pre))

print(accuracy_score(y_test,rf_pre)*100)
test = pd.read_csv("../input/test.csv")

test.head(5) 
sns.heatmap(test.isnull());
test.info()
test.drop('Cabin',axis=1,inplace=True)

test['Age'].fillna((test['Age'].median()), inplace=True)

test['Fare'].fillna((test['Fare'].median()), inplace=True)

sex_test = pd.get_dummies(test['Sex'],drop_first=True)

embark_test= pd.get_dummies(test['Embarked'],drop_first=True)

test.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)

test = pd.concat([test,sex_test,embark_test],axis=1)

test.head()
prediction = logmodel.predict(test)

prediction
test_pred = pd.DataFrame(prediction, columns= ['Survived'])
Survived_dataset = pd.concat([test, test_pred], axis=1, join='inner')
Survived_dataset.head()
dataset = Survived_dataset[['PassengerId','Survived']]

dataset.head(10)
data_to_submit = pd.DataFrame(Survived_dataset[['PassengerId','Survived']])
data_to_submit.to_csv('csv_to_submit.csv', index = False)
