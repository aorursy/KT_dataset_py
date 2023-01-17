import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train_data = pd.read_csv('../input/titanic/train.csv')

train_data.head(10)
test_data = pd.read_csv('../input/titanic/test.csv')

test_data.head(10)
train_data.shape
test_data.shape
train_data.info()
test_data.info()
# Checking Missing values in train_data

train_data.isnull().sum()
# Checking Missing values in test_data

test_data.isnull().sum()
train_data.columns
train_data.head()
test_data.head()
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].mean())

test_data['Age'] = test_data['Age'].fillna(test_data['Age'].mean())
train_data.isnull().sum()
test_data.isnull().sum()
sns.catplot(x = 'Embarked', kind = 'count', data = train_data)
train_data['Embarked'] = train_data['Embarked'].fillna("S")
train_data.isnull().sum()
train_data['Cabin'] = train_data['Cabin'].fillna("Missing")

test_data['Cabin'] = test_data['Cabin'].fillna("Missing")
train_data.isnull().sum()
test_data.isnull().sum()
test_data['Fare'] = test_data['Fare'].median()
train_data.isnull().sum()
test_data.isnull().sum()
## get dummy variables for Column sex and embarked since they are categorical value.

train_data = pd.get_dummies(train_data, columns=["Sex"], drop_first=True)

train_data = pd.get_dummies(train_data, columns=["Embarked"],drop_first=True)





#Mapping the data.

train_data['Fare'] = train_data['Fare'].astype(int)

train_data.loc[train_data.Fare<=7.91,'Fare']=0

train_data.loc[(train_data.Fare>7.91) &(train_data.Fare<=14.454),'Fare']=1

train_data.loc[(train_data.Fare>14.454)&(train_data.Fare<=31),'Fare']=2

train_data.loc[(train_data.Fare>31),'Fare']=3



train_data['Age']=train_data['Age'].astype(int)

train_data.loc[ train_data['Age'] <= 16, 'Age']= 0

train_data.loc[(train_data['Age'] > 16) & (train_data['Age'] <= 32), 'Age'] = 1

train_data.loc[(train_data['Age'] > 32) & (train_data['Age'] <= 48), 'Age'] = 2

train_data.loc[(train_data['Age'] > 48) & (train_data['Age'] <= 64), 'Age'] = 3

train_data.loc[train_data['Age'] > 64, 'Age'] = 4
## get dummy variables for Column sex and embarked since they are categorical value.

test_data = pd.get_dummies(test_data, columns=["Sex"], drop_first=True)

test_data = pd.get_dummies(test_data, columns=["Embarked"],drop_first=True)





#Mapping the data.

test_data['Fare'] = test_data['Fare'].astype(int)

test_data.loc[test_data.Fare<=7.91,'Fare']=0

test_data.loc[(test_data.Fare>7.91) &(test_data.Fare<=14.454),'Fare']=1

test_data.loc[(test_data.Fare>14.454)&(test_data.Fare<=31),'Fare']=2

test_data.loc[(test_data.Fare>31),'Fare']=3



test_data['Age']=test_data['Age'].astype(int)

test_data.loc[ test_data['Age'] <= 16, 'Age']= 0

test_data.loc[(test_data['Age'] > 16) & (test_data['Age'] <= 32), 'Age'] = 1

test_data.loc[(test_data['Age'] > 32) & (test_data['Age'] <= 48), 'Age'] = 2

test_data.loc[(test_data['Age'] > 48) & (test_data['Age'] <= 64), 'Age'] = 3

test_data.loc[test_data['Age'] > 64, 'Age'] = 4
# In our data the Ticket and Cabin,Name are the base less,leds to the false prediction so Drop both of them.

train_data.drop(['Ticket','Cabin','Name'],axis=1,inplace=True)

test_data.drop(['Ticket','Cabin','Name'],axis=1,inplace=True)
train_data.describe()
train_data.Survived.value_counts()/len(train_data)*100

#This signifies almost 61% people in the ship died and 38% survived.
train_data.groupby("Survived").mean()
train_data.groupby("Sex_male").mean()
train_data.corr()
#Heatmap

plt.subplots(figsize=(10,8))

sns.heatmap(train_data.corr(),annot=True,cmap='Blues_r')

plt.title("Correlation Among Variables", fontsize = 20);
sns.barplot(x="Sex_male",y="Survived",data=train_data)

plt.title("Gender Distribution - Survived", fontsize = 16)
sns.barplot(x='Pclass',y='Survived',data=train_data)

plt.title("Passenger Class Distribution - Survived", fontsize = 16)
from sklearn.metrics import accuracy_score, log_loss

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn import metrics
X = train_data.drop(['Survived'], axis=1)

y = train_data["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.22, random_state = 5)
print(len(X_train),len(X_test),len(y_train),len(y_test))
#Logistic Regression

logReg = LogisticRegression()

logReg.fit(X_train,y_train)
logReg_predict = logReg.predict(X_test)

logReg_score = logReg.score(X_test,y_test)

# print("Logistic Regression Prediction :",logReg_predict)

print("Logistic Regression Score :",logReg_score)
print("Accuracy Score of Logistic Regression Model:")

print(metrics.accuracy_score(y_test,logReg_predict))

print("\n","Classification Report:")

print(metrics.classification_report(y_test,logReg_predict),'\n')
SVC_model = SVC(probability=True)

SVC_model.fit(X_train,y_train)
SVC_predict = SVC_model.predict(X_test)

SVC_score = SVC_model.score(X_test,y_test)

#print("Support Vector Classifier Prediction :",SVC_predict)

print("Support Vector Classifier Score :",SVC_score)
print("Accuracy Score of Support Vector Classifier SVC Model:")

print(metrics.accuracy_score(y_test,SVC_predict))

print("\n","Classification Report:")

print(metrics.classification_report(y_test,SVC_predict),'\n')
decisionTreeModel = DecisionTreeClassifier(max_leaf_nodes=17, random_state=0)

decisionTreeModel.fit(X_train, y_train)
decisionTree_predict = decisionTreeModel.predict(X_test)

decisionTree_score = decisionTreeModel.score(X_test,y_test)

#print("Decision Tree Classifier Prediction :",len(decisionTree_predict))

print("Decision Tree Classifier Score :",decisionTree_score)
print("Accuracy Score of Decision Tree Classifier Model:")

print(metrics.accuracy_score(y_test,decisionTree_predict))

print("\n","Classification Report:")

print(metrics.classification_report(y_test,decisionTree_predict),'\n')
Random_forest = RandomForestClassifier(n_estimators=17)

Random_forest.fit(X_train,y_train)
randomForest_predict = Random_forest.predict(X_test)

randomForest_score = Random_forest.score(X_test,y_test)

# print("Random Forest Prediction :",RF_predict)

print("Random Forest Score :",randomForest_score)
print("Accuracy Score of Random Forest Classifier Model:")

print(metrics.accuracy_score(y_test,randomForest_predict))

print("\n","Classification Report:")

print(metrics.classification_report(y_test,randomForest_predict),'\n')
KNN_model = KNeighborsClassifier(n_neighbors=37)

KNN_model.fit(X_train, y_train)
KNN_predict = KNN_model.predict(X_test)

KNN_score = KNN_model.score(X_test,y_test)

#print("KNN Classifier Prediction :",KNN_predict)

print("KNN Classifier Score :",KNN_score)
print("Accuracy Score of KNN Model:")

print(metrics.accuracy_score(y_test,KNN_predict))

print("\n","Classification Report:")

print(metrics.classification_report(y_test,KNN_predict),'\n')
from sklearn.ensemble import GradientBoostingClassifier



gbk = GradientBoostingClassifier(random_state=101, n_estimators=150,min_samples_split=100, max_depth=6)

gbk.fit(X_train, y_train)
gbk_predict = gbk.predict(X_test)

gbk_score = gbk.score(X_test,y_test)

#print("Gradient Boosting Prediction :",gbk_predict)

print("Gradient Boosting Score :",gbk_score)
print("Accuracy Score of Gradient Boosting Model:")

print(metrics.accuracy_score(y_test,gbk_predict))
from sklearn import ensemble

from sklearn.model_selection import GridSearchCV
GridList =[ {'n_estimators' : [10, 15, 20, 25, 30, 35, 40], 'max_depth' : [5,10,15, 20]},]

randomForest_ensemble = ensemble.RandomForestClassifier(random_state=31, max_features= 3)

gridSearchCV = GridSearchCV(randomForest_ensemble,GridList, cv = 5)
gridSearchCV.fit(X_train,y_train)
gridSearchCV_predict = gridSearchCV.predict(X_test)

gridSearchCV_score = gridSearchCV.score(X_test,y_test)

#print("Grid SearchCV Prediction :",gridSearchCV_predict)

print("Grid SearchCV Score :",gridSearchCV_score)
from tabulate import tabulate
print(tabulate([['K-Nearest Neighbour', KNN_score],['Logistic Regression',logReg_score ],['Decision Tree',decisionTree_score ],['Random Forest',randomForest_score ],['SVC', SVC_score],['Gradient Boosting', gbk_score],['Grid SearchCV',gridSearchCV_score]], headers=['Model Algorithm', 'Score']))
test_data.head()
#set ids as PassengerId and predict survival 

ids = test_data['PassengerId']

print(len(ids))

predictions = gridSearchCV.predict(test_data)
#set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.head(10) # Output preview
output.to_csv('submission.csv', index=False) # Submission csv file