# linear algebra

import numpy as np 



# data processing

import pandas as pd 



# data visualization

import seaborn as sns

%matplotlib inline

from matplotlib import pyplot as plt

from matplotlib import style



# Algorithms

from sklearn import linear_model

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier
titanic_train = pd.read_csv("/kaggle/input/titanic/train.csv")

titanic_test = pd.read_csv("/kaggle/input/titanic/test.csv")

titanic_train.head(10)

titanic_train.info()

titanic_train.describe()
print("Cabin:","\n", titanic_train['Cabin'].describe())

print("Sex:","\n",titanic_train['Sex'].describe())

print("Embarked:","\n",titanic_train['Embarked'].describe())
titanic_train.isnull().sum()
Young_M= titanic_train['Name'].str.contains('Master', regex= True, na=False)

Young_F= titanic_train['Name'].str.contains('Miss', regex= True, na=False)

Adult_M= titanic_train['Name'].str.contains('Mr\.', regex= True, na=False)

Adult_F= titanic_train['Name'].str.contains('Mrs', regex= True, na=False)



print("Mean Age for Young Males: ",titanic_train.loc[Young_M,'Age'].mean())

print("Mean Age for Young Females: ",titanic_train.loc[Young_F,'Age'].mean())

print("Mean Age for Adult Males: ",titanic_train.loc[Adult_M,'Age'].mean())

print("Mean Age for Adult Females: ",titanic_train.loc[Adult_F,'Age'].mean())

titanic_train.loc[Young_M,'Age'] = titanic_train.loc[Young_M,'Age'].fillna(titanic_train.loc[Young_M,'Age'].mean())

titanic_train.loc[Young_F,'Age'] = titanic_train.loc[Young_F,'Age'].fillna(titanic_train.loc[Young_F,'Age'].mean())

titanic_train.loc[Adult_M,'Age'] = titanic_train.loc[Adult_M,'Age'].fillna(titanic_train.loc[Adult_M,'Age'].mean())

titanic_train.loc[Adult_F,'Age'] = titanic_train.loc[Adult_F,'Age'].fillna(titanic_train.loc[Adult_F,'Age'].mean())



titanic_train=titanic_train.fillna(method="bfill")

titanic_train=titanic_train.fillna(method="ffill")
titanic_train.isnull().sum()
titanic_train.head()
Young_M= titanic_test['Name'].str.contains('Master', regex= True, na=False)

Young_F= titanic_test['Name'].str.contains('Miss', regex= True, na=False)

Adult_M= titanic_test['Name'].str.contains('Mr\.', regex= True, na=False)

Adult_F= titanic_test['Name'].str.contains('Mrs', regex= True, na=False)



print("Mean Age for Young Males: ",titanic_test.loc[Young_M,'Age'].mean())

print("Mean Age for Young Females: ",titanic_test.loc[Young_F,'Age'].mean())

print("Mean Age for Adult Males: ",titanic_test.loc[Adult_M,'Age'].mean())

print("Mean Age for Adult Females: ",titanic_test.loc[Adult_F,'Age'].mean())



titanic_test.loc[Young_M,'Age'] = titanic_test.loc[Young_M,'Age'].fillna(titanic_test.loc[Young_M,'Age'].mean())

titanic_test.loc[Young_F,'Age'] = titanic_test.loc[Young_F,'Age'].fillna(titanic_test.loc[Young_F,'Age'].mean())

titanic_test.loc[Adult_M,'Age'] = titanic_test.loc[Adult_M,'Age'].fillna(titanic_test.loc[Adult_M,'Age'].mean())

titanic_test.loc[Adult_F,'Age'] = titanic_test.loc[Adult_F,'Age'].fillna(titanic_test.loc[Adult_F,'Age'].mean())



titanic_test=titanic_test.fillna(method="bfill")

titanic_test=titanic_test.fillna(method="ffill")



titanic_test.head()
titanic_test.isnull().sum()
titanic_train['Survived'].value_counts()

females = titanic_train.loc[titanic_train.Sex == 'female']["Survived"]

survived_F = sum(females)/len(females)

survived_F

males = titanic_train.loc[titanic_train.Sex == 'male']["Survived"]

survived_M = sum(males)/len(males)

survived_M
sur_sex = sns.countplot(x='Survived',hue='Sex',data=titanic_train,palette='CMRmap_r')

sur_sex.set_xticklabels(sur_sex.get_xticklabels(),rotation=90)

for p in sur_sex.patches:

    height = p.get_height()

    sur_sex.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")

sns.set(rc={'figure.figsize':(8,6)})
sur_pclass = sns.countplot(x='Survived',hue='Pclass',data=titanic_train,palette='ocean')

sur_pclass.set_xticklabels(sur_pclass.get_xticklabels(),rotation=90)

for p in sur_pclass.patches:

    height = p.get_height()

    sur_pclass.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")

sns.set(rc={'figure.figsize':(8,6)})
sur_embarked = sns.countplot(x='Survived',hue='Embarked',data=titanic_train,palette='spring')

sur_embarked.set_xticklabels(sur_embarked.get_xticklabels(),rotation=90)

for p in sur_embarked.patches:

    height = p.get_height()

    sur_embarked.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")

sns.set(rc={'figure.figsize':(10,6)})
titanic_train['family'] = titanic_train['SibSp'] + titanic_train['Parch'] + 1

sur_embarked = sns.countplot(x='Survived',hue='family',data=titanic_train,palette='plasma')

sur_embarked.set_xticklabels(sur_embarked.get_xticklabels(),rotation=90)

for p in sur_embarked.patches:

    height = p.get_height()

    sur_embarked.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")

sns.set(rc={'figure.figsize':(12,12)})
sns.boxplot(x='Survived', y='Fare', data=titanic_train, palette='autumn' )

sns.set(rc={'figure.figsize':(12,12)})
sns.boxplot(x='Survived', y='Age', data=titanic_train, palette='viridis' )

sns.set(rc={'figure.figsize':(10,10)})
corr = titanic_train.corr()

sns.heatmap(corr, annot=True)
from sklearn.preprocessing import OneHotEncoder

X_train = titanic_train[['Pclass','Age','SibSp','Parch','Fare']]

X_train = pd.concat([X_train,pd.get_dummies(titanic_train[['Sex','Embarked']])], axis=1)

print(X_train.head())

Y_train= titanic_train['Survived']

from sklearn.preprocessing import OneHotEncoder

X_test = titanic_test[['Pclass','Age','SibSp','Parch','Fare']]

X_test = pd.concat([X_test,pd.get_dummies(titanic_test[['Sex','Embarked']])], axis=1)

print(X_test.head())



LR_survived = LogisticRegression(solver='lbfgs',max_iter=10000)

LR_survived.fit(X_train, Y_train)

LR_pred = LR_survived.predict(X_test)

LR_Accuracy = "{:.2f} %".format(LR_survived.score(X_train,Y_train)*100)

print("Accuracy",LR_Accuracy)
SVC_survived = SVC()

SVC_survived.fit(X_train, Y_train)

SVC_pred = SVC_survived.predict(X_test)

SVC_Accuracy = "{:.2f} %".format(SVC_survived.score(X_train,Y_train)*100)

print("Accuracy",SVC_Accuracy)
KNN_survived = KNeighborsClassifier(n_neighbors = 3)

KNN_survived.fit(X_train, Y_train)

KNN_pred = KNN_survived.predict(X_test)

KNN_Accuracy = "{:.2f} %".format(KNN_survived.score(X_train,Y_train)*100)

print("Accuracy",KNN_Accuracy)
NB_survived = GaussianNB()

NB_survived.fit(X_train, Y_train)

NB_pred = NB_survived.predict(X_test)

NB_Accuracy = "{:.2f} %".format(NB_survived.score(X_train,Y_train)*100)

print("Accuracy",NB_Accuracy)
DT_survived = DecisionTreeClassifier()

DT_survived.fit(X_train, Y_train)

DT_pred = DT_survived.predict(X_test)

DT_Accuracy = "{:.2f} %".format(DT_survived.score(X_train,Y_train)*100)

print("Accuracy",DT_Accuracy)
RF_survived = RandomForestClassifier(n_estimators=10)

RF_survived.fit(X_train, Y_train)

RF_pred = RF_survived.predict(X_test)

RF_Accuracy = "{:.2f} %".format(RF_survived.score(X_train,Y_train)*100)

print("Accuracy",RF_Accuracy)
XGB_survived = XGBClassifier()

XGB_survived.fit(X_train,Y_train)

XGB_pred = XGB_survived.predict(X_test)

XGB_Accuracy = "{:.2f} %".format(XGB_survived.score(X_train,Y_train)*100)

print("Accuracy: ",XGB_Accuracy)
Models = pd.DataFrame({

    'Model': ['Logistic Regression','Support Vector Machines', 'KNN', 'Gaussian Naive Bayes','Decision Tree','Random Forest','XGBClassifier'],

    'Score': [LR_Accuracy,SVC_Accuracy, KNN_Accuracy, NB_Accuracy,DT_Accuracy, RF_Accuracy, XGB_Accuracy]})

Models.sort_values(by='Score', ascending=False)
# submission file from each model

SVC_file = pd.DataFrame({"PassengerId": titanic_test["PassengerId"], "Survived": SVC_pred})

SVC_file.to_csv('SVC_file.csv', index=False)



LR_file = pd.DataFrame({"PassengerId": titanic_test["PassengerId"], "Survived": LR_pred})

LR_file.to_csv('LR_file.csv', index=False)



KNN_file = pd.DataFrame({"PassengerId": titanic_test["PassengerId"], "Survived": KNN_pred})

KNN_file.to_csv('KNN_file.csv', index=False)



NB_file = pd.DataFrame({"PassengerId": titanic_test["PassengerId"], "Survived": NB_pred})

NB_file.to_csv('NB_file.csv', index=False)



DT_file = pd.DataFrame({"PassengerId": titanic_test["PassengerId"], "Survived": DT_pred})

DT_file.to_csv('DT_file.csv', index=False)



RF_file = pd.DataFrame({"PassengerId": titanic_test["PassengerId"], "Survived": RF_pred})

RF_file.to_csv('RF_file.csv', index=False)



XGB_file = pd.DataFrame({"PassengerId": titanic_test["PassengerId"], "Survived": XGB_pred})

XGB_file.to_csv('XGB_file.csv', index=False)