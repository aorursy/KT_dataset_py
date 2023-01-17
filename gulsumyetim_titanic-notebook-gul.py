pwd
import pandas as pd
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
train = train_data.copy()

test = test_data.copy()
train.head()
test.head()
train.info()
train.describe()
train['Pclass'].value_counts()
train['Age'].value_counts()
train['SibSp'].value_counts()
train['Embarked'].value_counts()
train['Cabin'].value_counts()
train['Ticket'].value_counts()
train['Parch'].value_counts()
train['Sex'].value_counts()
import matplotlib.pyplot as plt

import seaborn as sns
sns.barplot(x = 'Pclass', y = 'Survived', data= train);
sns.barplot(x = 'Sex', y = 'Survived', data= train);
sns.barplot(x = 'Embarked', y = 'Survived', data= train);
sns.barplot(x = 'Age', y = 'Survived', data= train);
train.head()
train = train.drop(['Ticket'], axis = 1)
test = test.drop(['Ticket'], axis = 1)
train.head()
train.describe().T
sns.boxplot(x = train['Fare']);
Q1 = train['Fare'].quantile(0.25)
Q3 = train['Fare'].quantile(0.75)
IQR = Q3 - Q1
lower_limit = Q1- 1.5*IQR

lower_limit
upper_limit = Q3 + 1.5*IQR

upper_limit
train['Fare'] > (upper_limit)
train.sort_values("Fare", ascending=False).head()
train["Fare"]=train["Fare"].replace(512.3292,300)
train.sort_values("Fare", ascending=False).head()
train.isnull().sum()
train["Age"]=train["Age"].fillna(train["Age"].mean())
test["Age"]=test["Age"].fillna(test["Age"].mean())
train.isnull().sum()
test.isnull().sum()
train["Embarked"].value_counts()
train["Embarked"]=train["Embarked"].fillna("S")
test["Embarked"].value_counts()
test["Embarked"]=test["Embarked"].fillna("S")
train.isnull().sum()
test.isnull().sum()
test[test["Fare"].isnull()]
test[["Pclass","Fare"]].groupby("Pclass").mean()
test["Fare"]=test["Fare"].fillna(12)
test["Fare"].isnull()
test["Fare"].isnull().sum()
train["CabinBool"] = (train["Cabin"].notnull().astype('int'))

test["CabinBool"] = (test["Cabin"].notnull().astype('int'))
train = train.drop(['Cabin'], axis = 1)
test  = test.drop(["Cabin"], axis = 1)
train.head()
test.head()
train.isnull().sum()
test.isnull().sum()
embarked_mapping = {"S": 1, "C": 2, "Q": 3}
train["Embarked"]=train["Embarked"].map(embarked_mapping)
test["Embarked"]=test["Embarked"].map(embarked_mapping)
train.head()
test.head()
from sklearn import preprocessing



lbe = preprocessing.LabelEncoder()

train["Sex"] = lbe.fit_transform(train["Sex"])

test["Sex"] = lbe.fit_transform(test["Sex"])
train.head()
train["Title"] = train["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)

test["Title"] = test["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)
train.head()
train["Title"].value_counts()
train['Title'] = train['Title'].replace(['Dr', 'Rev', 'Col', 'Major', 'Lady', 'Capt', 'Jonkheer', 'Dona','Don'], 'Rare')
train['Title'] = train['Title'].replace(['Countess', 'Sir'], 'Royal')

train['Title'] = train['Title'].replace('Mlle', 'Miss')
train['Title'] = train['Title'].replace('Ms', 'Miss')

train['Title'] = train['Title'].replace('Mme', 'Mrs')
train.head()
test['Title'] = test['Title'].replace(['Lady', 'Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')

test['Title'] = test['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')

test['Title'] = test['Title'].replace('Mlle', 'Miss')

test['Title'] = test['Title'].replace('Ms', 'Miss')

test['Title'] = test['Title'].replace('Mme', 'Mrs')
test.head()
train[["Title","PassengerId"]].groupby("Title").count()
train[["Title","Survived"]].groupby(["Title"], as_index=False).mean()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 5}



train['Title'] = train['Title'].map(title_mapping)
train.head()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 5}



test['Title'] = test['Title'].map(title_mapping)
test.head()
train.isnull().sum()
test.isnull().sum()
train=train.drop(["Name"],axis=1)
test=test.drop(["Name"],axis=1)
train.head()
test.head()
import numpy as np
bins = [0, 5, 12, 18, 24, 35, 60,  np.inf] 
mylabels = ['Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

train['AgeGroup'] = pd.cut(train["Age"], bins, labels = mylabels)

test['AgeGroup'] = pd.cut(test["Age"], bins, labels = mylabels)
age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}

train['AgeGroup'] = train['AgeGroup'].map(age_mapping)

test['AgeGroup'] = test['AgeGroup'].map(age_mapping)
train.head()
train=train.drop(["Age"],axis=1)
test=test.drop(["Age"],axis=1)
train.head()
test.head()
train['FareBand'] = pd.qcut(train['Fare'], 4, labels = [1, 2, 3, 4])

test['FareBand'] = pd.qcut(test['Fare'], 4, labels = [1, 2, 3, 4])
train=train.drop(["Fare"],axis=1)
test=test.drop(["Fare"],axis=1)
train.head()
test.head()
train["FareBand"].value_counts()
train["FamilySize"] = train_data["SibSp"] + train_data["Parch"] + 1
test["FamilySize"] = test_data["SibSp"] + test_data["Parch"] + 1
train.head()
train['Single'] = train['FamilySize'].map(lambda s: 1 if s == 1 else 0)

train['SmallFam'] = train['FamilySize'].map(lambda s: 1 if  s == 2  else 0)

train['MedFam'] = train['FamilySize'].map(lambda s: 1 if 3 <= s <= 4 else 0)

train['LargeFam'] = train['FamilySize'].map(lambda s: 1 if s >= 5 else 0)
train.head()
test['Single'] = test['FamilySize'].map(lambda s: 1 if s == 1 else 0)

test['SmallFam'] = test['FamilySize'].map(lambda s: 1 if  s == 2  else 0)

test['MedFam'] = test['FamilySize'].map(lambda s: 1 if 3 <= s <= 4 else 0)

test['LargeFam'] = test['FamilySize'].map(lambda s: 1 if s >= 5 else 0)
test.head()
train = pd.get_dummies(train, columns = ["Title"])

train = pd.get_dummies(train, columns = ["Embarked"], prefix="Em")
train.head()
test = pd.get_dummies(test, columns = ["Title"])

test = pd.get_dummies(test, columns = ["Embarked"], prefix="Em")
test.head()
train["Pclass"] = train["Pclass"].astype("category")

train = pd.get_dummies(train, columns = ["Pclass"],prefix="Pc")
train.head()
test["Pclass"] = test["Pclass"].astype("category")

test = pd.get_dummies(test, columns = ["Pclass"],prefix="Pc")
train.head()
test.head()
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score





X = train.drop(['Survived', 'PassengerId'], axis=1)

Y = train["Survived"]





x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 67)
x_train.head()
x_train.shape
x_test.shape
from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression()

logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test)

acc_logreg = round(accuracy_score(y_pred, y_test) * 100, 2)

print(acc_logreg)
from sklearn.ensemble import RandomForestClassifier



randomforest = RandomForestClassifier()

randomforest.fit(x_train, y_train)

y_pred = randomforest.predict(x_test)

acc_randomforest = round(accuracy_score(y_pred, y_test) * 100, 2)

print(acc_randomforest)
from sklearn.ensemble import GradientBoostingClassifier



gbk = GradientBoostingClassifier()

gbk.fit(x_train, y_train)

y_pred = gbk.predict(x_test)

acc_gbk = round(accuracy_score(y_pred, y_test) * 100, 2)

print(acc_gbk)
from sklearn.model_selection import train_test_split, GridSearchCV
xgb_params = {

        'n_estimators': [200, 500],

        'subsample': [0.6, 1.0],

        'max_depth': [2,5,8],

        'learning_rate': [0.1,0.01,0.02],

        "min_samples_split": [2,5,10]}

xgb = GradientBoostingClassifier()



xgb_cv_model = GridSearchCV(xgb, xgb_params, cv = 10, n_jobs = -1, verbose = 2)
xgb_cv_model.fit(x_train, y_train)
xgb = GradientBoostingClassifier(learning_rate = xgb_cv_model.best_params_["learning_rate"], 

                    max_depth = xgb_cv_model.best_params_["max_depth"],

                    min_samples_split = xgb_cv_model.best_params_["min_samples_split"],

                    n_estimators = xgb_cv_model.best_params_["n_estimators"],

                    subsample = xgb_cv_model.best_params_["subsample"])

xgb_tuned =  xgb.fit(x_train,y_train)
y_pred = xgb_tuned.predict(x_test)

acc_gbk = round(accuracy_score(y_pred, y_test) * 100, 2)

print(acc_gbk)
ids = test['PassengerId']

predictions = xgb_tuned.predict(test.drop('PassengerId', axis=1))



#set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.to_csv('submissiongul.csv', index=False)
output.head()