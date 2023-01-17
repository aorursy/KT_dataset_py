import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt 

import seaborn as sns

import warnings 

warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)

from sklearn.model_selection import train_test_split, GridSearchCV
df_train = pd.read_csv("/kaggle/input/titanic/train.csv") 

df_test = pd.read_csv("/kaggle/input/titanic/test.csv")
train = df_train.copy() 

test = df_test.copy()
train.info()
train.head()
train.describe().T
train["Survived"].value_counts()
Survival_Rate = 342/891*100
print(str(Survival_Rate)+'%')
train["Sex"].value_counts()
train["Pclass"].value_counts()
train["SibSp"].value_counts()
train["Parch"].value_counts()
train["Embarked"].value_counts()
train["Cabin"].value_counts()
sns.catplot(x="Survived", kind="count", data=train);
sns.catplot(x="Sex", hue="Survived", kind="count", data=train);
sns.barplot(x = "Sex", y = "Survived", data = train);
sns.catplot(x="Pclass", hue="Survived", kind="count", data=train);
sns.barplot(x = "Pclass", y = "Survived", data = train);
sns.catplot(x="SibSp", hue="Survived", kind="count", data=train);
sns.barplot(x = "SibSp", y = "Survived", data = train);
sns.catplot(x="Parch", hue="Survived", kind="count", data=train);
sns.barplot(x = "Parch", y = "Survived", data = train);
plt.hist(train["Age"])

plt.show()
sns.boxplot(x="Survived", y="Age", data=train);
sns.boxplot(x="Survived", y="Fare", data=train);
train = train.drop(['Ticket'], axis = 1)

test = test.drop(['Ticket'], axis = 1)



train = train.drop(['Cabin'], axis = 1) 

test = test.drop(['Cabin'], axis = 1)
train = train.drop(['Name'], axis = 1)

test = test.drop(['Name'], axis = 1)
sns.boxplot(y="Fare", data=train);
Q1 = train['Fare'].quantile(0.25) 

Q3 = train['Fare'].quantile(0.75) 

IQR = Q3 - Q1
lower_limit = Q1 - 1.5*IQR 

lower_limit
upper_limit = Q3 + 1.5*IQR

upper_limit
train['Fare'] > (upper_limit)
train.sort_values("Fare", ascending=False).head()
test.sort_values("Fare", ascending=False)
count = 0

for i in train["Fare"] : 

    if i > 65.6344 :

        count = count + 1
print("The numbers of observations greater than upper limit of 65.6344 : " + str(count))
count = 0

for i in train["Fare"] : 

    if i > 200 :

        count = count + 1
print("The numbers of observations greater than 200 : " + str(count))
for i in train["Fare"] : 

    if i > 200 :

        train["Fare"].replace(i, 200, inplace=True)
train.sort_values("Fare", ascending=False).head()
for i in test["Fare"] : 

    if i > 200 :

        test["Fare"].replace(i, 200, inplace=True)
test.sort_values("Fare", ascending=False).head()
sns.boxplot(y="Fare", data=train);
train.describe().T
train.isnull().sum()
test.isnull().sum()
train["Age"] = train["Age"].fillna(train["Age"].mean())

test["Age"] = test["Age"].fillna(test["Age"].mean())

train.isnull().sum()
test.isnull().sum()
test["Fare"] = test["Fare"].fillna(test["Fare"].mean())
test.isnull().sum()
train["Embarked"].value_counts()
train["Embarked"] = train["Embarked"].fillna("S")
train["Embarked"].isnull().sum()
from sklearn.preprocessing import LabelEncoder

lbe = LabelEncoder()

lbe.fit_transform(train["Sex"])

train["Gender"] = lbe.fit_transform(train["Sex"])
train.drop(["Sex"], inplace = True, axis =1)
train.tail()
from sklearn.preprocessing import LabelEncoder

lbe = LabelEncoder()

lbe.fit_transform(train["Embarked"])

train["Embarked_new"] = lbe.fit_transform(train["Embarked"])
train.drop(["Embarked"], inplace = True, axis =1)
train.head()
lbe.fit_transform(test["Sex"])

test["Gender"] = lbe.fit_transform(test["Sex"])

test.drop(["Sex"], inplace = True, axis =1)

lbe.fit_transform(test["Embarked"])

test["Embarked_new"] = lbe.fit_transform(test["Embarked"])

test.drop(["Embarked"], inplace = True, axis =1)

test.head()
train["FamilySize"] = train["SibSp"] + train["Parch"] + 1

test["FamilySize"] = test["SibSp"] + test["Parch"] + 1
train.drop(["SibSp"], inplace = True, axis = 1)
test.drop(["SibSp"], inplace = True, axis = 1)
train.drop(["Parch"], inplace = True, axis = 1)
test.drop(["Parch"], inplace = True, axis = 1)
train.head()
test.head()
train = pd.get_dummies(train, columns = ["Gender"], prefix ="Gen") 

train = pd.get_dummies(train, columns = ["Embarked_new"], prefix="Em")

train = pd.get_dummies(train, columns = ["Pclass"], prefix="Pclass")

train = pd.get_dummies(train, columns = ["FamilySize"], prefix="Famsize")



test = pd.get_dummies(test, columns = ["Gender"], prefix ="Gen") 

test = pd.get_dummies(test, columns = ["Embarked_new"], prefix="Em")

test = pd.get_dummies(test, columns = ["Pclass"], prefix="Pclass")

test = pd.get_dummies(test, columns = ["FamilySize"], prefix="Famsize")
train.head()
test.head()
from sklearn.model_selection import train_test_split 

from sklearn.metrics import accuracy_score 

predictors = train.drop(['Survived', 'PassengerId'], axis=1) 

target = train["Survived"] 

x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.20, random_state = 42)
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
xgb_params = { 'n_estimators': [200, 500], 'subsample': [0.6, 1.0], 'max_depth': [2,5,8], 'learning_rate': [0.1,0.01,0.02], "min_samples_split": [2,5,10]}

xgb = GradientBoostingClassifier()

xgb_cv_model = GridSearchCV(xgb, xgb_params, cv = 10, n_jobs = -1, verbose = 2)

xgb_cv_model.fit(x_train, y_train)

xgb_cv_model.best_params_
xgb = GradientBoostingClassifier(learning_rate = 0.1, max_depth = 2, min_samples_split = 10, n_estimators = 200, subsample = 0.6)

xgb_tuned = xgb.fit(x_train,y_train)

y_pred = xgb_tuned.predict(x_test) 

acc_gbk = round(accuracy_score(y_pred, y_test) * 100, 2) 

print(acc_gbk)
ids = test['PassengerId'] 

predictions = randomforest.predict(test.drop('PassengerId', axis=1))
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions }) 

output.to_csv('submission.csv', index=False)

output.head()
output.describe().T
output["Survived"].value_counts()
Survival_Rate = 152/418*100
print(str(Survival_Rate)+'%')