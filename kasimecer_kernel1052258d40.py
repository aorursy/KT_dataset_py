# data analysis libraries:

import numpy as np

import pandas as pd



# data visualization libraries:

import matplotlib.pyplot as plt

import seaborn as sns



# to ignore warnings:

import warnings

warnings.filterwarnings('ignore')



# to display all columns:

pd.set_option('display.max_columns', None)



from sklearn.model_selection import train_test_split, GridSearchCV
# Read train and test data with pd.read_csv():

train_data = pd.read_csv("train.csv")

test_data = pd.read_csv("test.csv")
# copy data in order to avoid any change in the original:

train = train_data.copy()

test = test_data.copy()
train.head()
test.head()
train.info()
train.describe().T
train['Pclass'].value_counts()
train['Sex'].value_counts()
train['SibSp'].value_counts()
train['Parch'].value_counts()
train['Ticket'].value_counts()
train['Cabin'].value_counts()
train['Embarked'].value_counts()
sns.barplot(x = 'Pclass', y = 'Survived', data = train);
sns.barplot(x = 'SibSp', y = 'Survived', data = train);
sns.barplot(x = 'Parch', y = 'Survived', data = train);
sns.barplot(x = 'Sex', y = 'Survived', data = train);
train.head()
# We can drop the Ticket feature since it is unlikely to have useful information

train = train.drop(['Ticket'], axis = 1)

test = test.drop(['Ticket'], axis = 1)



train.head()
train.describe().T
# It looks like there is a problem in Fare max data. Visualize with boxplot.

sns.boxplot(x = train['Fare']);
Q1 = train['Fare'].quantile(0.25)

Q3 = train['Fare'].quantile(0.75)

IQR = Q3 - Q1



lower_limit = Q1- 1.5*IQR

lower_limit



upper_limit = Q3 + 1.5*IQR

upper_limit
# observations with Fare data higher than the upper limit:



train['Fare'] > (upper_limit)
train.sort_values("Fare", ascending=False).head()
# In boxplot, there are too many data higher than upper limit; we can not change all. Just repress the highest value -512- 

train['Fare'] = train['Fare'].replace(512.3292, 300)
train.sort_values("Fare", ascending=False).head()
test.sort_values("Fare", ascending=False)
test['Fare'] = test['Fare'].replace(512.3292, 300)
test.sort_values("Fare", ascending=False)
train.isnull().sum()
train["Age"] = train["Age"].fillna(train["Age"].mean())
test["Age"] = test["Age"].fillna(test["Age"].mean())
train.isnull().sum()
test.isnull().sum()
train.isnull().sum()
test.isnull().sum()
train["Embarked"].value_counts()
# Fill NA with the most frequent value:

train["Embarked"] = train["Embarked"].fillna("S")
test["Embarked"] = test["Embarked"].fillna("S")
train.isnull().sum()
test.isnull().sum()
test[test["Fare"].isnull()]
test[["Pclass","Fare"]].groupby("Pclass").mean()
test["Fare"] = test["Fare"].fillna(12)
test["Fare"].isnull().sum()
# Create CabinBool variable which states if someone has a Cabin data or not:



train["CabinBool"] = (train["Cabin"].notnull().astype('int'))

test["CabinBool"] = (test["Cabin"].notnull().astype('int'))



train = train.drop(['Cabin'], axis = 1)

test = test.drop(['Cabin'], axis = 1)



train.head()
train.isnull().sum()
test.isnull().sum()
# Map each Embarked value to a numerical value:



embarked_mapping = {"S": 1, "C": 2, "Q": 3}



train['Embarked'] = train['Embarked'].map(embarked_mapping)

test['Embarked'] = test['Embarked'].map(embarked_mapping)
train.head()
# Convert Sex values into 1-0:



from sklearn import preprocessing



lbe = preprocessing.LabelEncoder()

train["Sex"] = lbe.fit_transform(train["Sex"])

test["Sex"] = lbe.fit_transform(test["Sex"])
train.head()
train["Title"] = train["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)

test["Title"] = test["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)
train.head()
train['Title'] = train['Title'].replace(['Lady', 'Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')

train['Title'] = train['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')

train['Title'] = train['Title'].replace('Mlle', 'Miss')

train['Title'] = train['Title'].replace('Ms', 'Miss')

train['Title'] = train['Title'].replace('Mme', 'Mrs')
test['Title'] = test['Title'].replace(['Lady', 'Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')

test['Title'] = test['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')

test['Title'] = test['Title'].replace('Mlle', 'Miss')

test['Title'] = test['Title'].replace('Ms', 'Miss')

test['Title'] = test['Title'].replace('Mme', 'Mrs')
train.head()
test.head()
train[["Title","PassengerId"]].groupby("Title").count()
train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
# Map each of the title groups to a numerical value



title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 5}



train['Title'] = train['Title'].map(title_mapping)
train.isnull().sum()
test['Title'] = test['Title'].map(title_mapping)
test.head()
train = train.drop(['Name'], axis = 1)

test = test.drop(['Name'], axis = 1)
train.head()
bins = [0, 5, 12, 18, 24, 35, 60, np.inf]

mylabels = ['Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

train['AgeGroup'] = pd.cut(train["Age"], bins, labels = mylabels)

test['AgeGroup'] = pd.cut(test["Age"], bins, labels = mylabels)
# Map each Age value to a numerical value:

age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}

train['AgeGroup'] = train['AgeGroup'].map(age_mapping)

test['AgeGroup'] = test['AgeGroup'].map(age_mapping)
train.head()
#dropping the Age feature for now, might change:

train = train.drop(['Age'], axis = 1)

test = test.drop(['Age'], axis = 1)
train.head()
# Map Fare values into groups of numerical values:

train['FareBand'] = pd.qcut(train['Fare'], 4, labels = [1, 2, 3, 4])

test['FareBand'] = pd.qcut(test['Fare'], 4, labels = [1, 2, 3, 4])
# Drop Fare values:

train = train.drop(['Fare'], axis = 1)

test = test.drop(['Fare'], axis = 1)
train.head()
train.head()
train["FamilySize"] = train_data["SibSp"] + train_data["Parch"] + 1
test["FamilySize"] = test_data["SibSp"] + test_data["Parch"] + 1
# Create new feature of family size:



train['Single'] = train['FamilySize'].map(lambda s: 1 if s == 1 else 0)

train['SmallFam'] = train['FamilySize'].map(lambda s: 1 if  s == 2  else 0)

train['MedFam'] = train['FamilySize'].map(lambda s: 1 if 3 <= s <= 4 else 0)

train['LargeFam'] = train['FamilySize'].map(lambda s: 1 if s >= 5 else 0)
train.head()
# Create new feature of family size:



test['Single'] = test['FamilySize'].map(lambda s: 1 if s == 1 else 0)

test['SmallFam'] = test['FamilySize'].map(lambda s: 1 if  s == 2  else 0)

test['MedFam'] = test['FamilySize'].map(lambda s: 1 if 3 <= s <= 4 else 0)

test['LargeFam'] = test['FamilySize'].map(lambda s: 1 if s >= 5 else 0)
test.head()
# Convert Title and Embarked into dummy variables:



train = pd.get_dummies(train, columns = ["Title"])

train = pd.get_dummies(train, columns = ["Embarked"], prefix="Em")
train.head()
test = pd.get_dummies(test, columns = ["Title"])

test = pd.get_dummies(test, columns = ["Embarked"], prefix="Em")
test.head()
# Create categorical values for Pclass:

train["Pclass"] = train["Pclass"].astype("category")

train = pd.get_dummies(train, columns = ["Pclass"],prefix="Pc")
test["Pclass"] = test["Pclass"].astype("category")

test = pd.get_dummies(test, columns = ["Pclass"],prefix="Pc")
train.head()
test.head()
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

predictors = train.drop(['Survived', 'PassengerId'], axis=1)

target = train["Survived"]

x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.20, random_state = 0)
train
predictors
target
print(x_train.head())

x_train.shape
print(x_test.head())

x_test.shape
print(y_train.head())

y_train.shape
print(y_test.head())

y_test.shape
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
xgb_params = {

        'n_estimators': [ 200, 210, 230],

        'subsample': [ 0.6, 0.7, 0.8],

        'max_depth': [1.8, 2],

        'learning_rate': [0.01, 0.02, 0.025],

        "min_samples_split": [2, 3, 4, 6]}
xgb_params = {

        'n_estimators': [200, 500],

        'subsample': [0.6, 1.0],

        'max_depth': [2,5,8],

        'learning_rate': [0.1, 0.01, 0.02],

        "min_samples_split": [2,5,10]}
xgb = GradientBoostingClassifier()



xgb_cv_model = GridSearchCV(xgb, xgb_params, cv = 10, n_jobs = -1, verbose = 2)
xgb_cv_model.fit(x_train, y_train)
xgb_cv_model.best_params_
{'learning_rate': 0.02,

 'max_depth': 2,

 'min_samples_split': 5,

 'n_estimators': 200,

 'subsample': 0.6}
xgb = GradientBoostingClassifier(learning_rate = xgb_cv_model.best_params_["learning_rate"], 

                    max_depth = xgb_cv_model.best_params_["max_depth"],

                    min_samples_split = xgb_cv_model.best_params_["min_samples_split"],

                    n_estimators = xgb_cv_model.best_params_["n_estimators"],

                    subsample = xgb_cv_model.best_params_["subsample"])
xgb_tuned =  xgb.fit(x_train,y_train)
y_pred = xgb_tuned.predict(x_test)

acc_gbk = round(accuracy_score(y_pred, y_test) * 100, 2)

print(acc_gbk)
test
#set ids as PassengerId and predict survival 

ids = test['PassengerId']

predictions = xgb_tuned.predict(test.drop('PassengerId', axis=1))



#set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.to_csv('submission.csv', index=False)
output.head()