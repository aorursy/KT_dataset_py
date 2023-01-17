

import numpy as np

import pandas as pd 

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt

from sklearn.preprocessing import scale 

from sklearn.preprocessing import StandardScaler

from sklearn import model_selection

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.neural_network import MLPRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn import neighbors

from sklearn.svm import SVR

import seaborn as sns

%matplotlib inline

import missingno as msno

# to ignore warnings:

import warnings

warnings.filterwarnings('ignore')

# to display all columns:

pd.set_option('display.max_columns', None)

from sklearn.model_selection import train_test_split, GridSearchCV
# Read train and test data with pd.read_csv():

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
# copy data in order to avoid any change in the original:

train = train_data.copy()

test = test_data.copy()
train.head()
train.tail()
test.head()
train.shape
test.shape
train.ndim
test.ndim
train.describe().T
test.describe().T
train.columns
test.columns
test.dtypes
train.dtypes
train.info()

# Correlation matrix between numerical values (SibSp Parch Age and Fare values) and Survived 

g = sns.heatmap(train[["Survived","SibSp","Parch","Age","Fare"]].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
train["Age"].describe()
train["Age"].value_counts()
g = sns.FacetGrid(train, col='Survived')

g.map(plt.hist, 'Age', bins=20)
train["SibSp"].value_counts()
train["SibSp"].value_counts().plot.barh();
sns.barplot(x="SibSp", y="Survived", data=train);
train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train["Parch"].value_counts()
sns.barplot(x="Parch", y="Survived", data=train)

plt.show()
train["Fare"].value_counts()
sns.boxplot(x = train['Fare']);
Q1 = train['Fare'].quantile(0.25)

Q3 = train['Fare'].quantile(0.75)

IQR = Q3 - Q1



lower_limit = Q1- 1.5*IQR

print(lower_limit)



upper_limit = Q3 + 1.5*IQR

upper_limit
train["Survived"].value_counts()
train["Survived"].value_counts().plot.barh();
train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train["Sex"].value_counts()
train["Sex"].value_counts().plot.barh();
sns.catplot(x = "Sex", y = "Age", hue= "Survived",data = train);
sns.barplot(x="Sex", y="Survived", data=train)
train["Pclass"].value_counts()
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
sns.barplot(x="Pclass", y="Survived", data=train);
print("Pclass Percantage = 1  survived:", train["Survived"][train["Pclass"] == 1].value_counts(normalize = True)[1]*100)



print("Pclass Percantage = 2  survived :", train["Survived"][train["Pclass"] == 2].value_counts(normalize = True)[1]*100)

print("Pclass Percantage = 3 survived:", train["Survived"][train["Pclass"] == 3].value_counts(normalize = True)[1]*100)
train["Embarked"].value_counts()
g = sns.factorplot(x="Embarked", y="Survived",  data=train,

                   size=6, kind="bar", palette="muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
g = sns.factorplot("Pclass", col="Embarked",  data=train,

                   size=6, kind="count", palette="muted")

g.despine(left=True)

g = g.set_ylabels("Count")
train["Cabin"].value_counts()
train.isnull().sum()
g = sns.factorplot(y="Age",x="Sex",data=train,kind="box")

g = sns.factorplot(y="Age",x="Sex",hue="Pclass", data=train,kind="box")

g = sns.factorplot(y="Age",x="Parch", data=train,kind="box")

g = sns.factorplot(y="Age",x="SibSp", data=train,kind="box")
## Fill Age with the median age of similar rows according to Pclass, Parch and SibSp

# Index of NaN age rows

index_NaN_age = list(train["Age"][train["Age"].isnull()].index)



for i in index_NaN_age :

    age_med = train["Age"].median()

    age_pred = train["Age"][((train['SibSp'] == train.iloc[i]["SibSp"]) & (train['Parch'] == train.iloc[i]["Parch"]) & (train['Pclass'] == train.iloc[i]["Pclass"]))].median()

    if not np.isnan(age_pred) :

        train['Age'].iloc[i] = age_pred

    else :

        train['Age'].iloc[i] = age_med
index_NaN_age = list(test["Age"][test["Age"].isnull()].index)



for i in index_NaN_age :

    age_med = test["Age"].median()

    age_pred = test["Age"][((test['SibSp'] == test.iloc[i]["SibSp"]) & (test['Parch'] == test.iloc[i]["Parch"]) & (test['Pclass'] == test.iloc[i]["Pclass"]))].median()

    if not np.isnan(age_pred) :

        test['Age'].iloc[i] = age_pred

    else :

        test['Age'].iloc[i] = age_med
train.isnull().sum()


train["Embarked"] = train["Embarked"].fillna("S")

test["Embarked"] = test["Embarked"].fillna("S")
train.isnull().sum()
test.isnull().sum()
test[test["Fare"].isnull()]
test[["Pclass","Fare"]].groupby("Pclass").mean()
test["Fare"] = test["Fare"].fillna(12)
test["Fare"].isnull().sum()
train["Yeni_cabin"] = (train["Cabin"].notnull().astype('int'))

test["Yeni_Cabin"] = (test["Cabin"].notnull().astype('int'))

print("Percentage of Yeni_cabin = 1 who survived:", train["Survived"][train["Yeni_cabin"] == 1].value_counts(normalize = True)[1]*100)



print("Percentage of Yeni_cabin = 0 who survived:", train["Survived"][train["Yeni_cabin"] == 0].value_counts(normalize = True)[1]*100)



sns.barplot(x="Yeni_cabin", y="Survived", data=train)



train = train.drop(['Cabin'], axis = 1)

test = test.drop(['Cabin'], axis = 1)

train.head()
train.isnull().sum()
test.isnull().sum()
train = train.drop(['Ticket'], axis = 1)

test = test.drop(['Ticket'], axis = 1)
train.head()
# convert Sex into categorical value 0 for male and 1 for female

train["Sex"] = train["Sex"].map({"male": 0, "female":1})

test["Sex"] = test["Sex"].map({"male": 0, "female":1})

train.head()
train["unvan"] = train["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)

test["unvan"] = test["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)
train['unvan'] = train['unvan'].replace(['Lady', 'Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')

train['unvan'] = train['unvan'].replace(['Countess', 'Lady', 'Sir'], 'Royal')

train['unvan'] = train['unvan'].replace('Mlle', 'Miss')

train['unvan'] = train['unvan'].replace('Ms', 'Miss')

train['unvan'] = train['unvan'].replace('Mme', 'Mrs')
test['unvan'] = test['unvan'].replace(['Lady', 'Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')

test['unvan'] = test['unvan'].replace(['Countess', 'Lady', 'Sir'], 'Royal')

test['unvan'] = test['unvan'].replace('Mlle', 'Miss')

test['unvan'] = test['unvan'].replace('Ms', 'Miss')

test['unvan'] = test['unvan'].replace('Mme', 'Mrs')
train[['unvan', 'Survived']].groupby(['unvan'], as_index=False).mean()
# Map each of the unvan groups to a numerical value



unvan_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 5}



train['unvan'] = train['unvan'].map(unvan_mapping)
test['unvan'] = test['unvan'].map(unvan_mapping)
train.head()
test.head()
train = train.drop(['Name'], axis = 1)

test = test.drop(['Name'], axis = 1)
train.head()
# Map Fare values into groups of numerical values:

train['FareBand'] = pd.qcut(train['Fare'], 4, labels = [1, 2, 3, 4])

test['FareBand'] = pd.qcut(test['Fare'], 4, labels = [1, 2, 3, 4])
# Drop Fare values:

train = train.drop(['Fare'], axis = 1)

test = test.drop(['Fare'], axis = 1)
train.head()
bins = [0, 5, 12, 18, 24, 35, 60, np.inf]

mylabels = ['Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

train['AgeGroup'] = pd.cut(train["Age"], bins, labels = mylabels)

test['AgeGroup'] = pd.cut(test["Age"], bins, labels = mylabels)
# Map each Age value to a numerical value:

age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}

train['AgeGroup'] = train['AgeGroup'].map(age_mapping)

test['AgeGroup'] = test['AgeGroup'].map(age_mapping)
#dropping the Age feature for now, might change:

train = train.drop(['Age'], axis = 1)

test = test.drop(['Age'], axis = 1)
train.head()
# 5.Feature Engineering
### Embarked & Title

# Convert Title and Embarked into dummy variables:



train = pd.get_dummies(train, columns = ["unvan"])

train = pd.get_dummies(train, columns = ["Embarked"], prefix="Em")

train.head()
test = pd.get_dummies(test, columns = ["unvan"])

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
x_train.shape
x_test.shape
from sklearn.ensemble import GradientBoostingClassifier



gbk = GradientBoostingClassifier()

gbk.fit(x_train, y_train)

y_pred = gbk.predict(x_test)

acc_gbk = round(accuracy_score(y_pred, y_test) * 100, 2)

print(acc_gbk)
xgb_params = {

        'n_estimators': [200, 500],

        'subsample': [0.6, 1.0],

        'max_depth': [2,5,8],

        'learning_rate': [0.1,0.01,0.02],

        "min_samples_split": [2,5,10]}
xgb = GradientBoostingClassifier()



xgb_cv_model = GridSearchCV(xgb, xgb_params, cv = 10, n_jobs = -1, verbose = 2)
xgb_cv_model.fit(x_train, y_train)
xgb_cv_model.best_params_
xgb = GradientBoostingClassifier(learning_rate = xgb_cv_model.best_params_["learning_rate"], 

                    max_depth = xgb_cv_model.best_params_["max_depth"],

                    min_samples_split = xgb_cv_model.best_params_["min_samples_split"],

                    n_estimators = xgb_cv_model.best_params_["n_estimators"],

                    subsample = xgb_cv_model.best_params_["subsample"])
xgb_tuned =  xgb.fit(x_train,y_train)
y_pred = xgb_tuned.predict(x_test)

acc_gbk = round(accuracy_score(y_pred, y_test) * 100, 2)

print(acc_gbk)
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
!pip install lightgbm
from lightgbm import LGBMRegressor
lgb_model = LGBMRegressor().fit(x_train, y_train)
lgb_model
y_pred = lgb_model.predict(x_test)
lgbm_params = {"learning_rate": [0.01, 0.1, 0.5, 1],

              "n_estimators": [20,40,100,200,500,1000],

              "max_depth": [1,2,3,4,5,6,7,8,9,10]}
lgbm_cv_model = GridSearchCV(lgb_model, 

                             lgbm_params, 

                             cv = 10, 

                             n_jobs = -1, 

                             verbose =2).fit(x_train, y_train)
lgbm_cv_model.best_params_
lgbm_tuned = LGBMRegressor(learning_rate = 0.1, 

                          max_depth = 6, 

                          n_estimators = 20).fit(x_train, y_train)
y_pred = lgbm_tuned.predict(x_test)
np.sqrt(mean_squared_error(y_test, y_pred))
test
#set ids as PassengerId and predict survival 

ids = test['PassengerId']

predictions = xgb_tuned.predict(test.drop('PassengerId', axis=1))



#set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.to_csv('submission.csv', index=False)
output.head()