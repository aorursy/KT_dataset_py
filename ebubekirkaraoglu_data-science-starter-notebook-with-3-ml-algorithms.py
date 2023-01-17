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

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import LinearSVC

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from catboost import CatBoostClassifier
# Read train and test data with pd.read_csv():

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
# copy data in order to avoid any change in the original:

train = train_data.copy()

test = test_data.copy()
train.head(30)
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

x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.22, random_state = 0)
x_train.shape
x_test.shape
def base_models_1(train):

    

       

    from sklearn.model_selection import train_test_split

    from sklearn.metrics import accuracy_score

    

    predictors = train.drop(['Survived', 'PassengerId'], axis=1)

    target = train["Survived"]

    

    x_train, x_test, y_train, y_test = train_test_split(predictors, target, 

                                                    test_size = 0.22, 

                                                    random_state = 42)

    

    #results = []

    

    names = ["LogisticRegression","GaussianNB","KNN","LinearSVC","SVC",

             "CART","RF","GBM"]

    

    

    classifiers = [LogisticRegression(),GaussianNB(), KNeighborsClassifier(),LinearSVC(),SVC(),

                  DecisionTreeClassifier(),RandomForestClassifier(), GradientBoostingClassifier()]

    

    

    for name, clf in zip(names, classifiers):



        model = clf.fit(x_train, y_train)

        y_pred = model.predict(x_test)

        acc = accuracy_score(y_test, y_pred)

        msg = "%s: %f" % (name, acc)

        print(msg)
base_models_1(train)
def base_models(train):

    

       

    from sklearn.model_selection import cross_val_score, KFold 

    predictors = train.drop(['Survived', 'PassengerId'], axis=1)

    target = train["Survived"]

    

        

    results = []

    

    names = ["LogisticRegression","GaussianNB","KNN","LinearSVC","SVC",

             "CART","RF","GBM","XGBoost","LightGBM","CatBoost"]

    

    

    classifiers = [LogisticRegression(),GaussianNB(), KNeighborsClassifier(),LinearSVC(),SVC(),

                  DecisionTreeClassifier(),RandomForestClassifier(), GradientBoostingClassifier(),

                  XGBClassifier(), LGBMClassifier(), CatBoostClassifier(verbose = False)]

    

    

    for name, clf in zip(names, classifiers):

        

        kfold = KFold(n_splits=10, random_state=1001)

        cv_results = cross_val_score(clf, predictors, target, cv = kfold, scoring = "accuracy")

        results.append(cv_results)

        msg = "%s: %f (%f)" % (name, (cv_results.mean())*100, cv_results.std())

        

        print(msg)
base_models(train)
gb_params = {

        'n_estimators': [200, 500],

        'subsample': [0.6, 1.0],

        'max_depth': [2,5,8],

        'learning_rate': [0.1,0.01,0.02],

        "min_samples_split": [2,5,10]}
gb = GradientBoostingClassifier()



gb_cv_model = GridSearchCV(gb, gb_params, cv = 10, n_jobs = -1, verbose = 2)
gb_cv_model.fit(x_train, y_train)
gb = GradientBoostingClassifier(learning_rate = gb_cv_model.best_params_["learning_rate"], 

                    max_depth = gb_cv_model.best_params_["max_depth"],

                    min_samples_split = gb_cv_model.best_params_["min_samples_split"],

                    n_estimators = gb_cv_model.best_params_["n_estimators"],

                    subsample = gb_cv_model.best_params_["subsample"])
gb_tuned =   gb.fit(x_train,y_train)
y_pred = gb_tuned.predict(x_test)

acc_gbk = round(accuracy_score(y_pred, y_test) * 100, 2)

print(acc_gbk)

test