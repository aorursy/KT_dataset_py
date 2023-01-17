# Importing useful libraries



# Import data sets, Preprocessing and Feature Engineering

import pandas as pd

# Linear Algebra operations

import numpy as np



# Exploratory Data Analysis

import missingno as msno

import matplotlib.pyplot as plt

import seaborn as sns

import plotly



# Models and Model evaluation

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.metrics import make_scorer

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression



# To measure execution time

import time
# Load train and test datasets

train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")
train.head()
test.head()
train.describe()
test.describe()
# Show class imbalance

train.Survived.value_counts()
msno.matrix(train,figsize=(10,10))

train.isnull().sum()/train.shape[0]
msno.matrix(test,figsize=(10,10))

test.isnull().sum()/test.shape[0]
# Check if Sex and passenger class matter to survive rate

g = sns.catplot(x="Sex", 

                hue="Pclass", 

                col="Survived",

                data=train, kind="count",

                height=4, aspect=.7)
# concat train and test sets

train['train'] = 1

test['train'] = 0

data = train.append(test, ignore_index=True)
# Fill columns with few missing values

data.Fare = data.Fare.fillna(0)



# Fill Embarked with the most common value

data.Embarked = data.Embarked.fillna(data.Embarked.mode()[0])
# Title Mapping

data["Title"] = data["Name"].str.extract(' ([A-Za-z]+)\.',expand=False)

data.Title.unique()



# Create features with family size and mother status

data['FamilySize'] = data.SibSp + data.Parch + 1

data['Mother'] = np.where((data.Title=='Mrs') & (data.Parch >0),1,0)





# map gender and boys by title as Master represents mostly boys

titles = {

        "Mr" :         "man",

        "Mme":         "woman",

        "Ms":          "woman",

        "Mrs" :        "woman",

        "Master" :     "boy",

        "Mlle":        "woman",

        "Miss" :       "woman",

        "Capt":        "man",

        "Col":         "man",

        "Major":       "man",

        "Dr":          "man",

        "Rev":         "man",

        "Jonkheer":    "man",

        "Don":         "man",

        "Sir" :        "man",

        "Countess":    "woman",

        "Dona":        "woman",

        "Lady" :       "woman"

    } 



data["Gender"] = data["Title"].map(titles)
# Fill missing age values with median by title

data["Age"].fillna(data.groupby("Title")["Age"].transform("median"), inplace=True)



# divide age column into a range of values

cut_points = [-1,0,5,12,18,35,60,100]

label_names = ["Missing","Infant","Child","Teenager","Young Adult","Adult","Senior"]

data["Age_categories"] = pd.cut(data["Age"], cut_points, labels=label_names)
data.head()
# Creating dummy variables for categories

dummable = ['Mother', 'Gender', 'Age_categories', 'FamilySize', 'Sex', 'Pclass', 'Embarked']



for col in dummable:

  dummies = pd.get_dummies(data[col],prefix=col, drop_first=False)

  data = pd.concat([data, dummies], axis=1)

  data = data.drop(col, 1)
# generate test and train datasets with important variables

train_df = data[data['train'] == 1].drop(['Cabin', 'Name', 'PassengerId', 'Ticket', 'Title', 'train', 'SibSp', 'Parch'], 1)

test_df = data[data['train'] == 0].drop(['Cabin', 'Name', 'PassengerId', 'Ticket', 'Title', 'train', 'Survived', 'SibSp', 'Parch'], 1)
train_df.shape
# Global Variables

seed = 42

num_folds = 10

scoring = {'Accuracy': make_scorer(accuracy_score)}



# train and test

X = train_df.drop("Survived", axis=1)

y = train_df["Survived"]

X_test = test_df



X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.1, random_state = seed)
pipe = Pipeline(steps = [("clf",XGBClassifier())])

search_space = [

                {"clf":[RandomForestClassifier()],

                 "clf__n_estimators": [100],

                 "clf__criterion": ["entropy"],

                 "clf__max_leaf_nodes": [64],

                 "clf__random_state": [seed]

                 },

                {"clf":[LogisticRegression()],

                 "clf__solver": ["liblinear"]

                 },

                {"clf":[XGBClassifier()],

                 "clf__n_estimators": [50,100],

                 "clf__max_depth": [4],

                 "clf__learning_rate": [0.001, 0.01,0.1],

                 "clf__random_state": [seed],

                 "clf__subsample": [1.0],

                 "clf__colsample_bytree": [1.0],

                 

                 }

                ]



# create grid search

kfold = StratifiedKFold(n_splits=num_folds,random_state=seed)



# return_train_score=True

# official documentation: "computing the scores on the training set can be

# computationally expensive and is not strictly required to

# select the parameters that yield the best generalization performance".

grid = GridSearchCV(estimator=pipe, 

                    param_grid=search_space,

                    cv=kfold,

                    scoring=scoring,

                    return_train_score=True,

                    n_jobs=-1,

                    refit="Accuracy")



tmp = time.time()



# fit grid search

best_model = grid.fit(X_train,y_train)



print("CPU Training Time: %s seconds" % (str(time.time() - tmp)))
print("Best: %f using %s" % (best_model.best_score_,best_model.best_params_))
result = pd.DataFrame(best_model.cv_results_)

result.head()
predict_first = best_model.best_estimator_.predict(X_valid)

print(accuracy_score(y_valid, predict_first))
predict_final = best_model.best_estimator_.predict(X_test)
submission = test[['PassengerId']].copy()

submission['Survived'] = np.rint(predict_final).astype(int)

print(submission)

submission.to_csv('submission.csv', index=False)
