# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBRegressor

from sklearn.ensemble import AdaBoostClassifier

X = pd.read_csv('../input/titanic/train.csv')

X_test = pd.read_csv('../input/titanic/test.csv')

gen = pd.read_csv('../input/titanic/gender_submission.csv')

data = [X, X_test]
X.head()
X.Survived.isnull().sum()
print(X.columns)

print(X.Cabin.isnull().sum())
X.drop(['PassengerId','Name','Ticket','Cabin'], axis = 1, inplace = True)

y = X['Survived']

X.drop('Survived',axis = 1, inplace = True)



X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,random_state=0)
# Number of missing values in each column of training data

missing_val_count_by_column = (X_train.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])
# Remove rows with missing Embarked value

#X_train.dropna(axis=0, subset=['Embarked'], inplace=True)
# Preprocessing for numerical data

numerical_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']]

numerical_transformer = SimpleImputer(strategy='constant')



# Preprocessing for categorical data

categorical_cols = [cname for cname in X_train.columns if X_train[cname].nunique() < 10 and X_train[cname].dtype == "object"]

categorical_transformer = Pipeline(steps=[ ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))])



# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer( transformers=[('num', numerical_transformer, numerical_cols), ('cat', categorical_transformer, categorical_cols)])

y_train.shape
def Predition(model):

    my_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

    my_pipeline.fit(X_train,y_train)

    pred = my_pipeline.predict(X_valid)

    score = accuracy_score(pred,y_valid)

    print("Accuracy of ", score)

    return (score)
def TestModel(model):

    my_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

    my_pipeline.fit(X_train,y_train)

    pred = my_pipeline.predict(X_test.drop(['PassengerId','Name','Ticket','Cabin'], axis = 1))

    score = accuracy_score(pred,gen['Survived'])

    print("Accuracy of ", score)

    return (pred)
Predition(LogisticRegression(penalty = 'l2',C=10,tol=0.0001))
Predition(LinearSVC(penalty = 'l2',loss = 'squared_hinge', C=1,max_iter=1000))
Predition(SVC(C=5,kernel = 'sigmoid',  max_iter=-1))
Predition(SGDClassifier(penalty='l2',loss='hinge',alpha = 0.0001,  max_iter=5000))
Predition(KNeighborsClassifier(n_neighbors = 13,leaf_size = 30))
Predition(RandomForestClassifier(n_estimators=10))
Predition(GaussianNB())
Predition(DecisionTreeClassifier())
Predition(Perceptron(penalty='l2', alpha=0.0001))
Predition(AdaBoostClassifier(base_estimator=None, n_estimators=50))
test_predictions = TestModel(AdaBoostClassifier(base_estimator=None, n_estimators=50))
submission = pd.DataFrame({

        "PassengerId": X_test.PassengerId,

        "Survived": test_predictions

    })

submission.to_csv("titanic_submission.csv", index=False)