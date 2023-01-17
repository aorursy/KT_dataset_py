import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from nltk.classify import SklearnClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

from sklearn.svm import SVR,NuSVR,LinearSVR,SVC #support vector regression

from sklearn.linear_model import LinearRegression,Ridge,Lasso#Ridge() and Lasso()

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import StratifiedKFold

test = pd.read_csv("../input/restaurant-revenue-prediction/test.csv")

train = pd.read_csv("../input/restaurant-revenue-prediction/train.csv")

train.head()
print(train.info())
train['P29'] = train['P29'].astype(int)

test['P29']    = test['P29'].astype(int)

test["P29"].fillna(test["P29"].median(), inplace=True)

train['P26'] = train['P26'].astype(int)

test['P26']    = test['P26'].astype(int)

train['P27'] = train['P27'].astype(int)

test['P27']    = test['P27'].astype(int)

train['P28'] = train['P28'].astype(int)

test['P28']    = test['P28'].astype(int)

train['P13'] = train['P13'].astype(int)

test['P13']    = test['P13'].astype(int)

train['P2'] = train['P2'].astype(int)

test['P2']    = test['P2'].astype(int)

train['P3'] = train['P3'].astype(int)

test['P3']    = test['P3'].astype(int)

train['P4'] = train['P4'].astype(int)

test['P4']    = test['P4'].astype(int)
train.describe()
train.describe(include=['object'])
train['City'].value_counts()
train['City Group'].value_counts()
train['Type'].value_counts()
corrmat = train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);
train["City Group"] = train["City Group"].map({"Big Cities": 0, "Other":1})

test["City Group"] = test["City Group"].map({"Big Cities": 0, "Other":1})

train["Type"] = train["Type"].map({"FC": 0, "IL":1,"DT":2})

test["Type"] = test["Type"].map({"FC": 0, "IL":1,"DT":2})

# Is city important or not

#How can we get groups of revenue and plot it against city groups and types to compare
test["Type"].fillna(test["Type"].median(), inplace=True)

train["revenue"].fillna(train["revenue"].median(), inplace=True)

train['revenue'] = train['revenue'].astype(int)

import numpy

Y_train=train["revenue"].apply(numpy.log)

X_train = train.drop(['City','Open Date','revenue','Id','City Group'], axis=1)

#X_test  = test.drop("Id",axis=1).copy()

X_test    = test.drop(['City','Open Date','Id','City Group'], axis=1)

X_train.head()

#X_test.head()
#from sklearn.impute import SimpleImputer

#my_imputer = SimpleImputer()

#imputed_X_train = my_imputer.fit_transform(X_train)

#imputed_X_test = my_imputer.transform(X_test)
#cols_with_missing = [col for col in X_train.columns 

#                                 if X_train[col].isnull().any()]

#X_train = X_train.drop(cols_with_missing, axis=1)

#X_test  = X_test.drop(cols_with_missing, axis=1)
test.head()

test=test.drop(['City','Open Date','City Group'], axis=1)
from sklearn import linear_model

import numpy 

from sklearn.ensemble import RandomForestRegressor

cls = RandomForestRegressor(n_estimators=100)

cls.fit(X_train, Y_train)

pred = cls.predict(X_test)

pred = numpy.exp(pred)

cls.score(X_train, Y_train)
output = pd.DataFrame({'Id': test['Id'],

                     'Prediction': pred})

output.to_csv('submission.csv', index=False)