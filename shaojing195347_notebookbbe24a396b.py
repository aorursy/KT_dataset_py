import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sys

import scipy

import matplotlib

import matplotlib.pyplot as plt

myfont = matplotlib.font_manager.FontProperties(fname=r'C:/Windows/Fonts/msyh.ttf')

import sklearn

import xgboost as xgb

from pandas.plotting import scatter_matrix

import matplotlib.pyplot as plt

from sklearn import model_selection

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

import warnings

warnings.filterwarnings("ignore")

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data_train = pd.read_csv("/kaggle/input/titanic/train.csv")

data_test = pd.read_csv("/kaggle/input/titanic/test.csv")
def set_Cabin_type(df):

    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"

    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"

    return df



data_train = set_Cabin_type(data_train)



dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix= 'Cabin')



dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix= 'Embarked')



dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')



dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')



data_train = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)



X = data_train.drop(['Survived','Name','Ticket','Cabin','Embarked','Sex','Pclass'],1)

Y = data_train.Survived

validation_size = 0.20

seed = 7



X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
xgboost = xgb.XGBClassifier(max_depth=5,learning_rate= 0.1, verbosity=1, objective='binary:logistic',n_estimators=50)

xgboost.fit(X_train, Y_train)
data_test = set_Cabin_type(data_test)



dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix= 'Cabin')



dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')



dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')



dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix= 'Pclass')



data_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)



X_test = data_test.drop(['Name','Ticket','Cabin','Embarked','Sex','Pclass'],1)
predictions = xgboost.predict(X_test)

result = pd.DataFrame({'PassengerId':data_test['PassengerId'], 'Survived':predictions.astype(np.int32)})

result.to_csv("titanic_xgb_predictions.csv", index=False)