# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV
train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")
train.head()
train.info()
train.describe().transpose()
train.isnull().sum()
print (train.groupby("Sex").Survived.mean())

sns.countplot(x = "Sex", hue= "Survived", data= train)
sns.distplot(train[train.Survived == 0]["Age"], kde = False, bins = 40)

sns.distplot(train[train.Survived == 1]["Age"], kde = False, bins = 40, color = "darkred")
print (train.groupby("Pclass").Survived.mean())

sns.countplot(x = "Pclass", hue = "Survived", data= train)
print (train.groupby("Embarked").Survived.mean())

sns.countplot(x= "Embarked", data = train, hue= "Survived")
print (train.groupby("Parch").Survived.mean())

sns.countplot(x= "Parch", hue = "Survived", data = train)
print (train.groupby("SibSp").Survived.mean())

sns.countplot(x = "SibSp", hue = "Survived", data = train)
sns.boxplot(x= "Pclass", y = "Age", hue = "Sex", data= train)
train.groupby(["Pclass", "Sex"]).Age.median()
#Imputing Age on the basis of Pclass and Sex

train.loc[:, "Age"] = train.groupby(["Pclass", "Sex"]).Age.apply(lambda x: x.fillna(x.median()))

test.loc[:, "Age"] = test.groupby(["Pclass", "Sex"]).Age.apply(lambda x : x.fillna(x.median()))
train.drop(["Cabin", "Name", "Ticket", "Fare"], axis = "columns", inplace = True)

test.drop(["Cabin", "Name", "Ticket", "Fare"], axis = "columns", inplace = True)
train.head()
X = train.drop(["PassengerId", "Survived"], axis = "columns")

y = train.Survived
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state = 1)
X_train.columns.to_list()
#Splitting the columns into continuous and categorical



numerical_cols = ["Age", "Parch", "SibSp"]

categorical_cols = ["Pclass", "Sex", "Embarked"]
numerical_transform = StandardScaler()

categorical_tranform = Pipeline(steps = [

    ("imputer", SimpleImputer(strategy = "most_frequent")),

    ("onehot", OneHotEncoder(handle_unknown = "ignore"))

])

preprocessor = ColumnTransformer(transformers=[

    ("num", numerical_transform, numerical_cols),

    ("cat", categorical_tranform, categorical_cols)

])
classifier = Pipeline(steps= [

    ("preprocessor", preprocessor),

    ("model", LogisticRegression(max_iter = 10000))

])



param_grid = {

    'model__C': [0.001, 0.01, 0.1,1, 10, 100, 1000]

}



clf_LR = GridSearchCV(classifier, param_grid, cv = 5, scoring= "accuracy")

clf_LR.fit(X_train, y_train)
print (clf_LR.best_params_)

print (clf_LR.best_score_)
classifier = Pipeline(steps= [

    ("preprocessor", preprocessor),

    ("model", SVC())

])



param_grid = {

    'model__C': [0.01, 0.1,1, 10, 100],

    "model__gamma": [0.001, 0.01, 0.1, 1]

}



clf_SVC = GridSearchCV(classifier, param_grid, cv = 5, scoring= "accuracy")

clf_SVC.fit(X_train, y_train)
print (clf_SVC.best_params_)

print (clf_SVC.best_score_)
models = [LogisticRegression(max_iter= 10000, C = 0.1), SVC(C = 10, gamma= 0.1), RandomForestClassifier(n_estimators= 50)]

scores = []

for model in models:

    classifier = Pipeline(steps= [

    ("preprocessor", preprocessor),

    ("model", model)

    ])

    cross = cross_val_score(classifier, X_train, y_train)

    scores.append(np.average(cross))

    

scores
classifier_final = Pipeline(steps= [

    ("preprocessor", preprocessor),

    ("model", SVC(C = 10, gamma = 0.1))

])
classifier_final.fit(X_train, y_train)
pred = classifier_final.predict(X_valid)

accuracy_score(y_valid, pred)
PassengerId = test.PassengerId
test.drop("PassengerId", axis = "columns", inplace = True)
predictions = classifier_final.predict(test)
output = pd.DataFrame({"PassengerId": PassengerId,

                      "Survived": predictions})

output.to_csv("submission.csv", index = False)