# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import scipy as sp

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
titanic_test = pd.read_csv("../input/test.csv")

titanic_test

from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression

ohe = OneHotEncoder()

titanic = pd.read_csv("../input/train.csv")

titanic["CabinGroup"] = titanic["Cabin"].fillna("").str[:1]

y_test = titanic["Survived"].values

X_numeric = titanic[["Age", "Fare"]].fillna(0).values

X_categorical = ohe.fit_transform(titanic[["Pclass", "Sex", "Embarked", "CabinGroup"]].fillna("").values)

X_train = sp.hstack([X_numeric,X_categorical.todense()])

model = LogisticRegression()

model.fit(X_train, y_train)

model.score(X_train, y_train)

%pylab inline

scatter(titanic["Age"], titanic["Fare"], c=titanic["Survived"])

xlabel("Age")

ylabel("Fare")
from sklearn import linear_model

model = linear_model.LogisticRegression(solver="lbfgs")

model.fit(titanic[["Age","Fare"]].fillna(0), titanic["Survived"])
scatter(titanic["Age"], titanic["Fare"], c=titanic["Survived"])

xlabel("Age")

ylabel("Fare")

x_min, x_max = 0, 80

y_min, y_max = 0, 500

xx, yy = np.meshgrid(np.arange(x_min, x_max, 5),

                     np.arange(y_min, y_max, 50))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)

contour(xx, yy, Z, colors="teal", levels=[0.5], linewidths=5)
from sklearn import neighbors

knn_model = neighbors.KNeighborsClassifier(5)

knn_model.fit(titanic[["Age","Fare"]].fillna(0), titanic["Survived"])
{"logreg": model.score(titanic[["Age","Fare"]].fillna(0), titanic["Survived"]), 

 "knn": knn_model.score(titanic[["Age","Fare"]].fillna(0), titanic["Survived"])}
from sklearn import compose, impute, pipeline, preprocessing



numeric_features = ["Age", "Fare"]

categorical_features = ["Pclass", "Sex"]



numeric_transformer = pipeline.make_pipeline(

  impute.SimpleImputer(strategy="median"),

  preprocessing.StandardScaler())

categorical_transformer = pipeline.make_pipeline(

  impute.SimpleImputer(strategy="constant", fill_value="NA"),

  preprocessing.OneHotEncoder(handle_unknown="ignore"))



preprocessor = compose.make_column_transformer(

  (numeric_transformer, numeric_features),

  (categorical_transformer, categorical_features))
preprocessor.fit(titanic)

X = preprocessor.transform(titanic)

y = titanic["Survived"]



model.fit(X, y)

knn_model.fit(X, y)



{"logreg": model.score(X, y), "knn": knn_model.score(X, y)}
X_test = preprocessor.transform(titanic_test)



y_pred = knn_model.predict(X_test)



submission = pd.DataFrame({

    "PassengerId": titanic_test["PassengerId"],

    "Survived": y_pred

})

submission.to_csv("submission.csv", index=False)
