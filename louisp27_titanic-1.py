# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import scipy as sp
import pandas as pd 
import os
print(os.listdir("../input"))
titanic = pd.read_csv("../input/train.csv")
# Any results you write to the current directory are saved as output

from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
ohe = OneHotEncoder()


#This separates the cabin group into just the first letter, rather than the exact room number
titanic["CabinGroup"] = titanic["Cabin"].fillna("").str[:1]
#y values aka outputs: if they survived.
y_train = titanic["Survived"].values
#x values: important values that can correlate to survival.
X_numeric = titanic[["Age", "Fare"]].fillna(0).values
#adding class to the OHE.
X_categorical = ohe.fit_transform(titanic[["Sex","CabinGroup","Pclass"]].fillna("").values)
X_train = sp.hstack([X_numeric,X_categorical.todense()])

print("Training Data")
model = LogisticRegression()
model.fit(X_train, y_train)
model.score(X_train, y_train)
#with training examples it records 80.2%


#titanicTest = pd.read_csv("../input/test.csv")

#y_test = titanicTest["Survived"].values
#X_numericTest = titanicTest[["Age", "Fare"]].fillna(0).values
#X_categoricalTest = ohe.fit_transform(titanicTest[["Sex"]].fillna("").values)
#X_trainTest = sp.hstack([X_numericTest,X_categoricalTest.todense()])
#model = LogisticRegression()
#model.fit(X_trainTest, y_trainTest)
#model.score(X_trainTest, y_trainTest)
from sklearn import compose, impute, pipeline, preprocessing
from sklearn import linear_model
from sklearn import neighbors
knn_model = neighbors.KNeighborsClassifier(7)
knn_model.fit(titanic[["Age","Fare",]].fillna(0), titanic["Survived"])

model = linear_model.LogisticRegression(solver="lbfgs")
model.fit(titanic[["Age","Fare"]].fillna(0), titanic["Survived"])

numeric_features = ["Age", "Fare"]
categorical_features = ["Pclass", "Sex", "Cabin"]

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

#   TESTING  ######

t_test = pd.read_csv("../input/test.csv")

XTEST = preprocessor.transform(t_test)

y_pred = knn_model.predict(XTEST)

knn_model.score(XTEST, y_pred)

submission = pd.DataFrame({"PassengerId": t_test["PassengerId"],
    "Survived": y_pred})
submission.to_csv("submission.csv", index=False)


