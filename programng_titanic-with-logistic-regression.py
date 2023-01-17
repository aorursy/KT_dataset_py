# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import model_selection

from sklearn.linear_model import LogisticRegression



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df_titanic = pd.read_csv("../input/train.csv")

print(df_titanic.head(5))

print(df_titanic.describe())

print(df_titanic.info())
df_titanic["Age"] = df_titanic["Age"].fillna(df_titanic["Age"].median())
print(df_titanic["Sex"].unique())
df_titanic.loc[df_titanic["Sex"] == "male", "Sex"] = 0

df_titanic.loc[df_titanic["Sex"] == "female", "Sex"] = 1
print(df_titanic["Embarked"].unique())
df_titanic["Embarked"] = df_titanic["Embarked"].fillna("S")

df_titanic.loc[df_titanic["Embarked"] == "S", "Embarked"] = 0

df_titanic.loc[df_titanic["Embarked"] == "C", "Embarked"] = 1

df_titanic.loc[df_titanic["Embarked"] == "Q", "Embarked"] = 2
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]



log_regr = LogisticRegression()

scores = model_selection.cross_val_score(log_regr, df_titanic[features], df_titanic["Survived"], cv=5)



print(scores.mean())
df_titanic_test = pd.read_csv("../input/test.csv")

df_titanic_test["Age"] = df_titanic_test["Age"].fillna(df_titanic["Age"].median())

df_titanic_test.loc[df_titanic_test["Sex"] == "male", "Sex"] = 0

df_titanic_test.loc[df_titanic_test["Sex"] == "female", "Sex"] = 1

df_titanic_test.loc[df_titanic_test["Embarked"] == "S", "Embarked"] = 0

df_titanic_test.loc[df_titanic_test["Embarked"] == "C", "Embarked"] = 1

df_titanic_test.loc[df_titanic_test["Embarked"] == "Q", "Embarked"] = 2

df_titanic_test["Fare"] = df_titanic_test["Fare"].fillna(df_titanic_test["Fare"].median())
log_regr_test = LogisticRegression()



log_regr_test.fit(df_titanic[features], df_titanic["Survived"])



predictions = log_regr_test.predict(df_titanic_test[features])



submission = pd.DataFrame({

    "PassengerId": df_titanic_test["PassengerId"],

    "Survived": predictions

})
submission.to_csv("titanic_predictions_logistic_regression.csv", index=False)