import numpy as np
import pandas as pd
train_df = pd.read_csv("../input/train.csv")
train_df.head()
train_df.describe()
train_df.describe(include=["object"])
train_df.dtypes
train_df.shape
clone_df = train_df.copy()
clone_df["Pclass_1"] = (clone_df["Pclass"] == 1) * 1
clone_df["Pclass_2"] = (clone_df["Pclass"] == 2) * 1
clone_df["Pclass_3"] = (clone_df["Pclass"] == 3) * 1
clone_df.drop("Pclass", axis=1, inplace=True)
clone_df.head()
clone_df["Sex_Male"] = (clone_df["Sex"] == "male") * 1
clone_df["Sex_Female"] = (clone_df["Sex"] == "female") * 1
clone_df.drop("Sex", axis=1, inplace=True)
clone_df.head()
clone_df["Embarked_C"] = (clone_df["Embarked"] == "C") * 1
clone_df["Embarked_S"] = (clone_df["Embarked"] == "S") * 1
clone_df["Embarked_Q"] = (clone_df["Embarked"] == "Q") * 1
clone_df.drop("Embarked", axis=1, inplace=True)
clone_df.head()
clone_df.drop(["Name", "Ticket", "Cabin"], axis=1, inplace=True)
clone_df.head()
clone_df.corr()
clone_df[clone_df["Pclass_1"] == 1].groupby(["Pclass_1", "Survived"])["Survived"].count()
clone_df[clone_df["Pclass_2"] == 1].groupby(["Pclass_2", "Survived"])["Survived"].count()
clone_df[clone_df["Pclass_3"] == 1].groupby(["Pclass_3", "Survived"])["Survived"].count()
clone_df[clone_df["Sex_Male"] == 1].groupby(["Sex_Male", "Survived"])["Survived"].count()
clone_df[clone_df["Sex_Female"] == 1].groupby(["Sex_Female", "Survived"])["Survived"].count()
clone_df[clone_df["Embarked_C"] == 1].groupby(["Embarked_C", "Survived"])["Survived"].count()
clone_df[clone_df["Embarked_C"] == 1].groupby(["Embarked_C", "Survived"])["Survived"].count()
clone_df[clone_df["Embarked_S"] == 1].groupby(["Embarked_S", "Survived"])["Survived"].count()
clone_df[clone_df["Embarked_Q"] == 1].groupby(["Embarked_Q", "Survived"])["Survived"].count()
feature_cols = ["Pclass_1", "Sex_Female", "Embarked_C"]
X_train = clone_df.loc[:, feature_cols]
X_train.shape
y_train = train_df.Survived
y_train.shape
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
test_df = pd.read_csv("../input/test.csv")
clone_test_df = pd.DataFrame({})
clone_test_df["Pclass_1"] = (test_df["Pclass"] == 1) * 1
clone_test_df["Sex_Female"] = (test_df["Sex"] == "female") * 1
clone_test_df["Embarked_C"] = (test_df["Embarked"] == "C") * 1
clone_test_df["Fare_>=_50"] = (test_df["Fare"] >= 50) * 1
X_test = clone_test_df.loc[:, feature_cols]
predictions = logreg.predict(X_test)
import time

kaggle_data = pd.DataFrame({"PassengerId": test_df["PassengerId"], "Survived": predictions}).set_index('PassengerId')
kaggle_data.to_csv('submission_' + str(time.time()) + '.csv')