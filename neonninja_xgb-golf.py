import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score

from xgboost import XGBClassifier
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

train["Sex"] = np.where(train["Sex"] == "female", 1, 0)

train.fillna(0, inplace=True)

test["Sex"] = np.where(test["Sex"] == "female", 1, 0)

test.fillna(0, inplace=True)
cols = ["Pclass", "Sex"]

x = train[cols]

y = train["Survived"]
np.random.seed(1337)

model = XGBClassifier()

model.fit(x, y)

preds = model.predict(test[cols])

test["Survived"] = preds
test[["PassengerId", "Survived"]].to_csv("submission.csv", index=False)