import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score

from xgboost import XGBClassifier
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
magic = "wheezy-copper-turtle-magic"

magicks = sorted(train[magic].unique())

print(magicks)

data_cols = [c for c in train.columns if c not in ['id', 'target', magic]]


model = XGBClassifier()

results = []

for m in magicks:

    print(m)

    train_subset = train[train[magic] == m]

    test_subset = test[test[magic] == m]

    model.fit(train_subset[data_cols], train_subset["target"])

    test_subset["target"] = model.predict(test_subset[data_cols])

    results.append(test_subset[["id", "target"]])

pd.concat(results).to_csv("submission.csv", index=False)