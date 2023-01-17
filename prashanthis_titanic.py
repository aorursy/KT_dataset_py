import numpy as np

import pandas as pd

import os

from sklearn.ensemble import RandomForestClassifier



trn_data = pd.read_csv('/kaggle/input/titanic/train.csv')

tst_data = pd.read_csv("/kaggle/input/titanic/test.csv")



y = trn_data['Survived']



features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(trn_data[features])

X_test = pd.get_dummies(tst_data[features])



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': tst_data.PassengerId, 'Survived': predictions})

output.to_csv('reference_submission.csv', index=False)



print("Done")