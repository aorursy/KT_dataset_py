import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from sklearn.preprocessing import LabelEncoder

from datetime import datetime

from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor

from sklearn.preprocessing import StandardScaler

import os
train_data = pd.read_csv('../input/titanic/train.csv')

test_data = pd.read_csv('../input/titanic/test.csv')

submission = pd.read_csv('../input/titanic/gender_submission.csv')
features = ["Pclass", "Sex", "SibSp", "Parch"]

y = train_data["Survived"]

x = pd.get_dummies(train_data[features])

x_test = pd.get_dummies(test_data[features])



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(x, y)

predictions = model.predict(x_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print(output)