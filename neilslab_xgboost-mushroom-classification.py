import pandas as pd

df = pd.read_csv('../input/mushrooms.csv')

print("Dataset shape: ", df.shape)

df.head()
df.dtypes
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

dataset = df.apply(le.fit_transform)

dataset.head()
targets = dataset['class']

targets.head()
columns = dataset.columns

data = pd.DataFrame(dataset['class'])

for i in (columns):

    x = pd.get_dummies(dataset[i])

    data = data.join(x, lsuffix='_left', rsuffix='_right')

features = data.drop('class', axis=1)

print(features.shape)

features = features.values
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(features, targets,

                                                   test_size=0.2,

                                                   random_state=0)
eval_set = [(X_test, Y_test)]

model = XGBClassifier()

model.fit(X_train, Y_train, eval_metric="error", eval_set=eval_set, verbose=False)
from sklearn.metrics import accuracy_score

pred = model.predict(features)

score = accuracy_score(targets, pred)

print("Classification Accuracy Score: ", score * 100)