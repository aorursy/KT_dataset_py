import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score
train = pd.read_csv("../input/liverpool-ion-switching/train.csv")

test = pd.read_csv("../input/liverpool-ion-switching/test.csv")
train.shape
test.shape
plt.figure(figsize=(15,7))

plt.xlabel("Open Channels")

plt.ylabel("Counts of Open Channel")

sns.countplot(train['open_channels'])
train.groupby('signal')['open_channels'].apply(lambda x: len(set(x))).plot()
X = train[['time', 'signal']].values

y = train['open_channels'].values
sc = StandardScaler()

X = sc.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
model = DecisionTreeClassifier()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

accuracy_score(y_pred,y_test)
model = ExtraTreesClassifier()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

accuracy_score(y_pred,y_test)