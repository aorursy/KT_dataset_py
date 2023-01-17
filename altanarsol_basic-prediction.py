import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv('../input/heart.csv')

df = df.fillna(0)
df.columns

y = df['target'].values

df = df.fillna(0)

X = df.drop(columns=['target'], axis=1).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=39)

model = GradientBoostingClassifier(random_state=39, n_estimators=50)

model.fit(X_train, y_train)

pred = model.predict(X_test)

accuracy = np.mean(pred == y_test)

print('accuracy: ', accuracy*100, '%')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=39)

model = RandomForestClassifier(random_state=39, n_estimators=100)

model.fit(X_train, y_train)

pred = model.predict(X_test)

accuracy = np.mean(pred == y_test)

print('accuracy: ', accuracy*100, '%')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=33)

model = RandomForestClassifier(random_state=33, n_estimators=100)

model.fit(X_train, y_train)

pred = model.predict(X_test)

accuracy = np.mean(pred == y_test)

print('accuracy: ', accuracy*100, '%')