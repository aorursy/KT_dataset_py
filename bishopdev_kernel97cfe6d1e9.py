import numpy as np

import pandas as pd

import seaborn as sns

from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import SGDClassifier



import os

print(os.listdir("../input"))
data = pd.read_csv('../input/winequality-red.csv')
data.head()
data.describe()
sns.pairplot(data)
results = data[['quality']]

results.head()
data.drop(columns = ['quality'])
x_train, x_test, y_train, y_test = train_test_split(data, results, test_size = 0.2, random_state = 42)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)

x_test = sc.fit_transform(x_test)
rfc = RandomForestClassifier(n_estimators=200)

rfc.fit(x_train, y_train)

pred_rfc = rfc.predict(x_test)
print(classification_report(y_test, pred_rfc))
x2_train = np.vstack([x_train, x_train, x_train, x_train, x_train, x_train, x_train])

y2_train = np.vstack([y_train, y_train, y_train, y_train, y_train, y_train, y_train])
rfc2 = RandomForestClassifier(n_estimators=200)

rfc2.fit(x2_train, y2_train)

pred_rfc2 = rfc2.predict(x_test)

print(classification_report(y_test, pred_rfc2))
sgd = SGDClassifier(penalty=None)

sgd.fit(x_train, y_train)

pred_sgd = sgd.predict(x_test)

print(classification_report(y_test, pred_sgd))
sns.countplot(results['quality'])
borders = (0, 6.9, 10)

groups = [0, 1]

results['quality'] = pd.cut(results['quality'], bins = borders, labels = groups)

results
sns.countplot(results['quality'])
x_binary_train, x_binary_test, y_binary_train, y_binary_test = train_test_split(data, results, test_size = 0.2, random_state = 42)
x_binary_train = sc.fit_transform(x_train)

x_binary_test = sc.fit_transform(x_test)
rfc_binary = RandomForestClassifier(n_estimators=200)

rfc_binary.fit(x_binary_train, y_binary_train)

pred_rfc_binary = rfc_binary.predict(x_binary_test)
print(classification_report(y_binary_test, pred_rfc_binary))