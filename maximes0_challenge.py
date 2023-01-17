import numpy as np

from sklearn.tree import DecisionTreeClassifier

from pandas import read_csv, DataFrame, concat

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
data_train = read_csv('../input/Xtrain.csv', header=0)  # Training input data

label_train = read_csv('../input/ytrain.csv', header=None)  # Training labels

data_test = read_csv('../input/Xtest.csv', header=0)  # Test input data
concat([data_train.dtypes, data_train.isna().any(axis=0)], keys=['type', 'missing value?'], axis=1, sort=False)
data_train = data_train.fillna(value=0)  # Fill missing valxues with 0

data_test = data_test.fillna(value=0)  # Fill missing values with 0
data = data_train.copy()

data['severity'] = label_train[0]

data.head()
plt.figure(figsize=(10, 6))

sns.distplot(data['severity']);
plt.figure(figsize=(15, 7))

sns.violinplot(y='severity', x='road', data=data)
plt.figure(figsize=(15, 7))

sns.violinplot(y='severity', x='user', data=data)
plt.figure(figsize=(15, 7))

sns.violinplot(y='severity', x='pedlocation', data=data)
X_train = data_train.values  # Training input data as Numpy array

y_train = label_train.values.ravel()  # Training labels as Numpy array

X_test = data_test.values  # Test input data as Numpy array
print('Shape of the training dataset:', X_train.shape)

print('Shape of the test dataset:', X_test.shape)
clf = DecisionTreeClassifier(max_depth=3)

clf.fit(X_train, y_train)

print('Tree score:', clf.score(X_train, y_train))
y_pred = clf.predict(X_test)  # Label predictions for the test set

DataFrame(y_pred).to_csv('ypred.csv', index_label='id', header=['prediction'])  # Save prediction