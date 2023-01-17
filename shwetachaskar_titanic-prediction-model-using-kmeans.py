import pandas as pd

import numpy as np

from sklearn.cluster import KMeans

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import MinMaxScaler

# Transform features by scaling each feature to a given range.

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
print("***** Train_Set *****")

print(train.head())

print("\n")

print("***** Test_Set *****")

print(test.head())
print("***** Train_Set *****")

print(train.describe())

print("\n")

print("***** Test_Set *****")

print(test.describe())
print("*****In the train set*****")

print(train.isna().sum())

print("\n")

print("*****In the test set*****")

print(test.isna().sum())
train.fillna(train.mean(), inplace=True)
test.fillna(test.mean(),inplace=True)
g = sns.FacetGrid(train, col='Survived')

g.map(plt.hist, 'Age', bins=20)
grid = sns.FacetGrid(train, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
train = train.drop(['Name','Ticket', 'Cabin','Embarked'], axis=1)

test = test.drop(['Name','Ticket', 'Cabin','Embarked'], axis=1)
labelEncoder = LabelEncoder()

labelEncoder.fit(train['Sex'])

labelEncoder.fit(test['Sex'])

train['Sex'] = labelEncoder.transform(train['Sex'])

test['Sex'] = labelEncoder.transform(test['Sex'])
train.info()
test.info()
X = np.array(train.drop(['Survived'], 1).astype(float))
y = np.array(train['Survived'])
train.info()
kmeans = KMeans(n_clusters=2)

kmeans.fit(X)
correct = 0

for i in range(len(X)):

    predict_me = np.array(X[i].astype(float))

    predict_me = predict_me.reshape(-1, len(predict_me))

    prediction = kmeans.predict(predict_me)

    if prediction[0] == y[i]:

        correct += 1

print(correct/len(X))
kmeans = kmeans = KMeans(n_clusters=2, max_iter=600, algorithm = 'auto')

kmeans.fit(X)
correct = 0

for i in range(len(X)):

    predict_me = np.array(X[i].astype(float))

    predict_me = predict_me.reshape(-1, len(predict_me))

    prediction = kmeans.predict(predict_me)

    if prediction[0] == y[i]:

        correct += 1

print(correct/len(X))
scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)
kmeans.fit(X_scaled)
correct = 0

for i in range(len(X)):

    predict_me = np.array(X[i].astype(float))

    predict_me = predict_me.reshape(-1, len(predict_me))

    prediction = kmeans.predict(predict_me)

    if prediction[0] == y[i]:

        correct += 1

print(correct/len(X))