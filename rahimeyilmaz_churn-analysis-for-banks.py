#Importing libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats



from sklearn.model_selection import train_test_split,cross_val_score

from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('../input/predicting-churn-for-bank-customers/Churn_Modelling.csv')

data.head()
nRow, nCol = data.shape

print(f'There are {nRow} rows and {nCol} columns')
data = data.drop(["RowNumber", "CustomerId", "Surname"], axis = 1)

data.head()
nRow, nCol = data.shape

print(f'There are {nRow} rows and {nCol} columns')
char_cols = data.dtypes.pipe(lambda x: x[x == 'object']).index

for c in char_cols:

    data[c] = pd.factorize(data[c])[0]

print(data)
f,ax = plt.subplots(figsize=(12,12))

sns.heatmap(data.corr(), annot = True,linewidths=.5, fmt = '.1f' , ax=ax ,cmap="YlGnBu")
churn = data[data['Exited'] == 1]

non_churn = data[data['Exited'] == 0]
data_train = data.sample(frac=0.8,random_state=200)

data_test = data.drop(data_train.index)

print(len(data_train))

print(len(data_test))
#X, y = make_hastie_10_2(random_state=0)

#X_train, X_test = X[:1000], X[1000:]

#y_train, y_test = y[:1000], y[1000:]



rfc = RandomForestClassifier(n_estimators=10, criterion = 'entropy' )

rfc.fit(X_train, y_train)
success = cross_val_score(estimator = rfc, X=X_train, y=y_train, cv=4)

print(success.mean())

print(success.std())