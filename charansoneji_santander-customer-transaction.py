import pandas as pd

import numpy as np
train_data=pd.read_csv('../input/santander-customer-transaction-prediction/train.csv',sep=',')

train_data.head()
train_data.shape

len_train = train_data.shape[0]
import seaborn as sns

import matplotlib.pyplot as plt

sns.scatterplot(x=train_data["target"],y=train_data['var_0'])
train_ID=train_data["ID_code"]

train_labels=train_data["target"]

features = train_data.columns[train_data.columns.str.startswith('var')].tolist()
test_data=pd.read_csv("../input/santander-customer-transaction-prediction/test.csv")

test_data.head()
def scale_data(X, scaler=None):

    if not scaler:

        scaler = MinMaxScaler(feature_range=(-1, 1))

#        scaler = StandardScaler()

        scaler.fit(X)

    X = scaler.transform(X)

    return X, scaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaled, scaler = scale_data(np.concatenate((train_data[features].values, test_data[features].values), axis=0))
train_data[features] = scaled[:len_train]

test_data[features] = scaled[len_train:]
train_data.head()
test_data.head()
train = train_data.drop(['target', 'ID_code'], axis=1).values

test=test_data.drop(["ID_code"],axis=1).values
from sklearn.model_selection import StratifiedKFold

from sklearn.svm import LinearSVC

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=101)

for i, (train_index, val_index) in enumerate(skf.split(train,train_labels)):

    Xtrain, Xval = train[train_index], train[val_index]

    ytrain, yval = train_labels[train_index], train_labels[val_index]

    model = LinearSVC(C=0.01, tol=0.0001, verbose=1, random_state=1001, max_iter=2000, dual=False)
model.fit(Xtrain, ytrain)
y_pred = model.predict(test)

y_pred
from sklearn.metrics import accuracy_score,r2_score

accuracy_score(train_data["target"],y_pred)