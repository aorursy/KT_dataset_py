import numpy as np

import pandas as pd

from sklearn import model_selection, preprocessing

from sklearn.ensemble import IsolationForest

rng = np.random.RandomState(42)

import xgboost as xgb



import datetime



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

macro = pd.read_csv('../input/macro.csv')

id_test = test.id
train.head()
train.shape
test.shape
macro.shape


for c in train.columns:

    if train[c].dtype == 'object':

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(train[c].values)) 

        train[c] = lbl.transform(list(train[c].values))

        #x_train.drop(c,axis=1,inplace=True)

        

for c in test.columns:

    if test[c].dtype == 'object':

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(test[c].values)) 

        test[c] = lbl.transform(list(test[c].values))

        #x_test.drop(c,axis=1,inplace=True)
train.head()
train.shape
def get_outliners(dataset):

    clf = IsolationForest(random_state=rng)

    clf.fit(dataset)

    result = clf.predict(dataset)

    return result
train.fillna(-999,inplace=True)

test.fillna(-999,inplace=True)
train.head




training_dataset = train[get_outliners(train)]
training_dataset.head().T
macro.corr().to_csv('corr_vals.csv')