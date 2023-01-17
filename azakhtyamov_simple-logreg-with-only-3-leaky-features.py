import os

import json

from tqdm import tqdm

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import gridspec

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

%matplotlib inline
train = pd.read_json("../input/train.json")

train['inc_angle'] = pd.to_numeric(train['inc_angle'],errors='coerce')

test = pd.read_json("../input/test.json")

test['inc_angle'] = pd.to_numeric(test['inc_angle'],errors='coerce')

train = train[train['inc_angle'].isnull() == False].reset_index(drop = True)

train['mean'] = np.NaN

train['count'] = np.NaN

test['mean'] = np.NaN

test['count'] = np.NaN



train['neighbor_iceberg'] = np.NaN

test['neighbor_iceberg'] = np.NaN
for i in train.index:

    mean = train[np.abs( train['inc_angle'] - train.loc[i, 'inc_angle'] ) < 0.00000001]

    count = mean.shape[0] + test[ np.abs( test['inc_angle'] - \

                                          train.loc[i, 'inc_angle'] ) < 0.00000001 ].shape[0] - 1

    train.loc[i, 'count'] = count

    

    try:

        mean = mean.drop(i)['is_iceberg'].mean()

    except:

        mean = np.NaN



    if mean == mean:

        train.loc[i, 'mean'] = mean

    else:

        train.loc[i, 'mean'] = 0.5

    

    temp = train[np.abs( train['inc_angle'] - train.loc[i, 'inc_angle'] ) < 0.025].drop(i)

    train.loc[i,'neighbor_iceberg'] = sum(temp['is_iceberg']==1)*1./(1+temp.shape[0])

    

for i in test.index:

    mean = train[ np.abs( train['inc_angle'] - test.loc[i, 'inc_angle'] ) < 0.00000001 ]

    count = mean.shape[0] + test[ np.abs( test['inc_angle'] - \

                                          test.loc[i, 'inc_angle'] ) < 0.00000001 ].shape[0] - 1

    test.loc[i,'count'] = count

    

    mean = mean['is_iceberg'].mean()



    if mean == mean:

        test.loc[i, 'mean'] = mean

    else:

        test.loc[i, 'mean'] = 0.5

        

    temp = train[np.abs( train['inc_angle'] - test.loc[i, 'inc_angle'] ) < 0.025]

    test.loc[i,'neighbor_iceberg'] = sum(temp['is_iceberg']==1)*1./(1+temp.shape[0])
clf = LogisticRegression()

clf.fit(train[['mean', 'count', 'neighbor_iceberg']], train['is_iceberg'])
predictions = pd.DataFrame([])

predictions['id'] = test['id'].copy()

predictions['is_iceberg'] = clf.predict_proba(test[['mean', 'count', 'neighbor_iceberg']])[:,1]

predictions.to_csv("./submission.csv", index=False)