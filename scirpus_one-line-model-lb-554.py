import numpy as np 

import pandas as pd 

from sklearn.metrics import r2_score
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
features = ['X0','X5','X261']
for c in features:

    print(c)

    catstats = np.zeros((test.shape[0],1))

    stats = train.groupby([c])['y'].mean().reset_index().rename(columns={c: c,'y': c+'_mean'})

    x = test.merge(stats, on=c, how='left')

    catstats[:, 0] = x[c+'_mean']

    test[c+'_mean'] = catstats[:, 0]

    test.drop(c,inplace=True,axis=1)

for c in features:

    print(c)

    catstats = np.zeros((train.shape[0],1))

    stats = train.groupby([c])['y'].mean().reset_index().rename(columns={c: c, 'y': c+'_mean'})

    x = train.merge(stats, on=c, how='left')

    catstats[:, 0] = x[c+'_mean']

    train[c+'_mean'] = catstats[:, 0]

    train.drop(c,inplace=True,axis=1)
train = train[['ID','X0_mean','X5_mean','X261_mean','y']]

test = test[['ID','X0_mean','X5_mean','X261_mean']]
train = train.fillna(train.mean())

test = test.fillna(test.mean())
def GP(data):

    p = (((((data["X0_mean"] + ((((((data["X5_mean"] + ((data["X261_mean"] + ((data["X5_mean"] + data["X0_mean"])/2.0))/2.0))/2.0) + data["X0_mean"])/2.0) + data["X0_mean"])/2.0))/2.0) + data["X0_mean"])/2.0))

    return p
r2_score(train.y, GP(train))
sub = pd.DataFrame({'ID':test.ID,'y':GP(test)})
sub.to_csv('gpsubmission.csv', index=False)