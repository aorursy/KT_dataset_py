import numpy as np 

import pandas as pd 

import os

import math

from sklearn.linear_model import Ridge

from sklearn.metrics import cohen_kappa_score
xtrain = pd.read_csv('../input/liverpool-ion-switching/train.csv')

xtest = pd.read_csv('../input/liverpool-ion-switching/test.csv')
# add group info

n_groups = 100

xtrain["group"] = 0

for i in range(n_groups):

    ids = np.arange(i*50000, (i+1)*50000)

    xtrain.loc[ids,"group"] = i

    

n_groups = 40

xtest["group"] = 0

for i in range(n_groups):

    ids = np.arange(i*50000, (i+1)*50000)

    xtest.loc[ids,"group"] = i    
# min-max standarization

xtrain['signal_2'] = 0

xtest['signal_2'] = 0



n_groups = 100

for i in range(n_groups):

    sub = xtrain[xtrain.group == i]

    signals = sub.signal.values

    imax, imin = np.max(signals), np.min(signals)

    signals = (signals - np.min(signals))/(np.max(signals) - np.min(signals))

    signals = signals*(imax-imin)

    xtrain.loc[sub.index,"signal_2"] = [0,] +list(np.array(signals[:-1]))

    

n_groups = 40

for i in range(n_groups):

    sub = xtest[xtest.group == i]

    signals = sub.signal.values

    imax, imin = np.max(signals), np.min(signals)

    signals = (signals - np.min(signals))/(np.max(signals) - np.min(signals))

    signals = signals*(imax-imin)

    xtest.loc[sub.index,"signal_2"] = [0,] +list(np.array(signals[:-1]))
# min-max standarization

xtrain['signal_3'] = 0

xtest['signal_3'] = 0



n_groups = 100

for i in range(n_groups):

    sub = xtrain[xtrain.group == i]

    signals = sub.signal.values

    mx, mv = signals.mean(), signals.std()

    signals = (signals - mx)/mv

    xtrain.loc[sub.index,"signal_3"] = [0,] +list(np.array(signals[:-1]))

    

n_groups = 40

for i in range(n_groups):

    sub = xtest[xtest.group == i]

    signals = sub.signal.values

    mx, mv = signals.mean(), signals.std()

    signals = (signals - mx)/mv

    xtest.loc[sub.index,"signal_3"] = [0,] +list(np.array(signals[:-1]))
xtrain['fold'] = xtrain.group // 10

mvalid = np.zeros((xtrain.shape[0],1))

mfull = np.zeros((xtest.shape[0],1))



# features to use

feature_list = ['signal_2', 'signal_3']
for fold in range(10):

    id0 = np.where(xtrain.fold != fold)[0]

    id1 = np.where(xtrain.fold == fold)[0]

    

    x0 = xtrain[feature_list].iloc[id0]

    x1 = xtrain[feature_list].iloc[id1]

    y0 = xtrain[['open_channels']].iloc[id0]

    y1 = xtrain[['open_channels']].iloc[id1]

    

    model = Ridge(alpha= 2)

    model.fit(x0, y0)

    

    xpred = np.clip(model.predict(x1), 0, 10).astype(int)

    mvalid[id1,:] = xpred

    print(cohen_kappa_score(xpred, y1, weights = 'quadratic'))

    

print('final')

print(cohen_kappa_score(mvalid, xtrain[['open_channels']], weights = 'quadratic'))