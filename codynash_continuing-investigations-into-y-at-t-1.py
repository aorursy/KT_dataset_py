# setup

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

%matplotlib inline



print("Loading file data.") ; 

with pd.HDFStore('../input/train.h5', "r") as train: train = train.get("train") 

min_y, max_y = min(train.y), max(train.y)

train.sort_values(by=['id', 'timestamp'], inplace=True)

train['y1'] = train.groupby('id')['y'].shift(1).fillna(0) # setup y(t-1) values
# my interpretation of the model from:

# https://www.kaggle.com/chenjx1005/two-sigma-financial-modeling/physical-meanings-of-technical-20-30/discussion

# using formula from

# https://www.kaggle.com/c/two-sigma-financial-modeling/discussion/29142



alpha1 = 0.92

alpha0 = 0.07

train['f0'] = train['technical_20'] - train['technical_30'] + train['technical_13'] 

train['f1'] = train.groupby('id')['f0'].shift(1).values

train['f2'] = ( train['f0'] - train['f1'] * alpha1 ) / alpha0

train['f2'] = train['f2'].clip(min_y,max_y).fillna(0)



train['f2d'] = train['y1'] - train['f2'] # check difference between true y1



print('Number points within 0.0005 of true y: {}'.format(np.sum( abs(train['f2d']) < 0.0005 )) )

plt.hist(train['f2d'],bins=200) ; plt.grid() ; plt.show()



n0s = 10000

plt.scatter( train['f2'][:n0s], train['y1'][:n0s], alpha=0.1 ) # use transparency to see dense regions

plt.grid() ; plt.show()
# Rbauld's notebook proposed using ewma's of a few other features, let's evaluate.

# It seems like t13 should be added as well.

# https://www.kaggle.com/rbauld/two-sigma-financial-modeling/rebuilding-y-t-1/discussion



train['y_shifted'] = train['y1']

def ewm_mean(x,span_in):

    return(x.ewm(span=span_in).mean())

train['EWM_26_mean_s']  = train.groupby('id')['y_shifted'].apply(lambda x: ewm_mean(x,span_in=26))



# ewm_features = ['technical_30','technical_20','technical_21','technical_19','technical_17','technical_11','technical_2']

# t13 added to ewm features

ewm_features = ['technical_30','technical_20','technical_21','technical_19','technical_17','technical_11','technical_2','technical_13']

mean_values = train[ewm_features].mean(axis=0)

train[ewm_features] = train[ewm_features].fillna(mean_values)



# n0 = int(len(train)/4) # really slow

n0 = 100000

import sklearn as sk

from sklearn import ensemble

my_model = sk.ensemble.GradientBoostingRegressor(loss='ls', max_depth=5, learning_rate=0.05)

my_model.fit(X=train.loc[:n0,ewm_features],y=train.loc[:n0,'EWM_26_mean_s'])

train['EWM_26s_pred'] = my_model.predict(X=train[ewm_features]) 



# Inverse transform

def ewm_reverse(data,span=26):

    alpha = 2/(span+1)

    return (data-(1-alpha)*data.shift(1).fillna(0))/alpha

train['yEWM_26'] = train.groupby('id')['EWM_26s_pred'].apply(lambda x: ewm_reverse(x, span=26))



train['f2'] = train['yEWM_26'].clip(min_y,max_y).fillna(0)

train['f2d'] = train['y1'] - train['f2']



print('Number points within 0.0005 of true y: {}'.format(np.sum( abs(train['f2d']) < 0.0005 )) )

plt.hist(train['f2d'],bins=200) ; plt.grid() ; plt.show()

n0s = 10000

plt.scatter( train['f2'][:n0s], train['y1'][:n0s], alpha=0.1 ) # use transparency to see dense regions

plt.grid() ; plt.show()
# It seems more elegant to only have a single alpha value, and that they would be related

# (As noted by Ricardus in the thread)

# this also allows us to scale the data easily

# lets look at the cases without t13, with t13 added and with t13 subtracted.



alpha0 = 0.9327 # identified by manual newtonian maximization



title0 = 't20-t30'

train['f0'] = train['technical_20'] - train['technical_30'] # + train['technical_13'] 

train['f1'] = train.groupby('id')['f0'].shift(1).values

train['f2'] = train['f0'] - train['f1'] * alpha0

train['f2'] = train['f2'] / (1-alpha0) # scale

train['f2'] = train['f2'].clip(min_y,max_y).fillna(0)

train['f2d'] = train['y1'] - train['f2'] # check difference between true y1

print(title0) ; print('Number points within 0.0005 of true y: {}'.format(np.sum( abs(train['f2d']) < 0.0005 )) )

plt.hist(train['f2d'],bins=200) ; plt.grid() ; plt.show()

n0s = 10000 ; plt.scatter( train['f2'][:n0s], train['y1'][:n0s], alpha=0.1 ) ; plt.grid() ; plt.show()



title0 = 't20-t30-t13'

train['f0'] = train['technical_20'] - train['technical_30'] - train['technical_13'] 

train['f1'] = train.groupby('id')['f0'].shift(1).values

train['f2'] = ( ( train['f0'] - train['f1'] * alpha0 ) / (1-alpha0) ).clip(min_y,max_y).fillna(0)

train['f2d'] = train['y1'] - train['f2'] # check difference between true y1

print(title0) ; print('Number points within 0.0005 of true y: {}'.format(np.sum( abs(train['f2d']) < 0.0005 )) )

plt.hist(train['f2d'],bins=200) ; plt.grid() ; plt.show()

n0s = 10000 ; plt.scatter( train['f2'][:n0s], train['y1'][:n0s], alpha=0.1 ) ; plt.grid() ; plt.show()



title0 = 't20-t30+t13'

train['f0'] = train['technical_20'] - train['technical_30'] + train['technical_13'] 

train['f1'] = train.groupby('id')['f0'].shift(1).values

train['f2'] = ( ( train['f0'] - train['f1'] * alpha0 ) / (1-alpha0) ).clip(min_y,max_y).fillna(0)

train['f2d'] = train['y1'] - train['f2'] # check difference between true y1

print(title0) ; print('Number points within 0.0005 of true y: {}'.format(np.sum( abs(train['f2d']) < 0.0005 )) )

plt.hist(train['f2d'],bins=200) ; plt.grid() ; plt.show()

n0s = 10000 ; plt.scatter( train['f2'][:n0s], train['y1'][:n0s], alpha=0.1 ) ; plt.grid() ; plt.show()

# Errors from t20t30t13 are frequently coincident, but there are other error components



plt.figure(figsize=(12,5))

n0s, n1s = 0, 200



ids0 = train.id.unique()

timestamps0 = train.timestamp.unique()

for id in ids0[:11]:

    rows0 = train.id.isin([id]) & train.timestamp.isin(timestamps0[n0s:n1s])

    plt.plot( train.loc[rows0,'timestamp'], train.loc[rows0,'f2d'] )

plt.grid()