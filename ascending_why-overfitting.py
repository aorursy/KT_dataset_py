import pandas as pd

import numpy as np

import kagglegym

import gc

import math

import time

import seaborn as sns

import xgboost as xgb

import matplotlib.pyplot as plt



%matplotlib inline
def get_reward(y_true, y_fit):

    R2 = 1 - np.sum((y_true - y_fit)**2) / np.sum((y_true - np.mean(y_true))**2)

    R = np.sign(R2) * math.sqrt(abs(R2))

    return(R)



def drop(group): 

    mean, std = group.mean(), group.std()

    inliers = (group - mean).abs() <= 1*std

    return inliers



def r2obj1(y_fit, dtrain):

    y_true = dtrain.get_label()

    deno = np.sum((y_true - np.mean(y_true))**2)

    grad = (-2.0/deno)*(y_true - y_fit)

    hess = (2.0/deno)*np.ones(shape=len(y_true))

    return grad, hess



def r2obj2(y_fit, dtrain):

    y_true = dtrain.get_label()

    grad = (-2.0*(y_true - y_fit))/deno

    hess = (2.0*np.ones(shape=len(y_true)))/deno

    return grad, hess



def r2eval(y_fit, dtrain):

    y_true = dtrain.get_label()

    R2_proxy = np.sum((y_true - y_fit)**2) / np.sum((y_true - np.mean(y_true))**2)

    return 'R2', R2_proxy
df_full = pd.read_hdf('../input/train.h5')



# Observed with histograns:

low_y_cut = -0.086093

high_y_cut = 0.093497



y_is_above_cut = (df_full.y > high_y_cut)

y_is_below_cut = (df_full.y < low_y_cut)

y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)



train = df_full.query('timestamp<=905')

test = df_full.query('timestamp>905')



mask = train.groupby('timestamp')['y'].apply(drop)

base_scor = train[mask].y.mean() #base score after removing outliers



train = train.loc[y_is_within_cut]

train.reset_index(drop=1,inplace=1)



train['tmp_deno'] = (train.y - train.timestamp.map(train.groupby('timestamp')['y'].mean()))**2

deno = train.timestamp.map(train.groupby('timestamp')['tmp_deno'].sum())

train.drop(['tmp_deno'],axis=1,inplace=1)



bsm = train.timestamp.map(train.groupby('timestamp')['y'].mean()).values #xgb base margin



feature_cols = [col for col in train.columns if col not in ['id','timestamp','y']]

cols = ['technical_20','technical_30']



dtrain = xgb.DMatrix(train[cols],label=train.y)

dtest = xgb.DMatrix(test[cols],label=test.y)



dtrain.set_base_margin(bsm)

dtest.set_base_margin([base_scor]*test.shape[0])
param = {

    'booster':'gbtree',

    'eta':0.01,

    'max_depth':2,

    #'gamma':0.1,

    'seed': 5

    }

watchlist = [(dtrain, 'train'), (dtest, 'eval')]

bst = xgb.train(param,dtrain,num_boost_round=1000,

                evals=watchlist,obj=r2obj2,feval=r2eval,

                maximize=False,early_stopping_rounds=10,verbose_eval=5)
bst.get_fscore()
np.sqrt(1-bst.best_score)
pred = bst.predict(dtest).clip(low_y_cut, high_y_cut)

get_reward(dtest.get_label(),pred)
env = kagglegym.make()



# We get our initial observation by calling "reset"

observation = env.reset()



y_actual_list = []

y_pred_list = []

r1_overall_reward_list = []

r1_reward_list = []

ts_list = []

while True:

    timestamp = observation.features["timestamp"][0]

    actual_y = list(df_full[df_full["timestamp"] == timestamp]["y"].values)

    test_x = xgb.DMatrix(observation.features[cols])

    test_x.set_base_margin([base_scor]*test_x.num_row())

    observation.target.y = bst.predict(test_x)

    

    target = observation.target

    observation, reward, done, info = env.step(target)

    

    if timestamp % 100 == 0:

        print("Timestamp #{}".format(timestamp))

    

    pred_y = list(target.y.values)

    y_actual_list.extend(actual_y)

    y_pred_list.extend(pred_y)

    overall_reward = get_reward(np.array(y_actual_list), np.array(y_pred_list))

    r1_overall_reward_list.append(overall_reward)

    r1_reward_list.append(reward)

    ts_list.append(timestamp)

    if done:

        break

    

print(info)
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(ts_list, r1_overall_reward_list, c='black', label='xgb_tree')

ax.plot(ts_list, [0]*len(ts_list), c='red', label='zero line')

ax.legend(loc='lower right')

ax.set_ylim([-0.04,0.04])

ax.set_xlim([850, 1850])

plt.title("Cumulative R value")

plt.show()