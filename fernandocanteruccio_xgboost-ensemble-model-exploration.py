#XGBoost prediction script

#importing modules



import kagglegym

import numpy as np

import pandas as pd

import xgboost as xgb

from time import time
#Making environment ################################################################################################################################################################



# The "environment" is our interface for code competitions

env = kagglegym.make()



# We get our initial observation by calling "reset"

observation = env.reset()



train = observation.train

# Note that the first observation we get has a "train" dataframe

print("Train has {} rows".format(len(observation.train)))



# The "target" dataframe is a template for what we need to predict:

print("Target column names: {}".format(", ".join(['"{}"'.format(col) for col in list(observation.target.columns)])))
# Feature enginnering and preprocessing ############################################################################################################################################



# https://www.kaggle.com/bguberfain/two-sigma-financial-modeling/univariate-model-with-clip/run/482189/code

# Clipped target value range to use

low_y_cut = -0.08

high_y_cut = 0.08



y_is_above_cut = (train.y > high_y_cut)

y_is_below_cut = (train.y < low_y_cut)

y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)



# Select the features to use

excl = ['id', 'sample', 'y', 'timestamp']

#feature_vars = [c for c in train.columns if c not in excl]

target_var = 'y'



targets = train.loc[y_is_within_cut, target_var]

y_train = targets.values



del y_is_above_cut, y_is_below_cut, excl, target_var, targets
# Model training routine ###########################################################################################################################################################



# Univariate linear models, first layer <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



features_ulm = ['technical_20', 'technical_19', 'technical_27', 'technical_30', 'technical_2', 'technical_36']



features_ulm_train = train.loc[y_is_within_cut, features_ulm]

feature_ulm_names = features_ulm_train.columns

X_ulm = features_ulm_train.values



# Train dataset

xglin_train = xgb.DMatrix(X_ulm, label=y_train, feature_names=feature_ulm_names)



# XGb model params

params_xglin = {'booster'         :'gblinear',

                'objective'       :'reg:linear',

                'eta'             : 0.1,

                'max_depth'       : 4,

                'subsample'       : 0.9,

                'min_child_weight': 1000,

                'seed'            : 42,

                'base_score'      : 0

                }



print ("Training linear models")

t0 = time()

bslin = xgb.train(params_xglin, xglin_train, 10)

print("Done: %.1fs" % (time() - t0))



# Boosted trees ensemble, first layer <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



# https://www.kaggle.com/fernandocanteruccio/two-sigma-financial-modeling/xgboost-feature-importance-analysis

features_bt = ['technical_35', 'fundamental_37', 'technical_20', 'technical_36', 'fundamental_36', 'fundamental_53',

                'fundamental_35', 'fundamental_11', 'fundamental_50', 'fundamental_34']



features_bt_train = train.loc[y_is_within_cut, features_bt]

feature_bt_names = features_bt_train.columns

X_bt = features_bt_train.values



# Train dataset

xgtrees_train = xgb.DMatrix(X_bt, label=y_train, feature_names=feature_bt_names)



# XGb model params

params_xgtrees = {'objective'       :'reg:linear',

                  'eta'             : 0.1,

                  'max_depth'       : 4,

                  'subsample'       : 0.9,

                  'min_child_weight': 1000,

                  'seed'            : 42,

                  'base_score'      : 0

                   }



print ("Training boosted trees")

t0 = time()

bst = xgb.train(params_xgtrees, xgtrees_train, 10)

print("Done: %.1fs" % (time() - t0))
# Predict-step-predict routine ####################################################################################################################################################

def gen_predictions(update_threshold, print_info=True):

    

    global bslin, bst

    

    env = kagglegym.make()



    # We get our initial observation by calling "reset"

    observation = env.reset()



    train = observation.train



    params_xglin.update({'process_type': 'update',

                         'updater'     : 'refresh',

                         'refresh_leaf': False})



    params_xgtrees.update({'process_type': 'update',

                           'updater'     : 'refresh',

                           'refresh_leaf': False})



    # init aux vars

    reward = 0.0

    reward_log = []

    timestamps_log = []

    pos_count = 0

    neg_count = 0



    total_pos = []

    total_neg = []



    print("Predicting")

    t0= time()

    while True:

    #    observation.features.fillna(mean_values, inplace=True)



        # Predict with univariate linear models

        features_ulm_pred = observation.features.loc[:,features_ulm].values

        X_ulm_pred = xgb.DMatrix(features_ulm_pred, feature_names=feature_ulm_names)



        y_ulm_pred = bslin.predict(X_ulm_pred).clip(low_y_cut, high_y_cut)



        # Predict with boosted trees

        features_bt_pred = observation.features.loc[:,features_bt].values

        X_bt_pred = xgb.DMatrix(features_bt_pred, feature_names=feature_bt_names)



        y_bt_pred = bst.predict(X_bt_pred).clip(low_y_cut, high_y_cut)



        # Average the predictions

        averaged_out = np.mean(np.vstack((y_ulm_pred, y_bt_pred)), axis=0)



        # Fill target df with predictions 

        observation.target.y = averaged_out



        observation.target.fillna(0, inplace=True)

        target = observation.target

        timestamp = observation.features["timestamp"][0]

        obs_old = observation

        observation, reward, done, info = env.step(target)



        if update_threshold is not None:

            if (reward > update_threshold):

                # update boosted trees model

                xgtrees_update = xgb.DMatrix(obs_old.features.loc[:,features_bt].values, averaged_out, feature_names=feature_bt_names)



                bst = xgb.train(params_xgtrees, xgtrees_update, 10, xgb_model=bst)



                # update boosted linear model 

                xglin_update = xgb.DMatrix(obs_old.features.loc[:,features_ulm].values, averaged_out, feature_names=feature_ulm_names)



                bslin = xgb.train(params_xglin, xglin_update, 10, xgb_model=bslin)



        

        timestamps_log.append(timestamp)

        reward_log.append(reward)



        if (reward < 0):

            neg_count += 1

        else:

            pos_count += 1



        total_pos.append(pos_count)

        total_neg.append(neg_count)

        

        if timestamp % 100 == 0:

            if print_info:

                print("Timestamp #{}".format(timestamp))

                print("Step reward:", reward)

                print("Mean reward:", np.mean(reward_log[-timestamp:]))

                print("Positive rewards count: {0}, Negative rewards count: {1}".format(pos_count, neg_count))

                print("Positive reward %:", pos_count / (pos_count + neg_count) * 100)



            pos_count = 0

            neg_count = 0



        if done:

            break

    print("Done: %.1fs" % (time() - t0))

    print("Total reward sum:", np.sum(reward_log))

    print("Final reward mean:", np.mean(reward_log))

    print("Total positive rewards count: {0}, Total negative rewards count: {1}".format(np.sum(total_pos), np.sum(total_neg)))

    print("Final positive reward %:", np.sum(total_pos) / (np.sum(total_pos) + np.sum(total_neg)) * 100)

    print(info)



    return reward_log, timestamps_log, info['public_score']



reward_log, timestamps_log, score = gen_predictions(None)
%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns



sns.set(style="whitegrid");
fig, ax = plt.subplots(figsize=(12,7))

ax.set_title("Rewards distribution");

sns.distplot(reward_log, kde=True);

print("Rewards count:",np.array(reward_log).shape)
def moving_average(a, n=3) :

    ret = np.cumsum(a, dtype=float)

    ret[n:] = ret[n:] - ret[:-n]

    return ret[n - 1:] / n



ma_window = 33



fig, ax = plt.subplots(figsize=(12,7))

ax.set_xlabel("Timestamp");

ax.set_title("Reward signal over time");

sns.tsplot(reward_log,timestamps_log,ax=ax,color='b');

sns.tsplot(np.hstack((np.zeros(ma_window-1),moving_average(reward_log, ma_window)))

           ,timestamps_log,ax=ax,color='r');

ax.set_ylabel('Reward');
reward_log_2, timestamps_log, score_2 = gen_predictions(0.01,print_info=False)

print("Percent change:", (score_2 - score) / score * 100)
fig, ax = plt.subplots(figsize=(12,7))

ax.set_title("Rewards distribution");

sns.distplot(reward_log, kde=True, ax=ax, label='Without update',color='b');

sns.distplot(reward_log_2, kde=True, ax=ax, label='With update',color='g');

plt.legend();
fig, ax = plt.subplots(figsize=(12,7))

ax.set_xlabel("Timestamp");

ax.set_title("Averaged reward signal over time (window = 33)");

sns.tsplot(np.hstack((np.zeros(ma_window-1),moving_average(reward_log, ma_window)))

           ,timestamps_log,ax=ax,color='b');

sns.tsplot(np.hstack((np.zeros(ma_window-1),moving_average(reward_log_2, ma_window)))

           ,timestamps_log,ax=ax,color='g');

ax.set_ylabel('Reward');

ax.set_ylim([-0.23, -0.08]);
reward_log_3, timestamps_log, score_3 = gen_predictions(-0.03,print_info=False)
fig, ax = plt.subplots(figsize=(12,7))

ax.set_title("Rewards distribution");

sns.distplot(reward_log, kde=True, ax=ax, label='Without update',color='b');

sns.distplot(reward_log_2, kde=True, ax=ax, label='With update',color='g');

sns.distplot(reward_log_3, kde=True, ax=ax, label='More agressive update',color='r');

plt.legend();

print("Percent change:", (score_3 - score) / score * 100)
fig, ax = plt.subplots(figsize=(12,7))

ax.set_xlabel("Timestamp");

ax.set_title("Averaged reward signal over time (window = 33)");

sns.tsplot(np.hstack((np.zeros(ma_window-1),moving_average(reward_log, ma_window)))

           ,timestamps_log,ax=ax,color='b');

sns.tsplot(np.hstack((np.zeros(ma_window-1),moving_average(reward_log_2, ma_window)))

           ,timestamps_log,ax=ax,color='g');

sns.tsplot(np.hstack((np.zeros(ma_window-1),moving_average(reward_log_3, ma_window)))

           ,timestamps_log,ax=ax,color='r');

ax.set_ylabel('Averaged Reward');

ax.set_ylim([-0.23, -0.08]);