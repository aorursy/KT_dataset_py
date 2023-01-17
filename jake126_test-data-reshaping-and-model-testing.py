# library read in



import pandas as pd

pd.set_option('display.max_columns', None)

import numpy as np

import seaborn as sns

import matplotlib.style as style

style.use('fivethirtyeight')

import matplotlib.pylab as plt

import calendar

import warnings

warnings.filterwarnings("ignore")



import datetime

from time import time

from tqdm import tqdm_notebook as tqdm

from collections import Counter

from scipy import stats



from sklearn.model_selection import GroupKFold

from typing import Any

from numba import jit

import lightgbm as lgb

import xgboost as xgb

from catboost import CatBoostRegressor, CatBoostClassifier

from sklearn import metrics

from itertools import product
# load data

train_clean = pd.read_csv('../input/data-reshaping-and-rudimentary-models/cleaned_variables.csv')

test = pd.read_csv('../input/data-science-bowl-2019/test.csv')

# specs = pd.read_csv('../input/data-science-bowl-2019/specs.csv')

sample_submission = pd.read_csv('../input/data-science-bowl-2019/sample_submission.csv')
# reduce leakage

train_clean.loc[train_clean.title == 'Bird Measurer (Assessment)', 'BM_tries'] -= 1

train_clean.loc[train_clean.title == 'Cart Balancer (Assessment)', 'CB_tries'] -= 1

train_clean.loc[train_clean.title == 'Cauldron Filler (Assessment)', 'CF_tries'] -= 1

train_clean.loc[train_clean.title == 'Chest Sorter (Assessment)', 'CS_tries'] -= 1

train_clean.loc[train_clean.title == 'Mushroom Sorter (Assessment)', 'MS_tries'] -= 1



train_clean.loc[(train_clean.title == 'Bird Measurer (Assessment)') & (train_clean.assessment == "Success"), 'BM_passes'] -= 1

train_clean.loc[(train_clean.title == 'Cart Balancer (Assessment)') & (train_clean.assessment == "Success"), 'CB_passes'] -= 1

train_clean.loc[(train_clean.title == 'Cauldron Filler (Assessment)') & (train_clean.assessment == "Success"), 'CF_passes'] -= 1

train_clean.loc[(train_clean.title == 'Chest Sorter (Assessment)') & (train_clean.assessment == "Success"), 'CS_passes'] -= 1

train_clean.loc[(train_clean.title == 'Mushroom Sorter (Assessment)') & (train_clean.assessment == "Success"), 'MS_passes'] -= 1
# data cleaning (summary of exploration kernel)



# make sure order is correct/logical (it currently seems to be sorted by game_time. Some events occur simultaneously, and such rows are randomly sorted so some of the event_counts are out of order). With below sorting, events are in chronological order by installation_id

test = test.sort_values(["installation_id","timestamp","event_count"]).reset_index()



# create accurate indicator column of assessment success

test["successes"] = pd.np.where((test.type == "Assessment") & ((test.event_code == 4100) | (test.event_code == 4110)), # consider assessment outcome event_codes (4100 and 4110)

                             pd.np.where(((test.event_data.str.contains("\"correct\":true,\"caterpillars\"")) & (test.title == "Bird Measurer (Assessment)")) | ((test.event_data.str.contains("\"correct\":true")) & (test.event_code == 4100) & (test.title != "Bird Measurer (Assessment)")),"Success",# if a successful stage 1 Bird Measurer (event_code 4110) OR event_code 4100 in any other assessment (provided that the event_data has a 'correct' indicator), assessment was a success and counts towards accuracy measures

                                         pd.np.where(((test.event_data.str.contains("\"correct\":true,\"hats\"")) & (test.title == "Bird Measurer (Assessment)")), "Success (not measured)", # if a successful stage 2 Bird Measurer (event_code = 4100), assessment was a success but doesn't count towards accuracy measures 

                                                     pd.np.where(((test.event_data.str.contains("\"correct\":false,\"hats\"")) & (test.title == "Bird Measurer (Assessment)")),"Failure (not measured)", # if a failed stage 2 Bird Measurer (event_code = 4100), assessment was a failure but doesn't count towards accuracy measures

                                                                 "Failure")) # all remaining 4110 event_codes are failures

                                        ),

                                  "No test", # if not an assessment, straightforward) 

                       )
# cumulative time in TREETOPCITY, MAGMAPEAK and CRYSTALCAVES

TTC_time = []

MP_time = []

CC_time = []

# cumulative sessions in each world

TTC_events = []

MP_events = []

CC_events = []

# cumulative attempts on Bird Measurer, Cart Balancer, Cauldron Filler, Chest Sorter and Mushroom Sorter

BM_tries = []

CB_tries = []

CF_tries = []

CS_tries = []

MS_tries = []

# total passes on each assessment

BM_passes = []

CB_passes = []

CF_passes = []

CS_passes = []

MS_passes = []

# assessment name

ass_title = []





# create data frame of unique users, and keep track of rows for final merging

user_list = []

# keep track of session id, for merging

session_list = []

# keep track of timestamp, for further analysis

session_time = []

# installation id lifetime

ID_lifetime = []

# want a list of respondents - each takes a test next



test_users = test['installation_id'].unique()

# 1000 users in the set

for user in tqdm(test_users):

    user_list.append(user)

    temp_data = test[test.installation_id == user]

    # which assessment is being taken?

    ass_title.append(temp_data.title[-1:].item())

    session_list.append(temp_data.game_session[-1:].item())

    session_time.append(temp_data.timestamp[-1:].item())

    # for subsequent variables, we don't want to consider current assessment

    temp_data.drop(temp_data.tail(1).index,inplace=True)

    # considering all remaining, we can get all world and assessment tries/passes

    BM_tries.append(temp_data[(temp_data.title == "Bird Measurer (Assessment)") & ((temp_data.successes == "Success") | (temp_data.successes == "Success (not measured)") | (temp_data.successes == "Failure") | (temp_data.successes == "Failure (not measured)"))].shape[0])

    CB_tries.append(temp_data[(temp_data.title == "Cart Balancer (Assessment)") & ((temp_data.successes == "Success") | (temp_data.successes == "Failure"))].shape[0])

    CF_tries.append(temp_data[(temp_data.title == "Cauldron Filler (Assessment)") & ((temp_data.successes == "Success") | (temp_data.successes == "Failure"))].shape[0])

    CS_tries.append(temp_data[(temp_data.title == "Chest Sorter (Assessment)") & ((temp_data.successes == "Success") | (temp_data.successes == "Failure"))].shape[0])

    MS_tries.append(temp_data[(temp_data.title == "Mushroom Sorter (Assessment)") & ((temp_data.successes == "Success") | (temp_data.successes == "Failure"))].shape[0])

    BM_passes.append(temp_data[(temp_data.title == "Bird Measurer (Assessment)") & ((temp_data.successes == "Success") | (temp_data.successes == "Success (not measured)"))].shape[0])

    CB_passes.append(temp_data[(temp_data.title == "Cart Balancer (Assessment)") & (temp_data.successes == "Success")].shape[0])

    CF_passes.append(temp_data[(temp_data.title == "Cauldron Filler (Assessment)") & (temp_data.successes == "Success")].shape[0])

    CS_passes.append(temp_data[(temp_data.title == "Chest Sorter (Assessment)") & (temp_data.successes == "Success")].shape[0])

    MS_passes.append(temp_data[(temp_data.title == "Mushroom Sorter (Assessment)") & (temp_data.successes == "Success")].shape[0])

    # for subsequent variables, filter to final row for each session

    temp_data = temp_data.groupby("game_session").tail(1)

    TTC_time.append(sum(temp_data[temp_data.world == "TREETOPCITY"].game_time))

    MP_time.append(sum(temp_data[temp_data.world == "MAGMAPEAK"].game_time))

    CC_time.append(sum(temp_data[temp_data.world == "CRYSTALCAVES"].game_time))

    TTC_events.append(sum(temp_data[temp_data.world == "TREETOPCITY"].event_count))

    MP_events.append(sum(temp_data[temp_data.world == "MAGMAPEAK"].event_count))

    CC_events.append(sum(temp_data[temp_data.world == "CRYSTALCAVES"].event_count))

    ID_lifetime.append(sum(temp_data.game_time))
test_variables = pd.DataFrame({'user': test_users,

             'session': session_list,

             'session_time': session_time,

             'id_lifetime': ID_lifetime,

             'TTC_time': TTC_time,

             'MP_time': MP_time,

             'CC_time': CC_time,

             'TTC_events': TTC_events,

             'MP_events': MP_events,

             'CC_events': CC_events,

             'BM_tries': BM_tries,

             'CB_tries': CB_tries,

             'CF_tries': CF_tries,

             'CS_tries': CS_tries,

             'MS_tries': MS_tries,

             'BM_passes': BM_passes,

             'CB_passes': CB_passes,

             'CF_passes': CF_passes,

             'CS_passes': CS_passes,

             'MS_passes': MS_passes,

             'title': ass_title})
# map response variable to numeric for xgboost



train_clean.assessment[train_clean.assessment == "Success"] = 1

train_clean.assessment[train_clean.assessment == "Failure"] = 0



train_BM = train_clean[train_clean.title == "Bird Measurer (Assessment)"]

train_CB = train_clean[train_clean.title == "Cart Balancer (Assessment)"]

train_CF = train_clean[train_clean.title == "Cauldron Filler (Assessment)"]

train_CS = train_clean[train_clean.title == "Chest Sorter (Assessment)"]

train_MS = train_clean[train_clean.title == "Mushroom Sorter (Assessment)"]

test_BM = test[test.title == "Bird Measurer (Assessment)"]

test_CB = test[test.title == "Cart Balancer (Assessment)"]

test_CF = test[test.title == "Cauldron Filler (Assessment)"]

test_CS = test[test.title == "Chest Sorter (Assessment)"]

test_MS = test[test.title == "Mushroom Sorter (Assessment)"]
# xgboost



import xgboost as xgb

from sklearn.metrics import mean_squared_error

import pandas as pd

import numpy as np
X_BM, y_BM = train_BM[['TTC_events','MP_events','CC_events','BM_tries','CB_tries',

                       'CF_tries','CS_tries','MS_tries','BM_passes','CB_passes','CF_passes',

                       'CS_passes','MS_passes']],train_BM['assessment']

X_CB, y_CB = train_CB[['TTC_events','MP_events','CC_events','BM_tries','CB_tries',

                       'CF_tries','CS_tries','MS_tries','BM_passes','CB_passes','CF_passes',

                       'CS_passes','MS_passes']],train_CB['assessment']

X_CF, y_CF = train_CF[['TTC_events','MP_events','CC_events','BM_tries','CB_tries',

                       'CF_tries','CS_tries','MS_tries','BM_passes','CB_passes','CF_passes',

                       'CS_passes','MS_passes']],train_CF['assessment']

X_CS, y_CS = train_CS[['TTC_events','MP_events','CC_events','BM_tries','CB_tries',

                       'CF_tries','CS_tries','MS_tries','BM_passes','CB_passes','CF_passes',

                       'CS_passes','MS_passes']],train_CS['assessment']

X_MS, y_MS = train_MS[['TTC_events','MP_events','CC_events','BM_tries','CB_tries',

                       'CF_tries','CS_tries','MS_tries','BM_passes','CB_passes','CF_passes',

                       'CS_passes','MS_passes']],train_MS['assessment']
# train test split



from sklearn.model_selection import train_test_split



X_BM_train, X_BM_test, y_BM_train, y_BM_test = train_test_split(X_BM, y_BM, test_size=0.2, random_state=123)

X_CB_train, X_CB_test, y_CB_train, y_CB_test = train_test_split(X_CB, y_CB, test_size=0.2, random_state=123)

X_CF_train, X_CF_test, y_CF_train, y_CF_test = train_test_split(X_CF, y_CF, test_size=0.2, random_state=123)

X_CS_train, X_CS_test, y_CS_train, y_CS_test = train_test_split(X_CS, y_CS, test_size=0.2, random_state=123)

X_MS_train, X_MS_test, y_MS_train, y_MS_test = train_test_split(X_MS, y_MS, test_size=0.2, random_state=123)
pd.crosstab(X_BM_train.BM_tries,y_BM)
xg_reg_BM = xgb.XGBRegressor(objective ='reg:logistic', colsample_bytree = 0.3, learning_rate = 0.1,

                             max_depth = 5, alpha = 10, n_estimators = 10,scale_pos_weight = 3)

xg_reg_CB = xgb.XGBRegressor(objective ='reg:logistic', colsample_bytree = 0.3, learning_rate = 0.1,

                             max_depth = 5, alpha = 10, n_estimators = 10)

xg_reg_CF = xgb.XGBRegressor(objective ='reg:logistic', colsample_bytree = 0.3, learning_rate = 0.1,

                             max_depth = 5, alpha = 10, n_estimators = 10)

xg_reg_CS = xgb.XGBRegressor(objective ='reg:logistic', colsample_bytree = 0.3, learning_rate = 0.1,

                             max_depth = 5, alpha = 10, n_estimators = 10,scale_pos_weight = 10)

xg_reg_MS = xgb.XGBRegressor(objective ='reg:logistic', colsample_bytree = 0.3, learning_rate = 0.1,

                             max_depth = 5, alpha = 10, n_estimators = 10)
xg_reg_BM.fit(X_BM_train,y_BM_train)

xg_reg_CB.fit(X_CB_train,y_CB_train)

xg_reg_CF.fit(X_CF_train,y_CF_train)

xg_reg_CS.fit(X_CS_train,y_CS_train)

xg_reg_MS.fit(X_MS_train,y_MS_train)



preds_BM = xg_reg_BM.predict(X_BM_test)

preds_CB = xg_reg_CB.predict(X_CB_test)

preds_CF = xg_reg_CF.predict(X_CF_test)

preds_CS = xg_reg_CS.predict(X_CS_test)

preds_MS = xg_reg_MS.predict(X_MS_test)
rmse_BM = np.sqrt(mean_squared_error(y_BM_test, preds_BM))

print("RMSE: %f" % (rmse_BM))
X_BM_test = X_BM_test.reset_index()
X_BM_test['preds'] = preds_BM

X_BM_test['assessment'] = y_BM_test.reset_index().assessment
X_BM_test
print(str(X_BM_test[X_BM_test.assessment == 0].preds.mean()) + " " + str(X_BM_test[X_BM_test.assessment == 1].preds.mean()))
# test model on actual data



test_BM = test_variables[test_variables.title == "Bird Measurer (Assessment)"][['TTC_events','MP_events','CC_events','BM_tries','CB_tries','CF_tries','CS_tries','MS_tries','BM_passes',

                                                                                'CB_passes','CF_passes','CS_passes','MS_passes']]

test_CB = test_variables[test_variables.title == "Cart Balancer (Assessment)"][['TTC_events','MP_events','CC_events','BM_tries','CB_tries','CF_tries','CS_tries','MS_tries','BM_passes',

                                                                                'CB_passes','CF_passes','CS_passes','MS_passes']]

test_CF = test_variables[test_variables.title == "Cauldron Filler (Assessment)"][['TTC_events','MP_events','CC_events','BM_tries','CB_tries','CF_tries','CS_tries','MS_tries','BM_passes',

                                                                                'CB_passes','CF_passes','CS_passes','MS_passes']]

test_CS = test_variables[test_variables.title == "Chest Sorter (Assessment)"][['TTC_events','MP_events','CC_events','BM_tries','CB_tries','CF_tries','CS_tries','MS_tries','BM_passes',

                                                                                'CB_passes','CF_passes','CS_passes','MS_passes']]

test_MS = test_variables[test_variables.title == "Mushroom Sorter (Assessment)"][['TTC_events','MP_events','CC_events','BM_tries','CB_tries','CF_tries','CS_tries','MS_tries','BM_passes',

                                                                                'CB_passes','CF_passes','CS_passes','MS_passes']]
# calculate accuracy groups



preds_BM = xg_reg_BM.predict(test_BM)

preds_CB = xg_reg_CB.predict(test_CB)

preds_CF = xg_reg_CF.predict(test_CF)

preds_CS = xg_reg_CS.predict(test_CS)

preds_MS = xg_reg_MS.predict(test_MS)



exp_no_tries = pd.DataFrame({'probs':np.concatenate((preds_BM,preds_CB,preds_CF,preds_CS,preds_MS))})



exp_no_tries['exp_no'] = (1/exp_no_tries['probs']).round()



conditions = [

    (exp_no_tries['exp_no'] == 1),

    (exp_no_tries['exp_no'] == 2),

    (exp_no_tries['exp_no'] >= 3)]

choices = ['3', '2', '1']

exp_no_tries['accuracy_group'] = np.select(conditions, choices)
# calculate accuracy groups by likely assessment efforts; expected tries to first success = 1/p



output = pd.DataFrame({'installation_id':[test_variables[test_variables.title == "Bird Measurer (Assessment)"].user.append(

    test_variables[test_variables.title == "Cart Balancer (Assessment)"].user).append(

    test_variables[test_variables.title == "Cauldron Filler (Assessment)"].user).append(

    test_variables[test_variables.title == "Chest Sorter (Assessment)"].user).append(

    test_variables[test_variables.title == "Mushroom Sorter (Assessment)"].user)][0],

                      'accuracy_group':exp_no_tries['accuracy_group']})
output.to_csv("submission.csv", index = False)