import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib

from sklearn import preprocessing 

import matplotlib.pyplot as plt

from scipy.stats import skew

from scipy.stats.stats import pearsonr





%config InlineBackend.figure_format = 'retina' #set 'png' here when working on notebook

%matplotlib inline
train = pd.read_csv('../input/dataset/Train_full.csv')

test = pd.read_csv('../input/dataset-v2/Test_small_features.csv')
test.shape
train.shape
train.head()
test.head()
# test['up_down'] = test['hour']

# for i in range(0,test.shape[0]):

#     if i == 0: continue

#     else:

#         if (test.at[i,'body'] > 0):

#             test.at[i-1,'up_down'] = 1

#         else:

#             test.at[i-1,'up_down'] = 0
# test.at[test.shape[0]-1, 'up_down'] = 1
# multiplier = 2/27

# # print(train.at[0,'EMA_20'])

# train['EMA_26'] = train['Close']

# prev_value = 0

# for i in train['EMA_26'].index:

#     if i == 0: 

#         prev_value = train.at[i,'EMA_26']

#         continue

#     else:

# #         print(all_data.at[i,'EMA_50'])

#         train.at[i,'EMA_26'] = multiplier*train.at[i,'EMA_26'] + prev_value*(1-multiplier)

#         prev_value = train.at[i,'EMA_26']

    

# # print(all_data['EMA_50'])
# multiplier = 2/13

# # print(train.at[0,'EMA_20'])

# train['EMA_12'] = train['Close']

# prev_value = 0

# for i in train['EMA_12'].index:

#     if i == 0: 

#         prev_value = train.at[i,'EMA_12']

#         continue

#     else:

# #         print(all_data.at[i,'EMA_50'])

#         train.at[i,'EMA_12'] = multiplier*train.at[i,'EMA_12'] + prev_value*(1-multiplier)

#         prev_value = train.at[i,'EMA_12']

    

# # print(all_data['EMA_50'])
# train.head()
# multiplier = 2/27

# # print(train.at[0,'EMA_20'])

# test['EMA_26'] = test['Close']

# prev_value = 0

# for i in test['EMA_26'].index:

#     if i == 0: 

#         prev_value = test.at[i,'EMA_26']

#         continue

#     else:

# #         print(all_data.at[i,'EMA_50'])

#         test.at[i,'EMA_26'] = multiplier*train.at[i,'EMA_26'] + prev_value*(1-multiplier)

#         prev_value = test.at[i,'EMA_26']

    

# # print(all_data['EMA_50'])
# multiplier = 2/13

# # print(train.at[0,'EMA_20'])

# test['EMA_12'] = test['Close']

# prev_value = 0

# for i in test['EMA_12'].index:

#     if i == 0: 

#         prev_value = test.at[i,'EMA_12']

#         continue

#     else:

# #         print(all_data.at[i,'EMA_50'])

#         test.at[i,'EMA_12'] = multiplier*train.at[i,'EMA_12'] + prev_value*(1-multiplier)

#         prev_value = test.at[i,'EMA_12']

    

# # print(all_data['EMA_50'])
# test.head()
#merge all data

all_data = pd.concat((train.loc[:,'Open':'lag_return_96'],

                      test.loc[:,'Open':'lag_return_96']))

all_data.head()
# all_data['closedivopen'] = all_data['Close']/all_data['Open']

# all_data['delta_SMA'] = all_data['SMA_20'] - all_data['SMA_50']

# all_data['delta_EMA'] = all_data['EMA_12'] - all_data['EMA_26']

# all_data['amount'] = all_data['Volume']*all_data['return_2']
all_data = all_data.drop(['Volume', 'upper_tail','lower_tail'], axis = 1)

# all_data = all_data.drop(['upper_tail','lower_tail'], axis = 1)
X_train = all_data[:train.shape[0]]

X_test = all_data[train.shape[0]:]

y = train.up_down
# X_train = train.drop(['Unnamed: 0', 'Volume', 'upper_tail','lower_tail'], axis = 1)

# X_test = test.drop(['Unnamed: 0', 'Volume', 'upper_tail','lower_tail'], axis = 1)
# X_train.head()
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

import xgboost
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

valid_score = []

num_model = 0

pred_test = np.zeros(len(X_test))

for i in range(1, 9, 1):

    num_model += 1

    model = XGBClassifier(max_depth=5, n_estimators=2000, n_jobs=16, 

                          random_state=4, subsample=0.9, gpu_id=0, 

                          colsample_bytree=0.9, max_bin=512, tree_method='gpu_hist')

    x_train, x_valid, y_train, y_valid = train_test_split(X_train, y, test_size = i*0.1, random_state = 8, shuffle = False)

    model.fit(x_train, y_train)

    y_pred = model.predict(x_valid)

    valid_score.append(accuracy_score(y_pred, y_valid))

    pred_test += model.predict(X_test)/num_model

means = np.mean(valid_score)

var = np.std(valid_score)
print(means)

print(var)
# y = all_data['up_down']

# x_train, x_valid, y_train, y_valid = train_test_split(all_data, y, test_size = 0.2, stratify = dayofweek, random_state = 8, shuffle = True)

# x_train, x_valid, y_train, y_valid = train_test_split(X_train, y, test_size = 0.2, random_state = 8, shuffle = True)

# # x_train = X_train.loc[:,'Open':'lag_return_96']

# # y_train = X_train['up_down']

# # x_valid = X_test.loc[:,'Open':'lag_return_96']

# # y_valid = X_test['up_down']
# from xgboost import XGBClassifier

# model = XGBClassifier(max_depth = 5, n_estimators = 2000, n_jobs = 16, random_state = 4, subsample = 0.9, gpu_id = 0, colsample_bytree = 0.9, max_bin = 16, tree_method = 'gpu_hist')

# model.fit(X=x_train,y=y_train,eval_set = [(x_train,y_train),(x_valid, y_valid)], eval_metric = ['logloss'], early_stopping_rounds = 70)

# model.fit(X=x_train,y=y_train,eval_set = [(x_train,y_train),(x_valid, y_valid)], eval_metric = ['error'])
# from xgboost import plot_importance

# plot_importance(model, max_num_features = 20)

# prediction = model.predict(test.drop(['Unnamed: 0', 'Volume', 'upper_tail','lower_tail','up_down'], axis = 1), ntree_limit = model.best_ntree_limit)
# prediction
# mysubmit = pd.DataFrame({'up_down': prediction})
# mysubmit.shape
# mysubmit.to_csv('submission.csv', index=True)