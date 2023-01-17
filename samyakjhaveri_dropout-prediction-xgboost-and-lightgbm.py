import pandas as pd # used for dataframes

import numpy as np 

import xgboost as xgb # Gradient Boosting Algorithm

import matplotlib.pyplot as plt

import seaborn as sns

import gc # Garbage Collector required to extract unused and residual data and variables from memory

from sklearn.metrics import mean_squared_error as mse

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
fields1 = ['username', 'course_id', 'action', 'truth'] # specific columns to load into the dataframe

fields2 = ['username', 'course_id', 'time']

gc.enable()
data_train_action = pd.read_csv('../input/mooc-final/train/train.csv', usecols = fields1, nrows = 14582760) # load specific columns from train.csv

data_train_action.info()
data_train_time = pd.read_csv('../input/mooc-final/train/train.csv', usecols = fields2, nrows = 14582760) # load specific columns from train.csv

data_train_time.info()
data_test_action = pd.read_csv('../input/mooc-final/test/test.csv', usecols = fields1, nrows = 6472430) # load specific columns from test.csv

data_test_action.info()
data_test_time = pd.read_csv('../input/mooc-final/test/test.csv', usecols = fields2, nrows = 6472430) # load specific columns from test.csv

data_test_time.info()
data_train_action = pd.get_dummies(data_train_action, columns = ['action']) # Getting dummies of 'action' column to convert 'object' type data into float values

data_test_action = pd.get_dummies(data_test_action, columns = ['action'])
data_train_time['Datetime'] = pd.to_datetime(data_train_time['time']) # Converting 'time' column of 'object' type

data_test_time['Datetime'] = pd.to_datetime(data_test_time['time'])
data_train_time = data_train_time.drop(['time'], axis = 1) # Dropping 'time' column to reduce memory usage

data_test_time = data_test_time.drop(['time'], axis = 1)
gc.collect()
data_train_time['timestamp'] = data_train_time.Datetime.values.astype(np.int64) // 10 ** 9 # Converting data from 'datetime' type to timestamp

data_test_time['timestamp'] = data_test_time.Datetime.values.astype(np.int64) // 10 ** 9
data_train_time = data_train_time.drop(['Datetime'], axis = 1) # Dropping 'Datetime' column to reduce memory usage

data_test_time = data_test_time.drop(['Datetime'], axis = 1)
gc.collect()
data_train_time['time_difference'] = pd.DataFrame(data_train_time.timestamp.diff()) # Calculating difference in timestamps of consecutive activities

data_test_time['time_difference'] = pd.DataFrame(data_test_time.timestamp.diff())
data_train_time = data_train_time.groupby(['username', 'course_id']).sum() # Grouping data into unique user-course pairs

data_train_time = pd.DataFrame(data_train_time.reset_index())



data_test_time = data_test_time.groupby(['username', 'course_id']).sum()

data_test_time = pd.DataFrame(data_test_time.reset_index())



data_train_action = pd.DataFrame(data_train_action.groupby(['username', 'course_id']).sum())

data_train_action = pd.DataFrame(data_train_action.reset_index())



data_test_action = pd.DataFrame(data_test_action.groupby(['username', 'course_id']).sum())

data_test_action = pd.DataFrame(data_test_action.reset_index())
data_train = pd.merge(data_train_action, data_train_time, left_index = True, right_index = True) # merging data_train_time and data_train_action into a single dataframe
del data_train_action

del data_train_time
gc.collect()
data_test = pd.merge(data_test_action, data_test_time, left_index = True, right_index = True) # merging data_test_time and data_test_action into a single dataframe
del data_test_action

del data_test_time
gc.collect()
data_train['truth'] = np.where(data_train['truth'] >= 1, 1,0) # Converting all non-zero values into 1, sine the XGBoost Classifier algorithm requires binary (0 or 1) as 

# input data

data_train['action_click_about'] = np.where(data_train['action_click_about'] >= 1, 1,0)

data_train['action_click_courseware'] = np.where(data_train['action_click_courseware'] >= 1, 1,0)

data_train['action_click_forum'] = np.where(data_train['action_click_forum'] >= 1, 1,0)

data_train['action_click_info'] = np.where(data_train['action_click_info'] >= 1, 1,0)

data_train['action_click_progress'] = np.where(data_train['action_click_progress'] >= 1, 1,0)

data_train['action_close_courseware'] = np.where(data_train['action_close_courseware'] >= 1, 1,0)

data_train['action_delete_comment'] = np.where(data_train['action_delete_comment'] >= 1, 1,0)

data_train['action_load_video'] = np.where(data_train['action_load_video'] >= 1, 1,0)

data_train['action_pause_video'] = np.where(data_train['action_pause_video'] >= 1, 1,0)

data_train['action_play_video'] = np.where(data_train['action_play_video'] >= 1, 1,0)

data_train['action_problem_check_correct'] = np.where(data_train['action_problem_check_correct'] >= 1, 1,0)

data_train['action_problem_get'] = np.where(data_train['action_problem_get'] >= 1, 1,0)

data_train['action_problem_save'] = np.where(data_train['action_problem_save'] >= 1, 1,0)

data_train['action_seek_video'] = np.where(data_train['action_seek_video'] >= 1, 1,0)
data_test['truth'] = np.where(data_test['truth'] >= 1, 1,0)

data_test['action_click_about'] = np.where(data_test['action_click_about'] >= 1, 1,0)

data_test['action_click_courseware'] = np.where(data_test['action_click_courseware'] >= 1, 1,0)

data_test['action_click_forum'] = np.where(data_test['action_click_forum'] >= 1, 1,0)

data_test['action_click_info'] = np.where(data_test['action_click_info'] >= 1, 1,0)

data_test['action_click_progress'] = np.where(data_test['action_click_progress'] >= 1, 1,0)

data_test['action_close_courseware'] = np.where(data_test['action_close_courseware'] >= 1, 1,0)

data_test['action_delete_comment'] = np.where(data_test['action_delete_comment'] >= 1, 1,0)

data_test['action_load_video'] = np.where(data_test['action_load_video'] >= 1, 1,0)

data_test['action_pause_video'] = np.where(data_test['action_pause_video'] >= 1, 1,0)

data_test['action_play_video'] = np.where(data_test['action_play_video'] >= 1, 1,0)

data_test['action_problem_check_correct'] = np.where(data_test['action_problem_check_correct'] >= 1, 1,0)

data_test['action_problem_get'] = np.where(data_test['action_problem_get'] >= 1, 1,0)

data_test['action_problem_save'] = np.where(data_test['action_problem_save'] >= 1, 1,0)

data_test['action_seek_video'] = np.where(data_test['action_seek_video'] >= 1, 1,0)
data_train.head(10)
data_test.head(10)
train_length = len(data_train)

print(train_length)
data_train.head(10)
data_train.tail(10)
data_train1 = data_train.loc[:int(train_length/2)] # Splitting data_train into two halves to make training efficient
data_train1.head(10)
data_train1.tail(10)
data_train2 = data_train.loc[(int(train_length/2) + 1):]
data_train2.head(10)
data_train2.tail(10)
del data_train
gc.collect()
train_labels1 = data_train1['truth']

train_features1 = data_train1[['timestamp', 'time_difference',

                            'action_click_about', 'action_click_courseware', 'action_click_forum', 

                             'action_click_info', 'action_click_progress', 'action_close_courseware', 

                             'action_delete_comment', 'action_load_video', 'action_pause_video', 'action_play_video', 'action_problem_check_correct',

                             'action_problem_get', 'action_problem_save', 'action_seek_video']]



x_train1 = train_features1

y_train1 = np.ravel(train_labels1)
train_labels2 = data_train2['truth']

train_features2 = data_train2[['timestamp', 'time_difference',

                            'action_click_about', 'action_click_courseware', 'action_click_forum', 

                             'action_click_info', 'action_click_progress', 'action_close_courseware', 

                             'action_delete_comment', 'action_load_video', 'action_pause_video', 'action_play_video', 'action_problem_check_correct',

                             'action_problem_get', 'action_problem_save', 'action_seek_video']]



x_train2 = train_features2

y_train2 = np.ravel(train_labels2)
test_length = len(data_test)

print(test_length)
data_test1 = data_test.loc[:int(test_length/4)] # Splitting data_train to make testing efficient
test_labels1 = data_test1['truth']

test_features1 = data_test1[['timestamp', 'time_difference',

                            'action_click_about', 'action_click_courseware', 'action_click_forum', 

                             'action_click_info', 'action_click_progress', 'action_close_courseware', 

                             'action_delete_comment', 'action_load_video', 'action_pause_video', 'action_play_video', 'action_problem_check_correct',

                             'action_problem_get', 'action_problem_save', 'action_seek_video']]



x_test1 = test_features1

y_test1 = np.ravel(test_labels1)
model1 = xgb.XGBClassifier(

    tree_method = 'gpu_hist'  # THE MAGICAL PARAMETER THAT INTEGRATES KAGGLE'S GPU ACCELERATED KERNEL

)

%time model1.fit(x_train1, y_train1) # Fitting the data into the model
# model1.save_model('model1.model')
%time y_pred1 = model1.predict(x_test1)

accuracy1 = accuracy_score(y_test1, y_pred1)

print("Model 1 Accuracy: %.2f%%" % (accuracy1 * 100.0))
'''model2 = xgb.XGBClassifier()

model2.fit(x_train2, y_train2)

y_pred2 = model2.predict(x_test2)

accuracy2 = accuracy_score(y_test2, predictions2)

print("Model 2 Accuracy: %.2f%%" % (accuracy2 * 100.0))'''
# model2_update = 
'''y_pred2_update = model2_update.predict(x_test2)

accuracy2 = accuracy_score(y_test2, predictions2)

print("Model 2 Accuracy: %.2f%%" % (accuracy2 * 100.0))'''
'''model_loaded = xgb.XGBClassifier()

booster = xgb.Booster()

booster.load_model('../input/mooc-final/model1.model')

model_loaded._Booster = booster



%time y_pred1 = model_loaded.predict(x_test1) '''