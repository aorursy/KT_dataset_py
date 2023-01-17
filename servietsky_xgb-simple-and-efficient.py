import numpy as np

import pandas as pd

import os

import copy

import matplotlib.pyplot as plt

%matplotlib inline

from tqdm import tqdm_notebook

from sklearn.preprocessing import StandardScaler

from sklearn.svm import NuSVR, SVR

from sklearn.metrics import mean_absolute_error

pd.options.display.precision = 15

from collections import defaultdict

import lightgbm as lgb

import xgboost as xgb

import catboost as cat

import time

from collections import Counter

import datetime

from catboost import CatBoostRegressor

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold, GridSearchCV, train_test_split, TimeSeriesSplit, RepeatedStratifiedKFold

from sklearn import metrics

from sklearn.metrics import classification_report, confusion_matrix

from sklearn import linear_model

import gc

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

from bayes_opt import BayesianOptimization

import eli5

import shap

from IPython.display import HTML

import json

import altair as alt

from category_encoders.ordinal import OrdinalEncoder

import networkx as nx

import matplotlib.pyplot as plt

%matplotlib inline

from typing import List



import os

import time

import datetime

import json

import gc

from numba import jit



import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

from tqdm import tqdm_notebook



import lightgbm as lgb

import xgboost as xgb

from catboost import CatBoostRegressor, CatBoostClassifier

from sklearn import metrics

from typing import Any

from itertools import product

pd.set_option('max_rows', 500)

import re

from tqdm import tqdm

from joblib import Parallel, delayed
def read_data():

    print('Reading train.csv file....')

    train = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')

    print('Training.csv file have {} rows and {} columns'.format(train.shape[0], train.shape[1]))



    print('Reading test.csv file....')

    test = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')

    print('Test.csv file have {} rows and {} columns'.format(test.shape[0], test.shape[1]))



    print('Reading train_labels.csv file....')

    train_labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')

    print('Train_labels.csv file have {} rows and {} columns'.format(train_labels.shape[0], train_labels.shape[1]))



    print('Reading specs.csv file....')

    specs = pd.read_csv('/kaggle/input/data-science-bowl-2019/specs.csv')

    print('Specs.csv file have {} rows and {} columns'.format(specs.shape[0], specs.shape[1]))



    print('Reading sample_submission.csv file....')

    sample_submission = pd.read_csv('/kaggle/input/data-science-bowl-2019/sample_submission.csv')

    print('Sample_submission.csv file have {} rows and {} columns'.format(sample_submission.shape[0], sample_submission.shape[1]))

    return train, test, train_labels, specs, sample_submission



def encode_title(train, test, train_labels):

    # encode title

    train['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), train['title'], train['event_code']))

    test['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), test['title'], test['event_code']))

    all_title_event_code = list(set(train["title_event_code"].unique()).union(test["title_event_code"].unique()))

    # make a list with all the unique 'titles' from the train and test set

    list_of_user_activities = list(set(train['title'].unique()).union(set(test['title'].unique())))

    # make a list with all the unique 'event_code' from the train and test set

    list_of_event_code = list(set(train['event_code'].unique()).union(set(test['event_code'].unique())))

    list_of_event_id = list(set(train['event_id'].unique()).union(set(test['event_id'].unique())))

    # make a list with all the unique worlds from the train and test set

    list_of_worlds = list(set(train['world'].unique()).union(set(test['world'].unique())))

    # create a dictionary numerating the titles

    activities_map = dict(zip(list_of_user_activities, np.arange(len(list_of_user_activities))))

    activities_labels = dict(zip(np.arange(len(list_of_user_activities)), list_of_user_activities))

    activities_world = dict(zip(list_of_worlds, np.arange(len(list_of_worlds))))

    assess_titles = list(set(train[train['type'] == 'Assessment']['title'].value_counts().index).union(set(test[test['type'] == 'Assessment']['title'].value_counts().index)))

    # replace the text titles with the number titles from the dict

    train['title'] = train['title'].map(activities_map)

    test['title'] = test['title'].map(activities_map)

    train['world'] = train['world'].map(activities_world)

    test['world'] = test['world'].map(activities_world)

    train_labels['title'] = train_labels['title'].map(activities_map)

    win_code = dict(zip(activities_map.values(), (4100*np.ones(len(activities_map))).astype('int')))

    # then, it set one element, the 'Bird Measurer (Assessment)' as 4110, 10 more than the rest

    win_code[activities_map['Bird Measurer (Assessment)']] = 4110

    # convert text into datetime

    train['timestamp'] = pd.to_datetime(train['timestamp'])

    test['timestamp'] = pd.to_datetime(test['timestamp'])

    

    

    return train, test, train_labels, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code



def get_data(user_sample, test_set=False):

    '''

    The user_sample is a DataFrame from train or test where the only one 

    installation_id is filtered

    And the test_set parameter is related with the labels processing, that is only requered

    if test_set=False

    '''

    # Constants and parameters declaration

    last_activity = 0

    

    user_activities_count = {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}

    

    # new features: time spent in each activity

    last_session_time_sec = 0

    accuracy_groups = {0:0, 1:0, 2:0, 3:0}

    all_assessments = []

    accumulated_accuracy_group = 0

    accumulated_accuracy = 0

    accumulated_correct_attempts = 0 

    accumulated_uncorrect_attempts = 0

    accumulated_actions = 0

    counter = 0

    time_first_activity = float(user_sample['timestamp'].values[0])

    durations = []

    last_accuracy_title = {'acc_' + title: -1 for title in assess_titles}

    event_code_count: Dict[str, int] = {ev: 0 for ev in list_of_event_code}

    event_id_count: Dict[str, int] = {eve: 0 for eve in list_of_event_id}

    title_count: Dict[str, int] = {eve: 0 for eve in activities_labels.values()} 

    title_event_code_count: Dict[str, int] = {t_eve: 0 for t_eve in all_title_event_code}

    

    # itarates through each session of one instalation_id

    for i, session in user_sample.groupby('game_session', sort=False):

        # i = game_session_id

        # session is a DataFrame that contain only one game_session

        

        # get some sessions information

        session_type = session['type'].iloc[0]

        session_title = session['title'].iloc[0]

        session_title_text = activities_labels[session_title]

                    

            

        # for each assessment, and only this kind off session, the features below are processed

        # and a register are generated

        if (session_type == 'Assessment') & (test_set or len(session)>1):

            # search for event_code 4100, that represents the assessments trial

            all_attempts = session.query(f'event_code == {win_code[session_title]}')

            # then, check the numbers of wins and the number of losses

            true_attempts = all_attempts['event_data'].str.contains('true').sum()

            false_attempts = all_attempts['event_data'].str.contains('false').sum()

            # copy a dict to use as feature template, it's initialized with some itens: 

            # {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}

            features = user_activities_count.copy()

            features.update(last_accuracy_title.copy())

            features.update(event_code_count.copy())

            features.update(event_id_count.copy())

            features.update(title_count.copy())

            features.update(title_event_code_count.copy())

            features.update(last_accuracy_title.copy())

            

            # get installation_id for aggregated features

            features['installation_id'] = session['installation_id'].iloc[-1]

            # add title as feature, remembering that title represents the name of the game

            features['session_title'] = session['title'].iloc[0]

            # the 4 lines below add the feature of the history of the trials of this player

            # this is based on the all time attempts so far, at the moment of this assessment

            features['accumulated_correct_attempts'] = accumulated_correct_attempts

            features['accumulated_uncorrect_attempts'] = accumulated_uncorrect_attempts

            accumulated_correct_attempts += true_attempts 

            accumulated_uncorrect_attempts += false_attempts

            # the time spent in the app so far

            if durations == []:

                features['duration_mean'] = 0

            else:

                features['duration_mean'] = np.mean(durations)

            durations.append((session.iloc[-1, 2] - session.iloc[0, 2] ).seconds)

            # the accurace is the all time wins divided by the all time attempts

            features['accumulated_accuracy'] = accumulated_accuracy/counter if counter > 0 else 0

            accuracy = true_attempts/(true_attempts+false_attempts) if (true_attempts+false_attempts) != 0 else 0

            accumulated_accuracy += accuracy

            last_accuracy_title['acc_' + session_title_text] = accuracy

            # a feature of the current accuracy categorized

            # it is a counter of how many times this player was in each accuracy group

            if accuracy == 0:

                features['accuracy_group'] = 0

            elif accuracy == 1:

                features['accuracy_group'] = 3

            elif accuracy == 0.5:

                features['accuracy_group'] = 2

            else:

                features['accuracy_group'] = 1

            features.update(accuracy_groups)

            accuracy_groups[features['accuracy_group']] += 1

            # mean of the all accuracy groups of this player

            features['accumulated_accuracy_group'] = accumulated_accuracy_group/counter if counter > 0 else 0

            accumulated_accuracy_group += features['accuracy_group']

            # how many actions the player has done so far, it is initialized as 0 and updated some lines below

            features['accumulated_actions'] = accumulated_actions

            

            # there are some conditions to allow this features to be inserted in the datasets

            # if it's a test set, all sessions belong to the final dataset

            # it it's a train, needs to be passed throught this clausule: session.query(f'event_code == {win_code[session_title]}')

            # that means, must exist an event_code 4100 or 4110

            if test_set:

                all_assessments.append(features)

            elif true_attempts+false_attempts > 0:

                all_assessments.append(features)

                

            counter += 1

        

        # this piece counts how many actions was made in each event_code so far

        def update_counters(counter: dict, col: str):

                num_of_session_count = Counter(session[col])

                for k in num_of_session_count.keys():

                    x = k

                    if col == 'title':

                        x = activities_labels[k]

                    counter[x] += num_of_session_count[k]

                return counter

            

        event_code_count = update_counters(event_code_count, "event_code")

        event_id_count = update_counters(event_id_count, "event_id")

        title_count = update_counters(title_count, 'title')

        title_event_code_count = update_counters(title_event_code_count, 'title_event_code')



        # counts how many actions the player has done so far, used in the feature of the same name

        accumulated_actions += len(session)

        if last_activity != session_type:

            user_activities_count[session_type] += 1

            last_activitiy = session_type 

                        

    # if it't the test_set, only the last assessment must be predicted, the previous are scraped

    if test_set:

        return all_assessments[-1]

    # in the train_set, all assessments goes to the dataset

    return all_assessments



def get_train_and_test(train, test):

    compiled_train = []

    compiled_test = []

    for i, (ins_id, user_sample) in tqdm(enumerate(train.groupby('installation_id', sort = False)), total = 17000):

        compiled_train += get_data(user_sample)

    for ins_id, user_sample in tqdm(test.groupby('installation_id', sort = False), total = 1000):

        test_data = get_data(user_sample, test_set = True)

        compiled_test.append(test_data)

    reduce_train = pd.DataFrame(compiled_train)

    reduce_test = pd.DataFrame(compiled_test)

    categoricals = ['session_title']

    return reduce_train, reduce_test, categoricals



# read data

train, test, train_labels, specs, sample_submission = read_data()

# get usefull dict with maping encode

train, test, train_labels, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code = encode_title(train, test, train_labels)

# tranform function to get the train and test set

reduce_train, reduce_test, categoricals = get_train_and_test(train, test)
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =7, nthread = -1)
y = reduce_train['accuracy_group']

reduce_train.drop([ 'installation_id','accuracy_group'], inplace = True, axis = 1)

# model_xgb.fit(X=reduce_train, y=y)
model_xgb.fit(X=reduce_train, y=y)
reduce_test.drop([ 'installation_id','accuracy_group'], inplace = True, axis = 1)
preds = model_xgb.predict(reduce_test)
df_ajust = pd.DataFrame(preds, columns = ['preds'])



q1 = df_ajust['preds'].quantile(0.0045)

q2 = df_ajust['preds'].quantile(0.99)



df_ajust['preds'] = df_ajust['preds'].apply(lambda x: x if x > q1 else x*0.77)

df_ajust['preds'] = df_ajust['preds'].apply(lambda x: x if x < q2 else x*1.1)



pr1 = df_ajust['preds'].values

pr1[pr1 <= 1.12232214] = 0

pr1[np.where(np.logical_and(pr1 > 1.12232214, pr1 <= 1.73925866))] = 1

pr1[np.where(np.logical_and(pr1 > 1.73925866, pr1 <= 2.22506454))] = 2

pr1[pr1 > 2.22506454] = 3

sample_submission['accuracy_group'] = pr1.astype(int)

sample_submission.to_csv('submission.csv', index=False)