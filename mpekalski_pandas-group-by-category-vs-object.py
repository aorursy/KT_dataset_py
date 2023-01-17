import numpy as np

import pandas as pd

import datetime

from catboost import CatBoostClassifier

from time import time

from collections import Counter

from scipy import stats

import os
train = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv', nrows=2000)
# encode title

# make a list with all the unique 'titles' from the train and test set

list_of_user_activities = list(set(train['title'].value_counts().index))

# make a list with all the unique 'event_code' from the train and test set

list_of_event_code = list(set(train['event_code'].value_counts().index))

# create a dictionary numerating the titles

activities_map = dict(zip(list_of_user_activities, np.arange(len(list_of_user_activities))))

activities_labels = dict(zip(np.arange(len(list_of_user_activities)), list_of_user_activities))



# replace the text titles withing the number titles from the dict

train['title'] = train['title'].map(activities_map)

# this is the function that convert the raw data into processed features

def get_data(user_sample, test_set=False):

    # itarates through each session of one instalation_id

    print("="*80)

    for i, (game_session_id, session) in enumerate(user_sample.groupby('game_session', sort=False)):

        # i = game_session_id

        # session is a DataFrame that contain only one game_session

        print(i,"\n",'type' in session, 'title' in session)

        # get some sessions information

        session_type = session['type'].iloc[0]

        session_title = session['title'].iloc[0]        

        if int(i)>20:

            break

    return None
# here the get_data function is applyed to each installation_id and added to the compile_data list

compiled_data = []

# tqdm is the library that draws the status bar below

for i, (ins_id, user_sample) in enumerate(train.groupby('installation_id', sort=False)):

    # user_sample is a DataFrame that contains only one installation_id

    print(i,"\n",ins_id,"\n", user_sample.shape)

    get_data(user_sample)

    break
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ')
train = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv', parse_dates=['timestamp']

                         , date_parser=dateparse

                         , infer_datetime_format=False  

                         , dtype={'event_count':np.int16, 'event_code':np.int16, 'game_time':np.int32,

                                  'event_data':object,

                                  'game_session':'category',

                                  'event_id':'category',

                                  'installation_id':'category',

                                  'title':'category',

                                  'type':'category',      

                                  'world':'category'                               

                                 }, engine='c', nrows=2000)

               
# here the get_data function is applyed to each installation_id and added to the compile_data list

compiled_data = []

# tqdm is the library that draws the status bar below

for i, (ins_id, user_sample) in enumerate(train.groupby('installation_id', sort=False)):

    # user_sample is a DataFrame that contains only one installation_id

    print(i,"\n",ins_id,"\n", user_sample.shape)

    get_data(user_sample)

    break
# this is the function that convert the raw data into processed features

def get_data(user_sample, test_set=False):

    # itarates through each session of one instalation_id

    print("="*80)

    for i, (game_session_id, session) in enumerate(user_sample.groupby('game_session', sort=False, observed=True)):

        # i = game_session_id

        # session is a DataFrame that contain only one game_session

        print(i,"\n",'type' in session, 'title' in session)

        # get some sessions information

        session_type = session['type'].iloc[0]

        session_title = session['title'].iloc[0]        

        if int(i)>20:

            break

    return None



# here the get_data function is applyed to each installation_id and added to the compile_data list

compiled_data = []

# tqdm is the library that draws the status bar below

for i, (ins_id, user_sample) in enumerate(train.groupby('installation_id', sort=False, observed=True)):

    # user_sample is a DataFrame that contains only one installation_id

    print(i,"\n",ins_id,"\n", user_sample.shape)

    get_data(user_sample)

    break
train = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv', parse_dates=['timestamp']

                         , date_parser=dateparse

                         , infer_datetime_format=False  

                         , dtype={'event_count':np.int16, 'event_code':np.int16, 'game_time':np.int32,

                                  'event_data':object,

                                  'game_session':'category',

                                  'event_id':'category',

                                  'installation_id':object,

                                  'title':'category',

                                  'type':'category',      

                                  'world':'category'                               

                                 }, engine='c', nrows=2000)

               
# here the get_data function is applyed to each installation_id and added to the compile_data list

compiled_data = []

# tqdm is the library that draws the status bar below

for i, (ins_id, user_sample) in enumerate(train.groupby('installation_id', sort=False)):

    # user_sample is a DataFrame that contains only one installation_id

    print(i,"\n",ins_id,"\n", user_sample.shape)

    get_data(user_sample)

    break