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

train = pd.read_csv('../input/data-science-bowl-2019/train.csv')

train_labels = pd.read_csv('../input/data-science-bowl-2019/train_labels.csv')

test = pd.read_csv('../input/data-science-bowl-2019/test.csv')

specs = pd.read_csv('../input/data-science-bowl-2019/specs.csv')

sample_submission = pd.read_csv('../input/data-science-bowl-2019/sample_submission.csv')
# data cleaning (summary of exploration kernel)



# make sure order is correct/logical (it currently seems to be sorted by game_time. Some events occur simultaneously, and such rows are randomly sorted so some of the event_counts are out of order). With below sorting, events are in chronological order by installation_id

train = train.sort_values(["installation_id","timestamp","event_count"]).reset_index()



# remove training examples who didn't take an assessment (~8mil remaining)

keep_id = train[train.type == "Assessment"][['installation_id']].drop_duplicates()

train = pd.merge(train, keep_id, on="installation_id", how="inner")



# remove rows that we have no accuracy information for (~7mil remaining)

train = train[train.installation_id.isin(train_labels.installation_id.unique())]



# create accurate indicator column of assessment success

train["successes"] = pd.np.where((train.type == "Assessment") & ((train.event_code == 4100) | (train.event_code == 4110)), # consider assessment outcome event_codes (4100 and 4110)

                             pd.np.where(((train.event_data.str.contains("\"correct\":true,\"caterpillars\"")) & (train.title == "Bird Measurer (Assessment)")) | ((train.event_data.str.contains("\"correct\":true")) & (train.event_code == 4100) & (train.title != "Bird Measurer (Assessment)")),"Success",# if a successful stage 1 Bird Measurer (event_code 4110) OR event_code 4100 in any other assessment (provided that the event_data has a 'correct' indicator), assessment was a success and counts towards accuracy measures

                                         pd.np.where(((train.event_data.str.contains("\"correct\":true,\"hats\"")) & (train.title == "Bird Measurer (Assessment)")), "Success (not measured)", # if a successful stage 2 Bird Measurer (event_code = 4100), assessment was a success but doesn't count towards accuracy measures 

                                                     pd.np.where(((train.event_data.str.contains("\"correct\":false,\"hats\"")) & (train.title == "Bird Measurer (Assessment)")),"Failure (not measured)", # if a failed stage 2 Bird Measurer (event_code = 4100), assessment was a failure but doesn't count towards accuracy measures

                                                                 "Failure")) # all remaining 4110 event_codes are failures

                                        ),

                                  "No test", # if not an assessment, straightforward) 

                       )
# Primary information



# we want new columns on the training set for:

# total cumulative time spent in each world

# total cumulative events in each world

# total cumulative tries on each assessment

# total cumulative passes on each assessment (all from the same for loop)
train.head()
pd.crosstab(train[train.type == "Assessment"].title,train[train.type == "Assessment"].world)
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

# Finally, identifier that a row is indeed an assessment

assessments = []

ass_title = []





# create data frame of unique users, and keep track of rows for final merging

unique_users = train["installation_id"].drop_duplicates()

user_list = []

# keep track of session id, for merging

session_list = []

# keep track of timestamp, for further analysis

session_time = []

# installation id lifetime

ID_lifetime = []



for user in tqdm(unique_users):

#for user in tqdm(unique_users.head()):



    train_sub = train[train.installation_id == user]

    # default count increases of 0 for each user

    TTC_time_add = 0

    MP_time_add = 0

    CC_time_add = 0

    TTC_events_add = 0

    MP_events_add = 0

    CC_events_add = 0

    BM_tries_add = 0

    CB_tries_add = 0

    CF_tries_add = 0

    CS_tries_add = 0

    MS_tries_add = 0

    BM_passes_add = 0

    CB_passes_add = 0

    CF_passes_add = 0

    CS_passes_add = 0

    MS_passes_add = 0

    ID_lifetime_add = 0



    for index in train_sub.index:

        user_list.append(user)

        if(train_sub.event_count[index] == 1):

            ID_lifetime_add += 0

            time = 0

        else: 

            ID_lifetime_add += train_sub.game_time[index] - time # only add new time in each session

            time = train_sub.game_time[index]

        session_list.append(train_sub.game_session[index])

        session_time.append(train_sub.timestamp[index])

        if(train_sub.world[index] == "TREETOPCITY"):

            if(train_sub.event_count[index] == 1):

                TTC_time_add += 0

                TTC_session_time = 0

            else: 

                TTC_time_add += train_sub.game_time[index] - TTC_session_time # only want to add new time in each session

                TTC_session_time = train_sub.game_time[index] # update cumulative session time

            TTC_events_add += 1

            # edit this code if we don't want to consider stage 2 passes as normal assessments

            if((train_sub.title[index] == "Bird Measurer (Assessment)") & ((train_sub.successes[index] == "Success") | (train_sub.successes[index] == "Success (not measured)"))):

                BM_tries_add += 1

                BM_passes_add += 1

            elif((train_sub.title[index] == "Bird Measurer (Assessment)") & ((train_sub.successes[index] == "Failure") | (train_sub.successes[index] == "Failure (not measured)"))):

                BM_tries_add += 1

            elif((train_sub.title[index] == "Mushroom Sorter (Assessment)") & (train_sub.successes[index] == "Success")):

                MS_tries_add += 1

                MS_passes_add += 1

            elif((train_sub.title[index] == "Mushroom Sorter (Assessment)") & (train_sub.successes[index] == "Failure")):

                MS_tries_add += 1



        if(train_sub.world[index] == "MAGMAPEAK"):

            if(train_sub.event_count[index] == 1):

                MP_time_add += 0

                MP_session_time = 0

            else: 

                MP_time_add += train_sub.game_time[index] - MP_session_time # only want to add new time in each session

                MP_session_time = train_sub.game_time[index] # update cumulative session time

            MP_events_add += 1

            if((train_sub.title[index] == "Cauldron Filler (Assessment)") & (train_sub.successes[index] == "Success")):

                CF_tries_add += 1

                CF_passes_add += 1

            elif((train_sub.title[index] == "Cauldron Filler (Assessment)") & (train_sub.successes[index] == "Failure")):

                CF_tries_add += 1



        if(train_sub.world[index] == "CRYSTALCAVES"):

            if(train_sub.event_count[index] == 1):

                CC_time_add += 0

                CC_session_time = 0

            else: 

                CC_time_add += train_sub.game_time[index] - CC_session_time # only want to add new time in each session

                CC_session_time = train_sub.game_time[index] # update cumulative session time

            CC_events_add += 1

            if((train_sub.title[index] == "Cart Balancer (Assessment)") & (train_sub.successes[index] == "Success")):

                CB_tries_add += 1

                CB_passes_add += 1

            elif((train_sub.title[index] == "Cart Balancer (Assessment)") & (train_sub.successes[index] == "Failure")):

                CB_tries_add += 1

            elif((train_sub.title[index] == "Chest Sorter (Assessment)") & (train_sub.successes[index] == "Success")):

                CS_tries_add += 1

                CS_passes_add += 1

            elif((train_sub.title[index] == "Chest Sorter (Assessment)") & (train_sub.successes[index] == "Failure")):

                CS_tries_add += 1

        # append assessment indicator

        assessments.append(train_sub.successes[index])

        if((train_sub.successes[index] == 'Success') | (train_sub.successes[index] == 'Failure')):

            ass_title.append(train_sub.title[index])

        else:

            ass_title.append('No test')



        ID_lifetime.append(ID_lifetime_add)

        TTC_time.append(TTC_time_add)

        MP_time.append(MP_time_add)

        CC_time.append(CC_time_add)

        TTC_events.append(TTC_events_add)

        MP_events.append(MP_events_add)

        CC_events.append(CC_events_add)

        BM_tries.append(BM_tries_add)

        CB_tries.append(CB_tries_add)

        CF_tries.append(CF_tries_add)

        CS_tries.append(CS_tries_add)

        MS_tries.append(MS_tries_add)

        BM_passes.append(BM_passes_add)

        CB_passes.append(CB_passes_add)

        CF_passes.append(CF_passes_add)

        CS_passes.append(CS_passes_add)

        MS_passes.append(MS_passes_add)

        

# merge all columns together
new_variables = pd.DataFrame({'user': user_list,

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

             'assessment': assessments,

             'title': ass_title})



# for training, we want a view only of measurable assessments



new_variables = new_variables[(new_variables.assessment == 'Success') | (new_variables.assessment == 'Failure') | (new_variables.assessment == 'Success (not measured)') | (new_variables.assessment == 'Failure (not measured)')]
new_variables



# merge on assessment-level variables
# save clean data



new_variables.to_csv("cleaned_variables.csv",index = False)