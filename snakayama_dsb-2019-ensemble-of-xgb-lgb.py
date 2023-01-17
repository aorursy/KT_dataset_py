#result

#ver1:LGBのみ、lgb cv mean QWK score : 0.5208321783509067、LB：0.536(+0.015)

#ver2:0.8LGB+0.2XGB

#lgb cv mean QWK score : 0.5076390431140777、xgb cv mean QWK score : 0.49945107302527625

#LB：0.534

#ver4:LGB+XGB(LidgeでEnsemble)



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import warnings

import lightgbm as lgb

import xgboost as xgb

import matplotlib.pyplot as plt

import gc

import seaborn as sns

import scipy as sp

import multiprocessing

import scipy as sp

import time

import random

import json



from multiprocessing import Lock, Process, Queue, current_process

from math import sqrt

from numba import jit

from functools import partial

from tqdm import tqdm as tqdm

from collections import Counter

from sklearn.metrics import cohen_kappa_score, mean_squared_error

from sklearn.metrics import confusion_matrix as sk_cmatrix

from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold, GridSearchCV, train_test_split, TimeSeriesSplit, RepeatedStratifiedKFold

from catboost import CatBoostRegressor

np.random.seed(724)

warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', 1000)

pd.set_option('max_rows', 500)
def read_data():

    start = time.time()

    print("Start read data")



    train = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')

    #train = pd.read_csv('../input/data-science-bowl-2019/train.csv', nrows=1200000)

    print('Training.csv file have {} rows and {} columns'.format(train.shape[0], train.shape[1]))



    print('Reading test.csv file....')

    test = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')

    #test = pd.read_csv('../input/data-science-bowl-2019/test.csv', nrows=30000)



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



    print("read data done, time - ", time.time() - start)

    return train, test, train_labels, specs, sample_submission

def encode_title(train, test, train_labels):

    start = time.time()



    print("Start encoding data")

    # encode title

    train['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), train['title'], train['event_code']))

    test['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), test['title'], test['event_code']))

    all_title_event_code = list(set(train["title_event_code"].unique()).union(test["title_event_code"].unique()))



    train['type_world'] = list(map(lambda x, y: str(x) + '_' + str(y), train['type'], train['world']))

    test['type_world'] = list(map(lambda x, y: str(x) + '_' + str(y), test['type'], test['world']))

    all_type_world = list(set(train["type_world"].unique()).union(test["type_world"].unique()))



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

    assess_titles = list(set(train[train['type'] == 'Assessment']['title'].value_counts().index).union(

        set(test[test['type'] == 'Assessment']['title'].value_counts().index)))

    # replace the text titles with the number titles from the dict

    train['title'] = train['title'].map(activities_map)

    test['title'] = test['title'].map(activities_map)

    train['world'] = train['world'].map(activities_world)

    test['world'] = test['world'].map(activities_world)

    train_labels['title'] = train_labels['title'].map(activities_map)

    win_code = dict(zip(activities_map.values(), (4100 * np.ones(len(activities_map))).astype('int')))

    # then, it set one element, the 'Bird Measurer (Assessment)' as 4110, 10 more than the rest

    win_code[activities_map['Bird Measurer (Assessment)']] = 4110

    # convert text into datetime

    train['timestamp'] = pd.to_datetime(train['timestamp'])

    test['timestamp'] = pd.to_datetime(test['timestamp'])

    print("End encoding data, time - ", time.time() - start)





    event_data = {}

    event_data["train_labels"] = train_labels

    event_data["win_code"] = win_code

    event_data["list_of_user_activities"] = list_of_user_activities

    event_data["list_of_event_code"] = list_of_event_code

    event_data["activities_labels"] = activities_labels

    event_data["assess_titles"] = assess_titles

    event_data["list_of_event_id"] = list_of_event_id

    event_data["all_title_event_code"] = all_title_event_code

    event_data["activities_map"] = activities_map

    event_data["all_type_world"] = all_type_world



    return train, test, event_data

def get_all_features(feature_dict, ac_data):

    if len(ac_data['durations']) > 0:

        feature_dict['installation_duration_mean'] = np.mean(ac_data['durations'])

        feature_dict['installation_duration_sum'] = np.sum(ac_data['durations'])

    else:

        feature_dict['installation_duration_mean'] = 0

        feature_dict['installation_duration_sum'] = 0



    return feature_dict
def get_data(user_sample, event_data, test_set):

    '''

    The user_sample is a DataFrame from train or test where the only one

    installation_id is filtered

    And the test_set parameter is related with the labels processing, that is only requered

    if test_set=False

    '''

    # Constants and parameters declaration

    last_assesment = {}



    last_activity = 0



    user_activities_count = {'Clip': 0, 'Activity': 0, 'Assessment': 0, 'Game': 0}



    assess_4020_acc_dict = {'Cauldron Filler (Assessment)_4020_accuracy': 0,

                            'Mushroom Sorter (Assessment)_4020_accuracy': 0,

                            'Bird Measurer (Assessment)_4020_accuracy': 0,

                            'Chest Sorter (Assessment)_4020_accuracy': 0}



    game_time_dict = {'Clip_gametime': 0, 'Game_gametime': 0,

                      'Activity_gametime': 0, 'Assessment_gametime': 0}



    last_session_time_sec = 0

    accuracy_groups = {0: 0, 1: 0, 2: 0, 3: 0}

    all_assessments = []

    accumulated_accuracy_group = 0

    accumulated_accuracy = 0

    accumulated_correct_attempts = 0

    accumulated_uncorrect_attempts = 0

    accumulated_actions = 0



    # Newly added features

    accumulated_game_miss = 0

    Cauldron_Filler_4025 = 0

    mean_game_round = 0

    mean_game_duration = 0

    mean_game_level = 0

    Assessment_mean_event_count = 0

    Game_mean_event_count = 0

    Activity_mean_event_count = 0

    chest_assessment_uncorrect_sum = 0



    counter = 0

    time_first_activity = float(user_sample['timestamp'].values[0])

    durations = []

    durations_game = []

    durations_activity = []

    last_accuracy_title = {'acc_' + title: -1 for title in event_data["assess_titles"]}

    last_game_time_title = {'lgt_' + title: 0 for title in event_data["assess_titles"]}

    ac_game_time_title = {'agt_' + title: 0 for title in event_data["assess_titles"]}

    ac_true_attempts_title = {'ata_' + title: 0 for title in event_data["assess_titles"]}

    ac_false_attempts_title = {'afa_' + title: 0 for title in event_data["assess_titles"]}

    event_code_count: dict[str, int] = {ev: 0 for ev in event_data["list_of_event_code"]}

    event_code_proc_count = {str(ev) + "_proc" : 0. for ev in event_data["list_of_event_code"]}

    event_id_count: dict[str, int] = {eve: 0 for eve in event_data["list_of_event_id"]}

    title_count: dict[str, int] = {eve: 0 for eve in event_data["activities_labels"].values()}

    title_event_code_count: dict[str, int] = {t_eve: 0 for t_eve in event_data["all_title_event_code"]}

    type_world_count: dict[str, int] = {w_eve: 0 for w_eve in event_data["all_type_world"]}

    session_count = 0



    # itarates through each session of one instalation_id

    for i, session in user_sample.groupby('game_session', sort=False):

        # i = game_session_id

        # session is a DataFrame that contain only one game_session

        # get some sessions information

        session_type = session['type'].iloc[0]

        session_title = session['title'].iloc[0]

        session_title_text = event_data["activities_labels"][session_title]



        if session_type == "Activity":

            Activity_mean_event_count = (Activity_mean_event_count + session['event_count'].iloc[-1]) / 2.0



        if session_type == "Game":

            Game_mean_event_count = (Game_mean_event_count + session['event_count'].iloc[-1]) / 2.0



            game_s = session[session.event_code == 2030]

            misses_cnt = cnt_miss(game_s)

            accumulated_game_miss += misses_cnt



            try:

                game_round = json.loads(session['event_data'].iloc[-1])["round"]

                mean_game_round = (mean_game_round + game_round) / 2.0

            except:

                pass



            try:

                game_duration = json.loads(session['event_data'].iloc[-1])["duration"]

                mean_game_duration = (mean_game_duration + game_duration) / 2.0

            except:

                pass



            try:

                game_level = json.loads(session['event_data'].iloc[-1])["level"]

                mean_game_level = (mean_game_level + game_level) / 2.0

            except:

                pass



        # for each assessment, and only this kind off session, the features below are processed

        # and a register are generated

        if (session_type == 'Assessment') & (test_set or len(session) > 1):

            # search for event_code 4100, that represents the assessments trial

            all_attempts = session.query(f'event_code == {event_data["win_code"][session_title]}')

            # then, check the numbers of wins and the number of losses

            true_attempts = all_attempts['event_data'].str.contains('true').sum()

            false_attempts = all_attempts['event_data'].str.contains('false').sum()

            # copy a dict to use as feature template, it's initialized with some itens:

            # {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}

            features = user_activities_count.copy()

            features.update(last_accuracy_title.copy())

            features.update(event_code_count.copy())

            features.update(title_count.copy())

            features.update(game_time_dict.copy())

            features.update(event_id_count.copy())

            features.update(title_event_code_count.copy())

            features.update(assess_4020_acc_dict.copy())

            features.update(type_world_count.copy())

            features.update(last_game_time_title.copy())

            features.update(ac_game_time_title.copy())

            features.update(ac_true_attempts_title.copy())

            features.update(ac_false_attempts_title.copy())



            features.update(event_code_proc_count.copy())

            features['installation_session_count'] = session_count

            features['accumulated_game_miss'] = accumulated_game_miss

            features['mean_game_round'] = mean_game_round

            features['mean_game_duration'] = mean_game_duration

            features['mean_game_level'] = mean_game_level

            features['Assessment_mean_event_count'] = Assessment_mean_event_count

            features['Game_mean_event_count'] = Game_mean_event_count

            features['Activity_mean_event_count'] = Activity_mean_event_count

            features['chest_assessment_uncorrect_sum'] = chest_assessment_uncorrect_sum



            variety_features = [('var_event_code', event_code_count),

                                ('var_event_id', event_id_count),

                                ('var_title', title_count),

                                ('var_title_event_code', title_event_code_count),

                                ('var_type_world', type_world_count)]



            for name, dict_counts in variety_features:

                arr = np.array(list(dict_counts.values()))

                features[name] = np.count_nonzero(arr)



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



            # ----------------------------------------------

            ac_true_attempts_title['ata_' + session_title_text] += true_attempts

            ac_false_attempts_title['afa_' + session_title_text] += false_attempts



            last_game_time_title['lgt_' + session_title_text] = session['game_time'].iloc[-1]

            ac_game_time_title['agt_' + session_title_text] += session['game_time'].iloc[-1]

            # ----------------------------------------------



            # the time spent in the app so far

            if durations == []:

                features['duration_mean'] = 0

                features['duration_std'] = 0

                features['last_duration'] = 0

                features['duration_max'] = 0

            else:

                features['duration_mean'] = np.mean(durations)

                features['duration_std'] = np.std(durations)

                features['last_duration'] = durations[-1]

                features['duration_max'] = np.max(durations)

            durations.append((session.iloc[-1, 2] - session.iloc[0, 2]).seconds)



            if durations_game == []:

                features['duration_game_mean'] = 0

                features['duration_game_std'] = 0

                features['game_last_duration'] = 0

                features['game_max_duration'] = 0

            else:

                features['duration_game_mean'] = np.mean(durations_game)

                features['duration_game_std'] = np.std(durations_game)

                features['game_last_duration'] = durations_game[-1]

                features['game_max_duration'] = np.max(durations_game)



            if durations_activity == []:

                features['duration_activity_mean'] = 0

                features['duration_activity_std'] = 0

                features['game_activity_duration'] = 0

                features['game_activity_max'] = 0

            else:

                features['duration_activity_mean'] = np.mean(durations_activity)

                features['duration_activity_std'] = np.std(durations_activity)

                features['game_activity_duration'] = durations_activity[-1]

                features['game_activity_max'] = np.max(durations_activity)



            # the accuracy is the all time wins divided by the all time attempts

            features['accumulated_accuracy'] = accumulated_accuracy / counter if counter > 0 else 0

            # --------------------------

            features['Cauldron_Filler_4025'] = Cauldron_Filler_4025 / counter if counter > 0 else 0



            Assess_4025 = session[(session.event_code == 4025) & (session.title == 'Cauldron Filler (Assessment)')]

            true_attempts_ = Assess_4025['event_data'].str.contains('true').sum()

            false_attempts_ = Assess_4025['event_data'].str.contains('false').sum()



            cau_assess_accuracy_ = true_attempts_ / (true_attempts_ + false_attempts_) if (

                                                                                                      true_attempts_ + false_attempts_) != 0 else 0

            Cauldron_Filler_4025 += cau_assess_accuracy_



            chest_assessment_uncorrect_sum += len(session[session.event_id == "df4fe8b6"])



            Assessment_mean_event_count = (Assessment_mean_event_count + session['event_count'].iloc[-1]) / 2.0

            # ----------------------------

            accuracy = true_attempts / (true_attempts + false_attempts) if (true_attempts + false_attempts) != 0 else 0

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

            features['accumulated_accuracy_group'] = accumulated_accuracy_group / counter if counter > 0 else 0

            accumulated_accuracy_group += features['accuracy_group']

            # how many actions the player has done so far, it is initialized as 0 and updated some lines below

            features['accumulated_actions'] = accumulated_actions



            # there are some conditions to allow this features to be inserted in the datasets

            # if it's a test set, all sessions belong to the final dataset

            # it it's a train, needs to be passed throught this clausule: session.query(f'event_code == {win_code[session_title]}')

            # that means, must exist an event_code 4100 or 4110

            if test_set:

                last_assesment = features.copy()



            if true_attempts + false_attempts > 0:

                all_assessments.append(features)



            counter += 1



        if session_type == 'Game':

            durations_game.append((session.iloc[-1, 2] - session.iloc[0, 2]).seconds)



        if session_type == 'Activity':

            durations_activity.append((session.iloc[-1, 2] - session.iloc[0, 2]).seconds)



        session_count += 1



        # this piece counts how many actions was made in each event_code so far

        def update_counters(counter: dict, col: str):

            num_of_session_count = Counter(session[col])

            for k in num_of_session_count.keys():

                x = k

                if col == 'title':

                    x = event_data["activities_labels"][k]

                counter[x] += num_of_session_count[k]

            return counter



        def update_proc(count: dict):

            res = {}

            for k, val in count.items():

                res[str(k) + "_proc"] = (float(val) * 100.0) / accumulated_actions

            return res



        event_code_count = update_counters(event_code_count, "event_code")





        event_id_count = update_counters(event_id_count, "event_id")

        title_count = update_counters(title_count, 'title')

        title_event_code_count = update_counters(title_event_code_count, 'title_event_code')

        type_world_count = update_counters(type_world_count, 'type_world')



        assess_4020_acc_dict = get_4020_acc(session, assess_4020_acc_dict, event_data)

        game_time_dict[session_type + '_gametime'] = (game_time_dict[session_type + '_gametime'] + (

                    session['game_time'].iloc[-1] / 1000.0)) / 2.0



        # counts how many actions the player has done so far, used in the feature of the same name

        accumulated_actions += len(session)

        event_code_proc_count = update_proc(event_code_count)



        if last_activity != session_type:

            user_activities_count[session_type] += 1

            last_activitiy = session_type



            # if it't the test_set, only the last assessment must be predicted, the previous are scraped

    if test_set:

        return last_assesment, all_assessments

    # in the train_set, all assessments goes to the dataset

    return all_assessments



def cnt_miss(df):

    cnt = 0

    for e in range(len(df)):

        x = df['event_data'].iloc[e]

        y = json.loads(x)['misses']

        cnt += y

    return cnt



def get_4020_acc(df, counter_dict, event_data):

    for e in ['Cauldron Filler (Assessment)', 'Bird Measurer (Assessment)',

              'Mushroom Sorter (Assessment)', 'Chest Sorter (Assessment)']:

        Assess_4020 = df[(df.event_code == 4020) & (df.title == event_data["activities_map"][e])]

        true_attempts_ = Assess_4020['event_data'].str.contains('true').sum()

        false_attempts_ = Assess_4020['event_data'].str.contains('false').sum()



        measure_assess_accuracy_ = true_attempts_ / (true_attempts_ + false_attempts_) if (

                                                                                                      true_attempts_ + false_attempts_) != 0 else 0

        counter_dict[e + "_4020_accuracy"] += (counter_dict[e + "_4020_accuracy"] + measure_assess_accuracy_) / 2.0



    return counter_dict



def get_users_data(users_list, return_dict,  event_data, test_set):

    if test_set:

        for user in users_list:

            return_dict.append(get_data(user, event_data, test_set))

    else:

        answer = []

        for user in users_list:

            answer += get_data(user, event_data, test_set)

        return_dict += answer



def get_data_parrallel(users_list, event_data, test_set):

    manager = multiprocessing.Manager()

    return_dict = manager.list()

    threads_number = event_data["process_numbers"]

    data_len = len(users_list)

    processes = []

    cur_start = 0

    cur_stop = 0

    for index in range(threads_number):

        cur_stop += (data_len-1) // threads_number



        if index != (threads_number - 1):

            p = Process(target=get_users_data, args=(users_list[cur_start:cur_stop], return_dict, event_data, test_set))

        else:

            p = Process(target=get_users_data, args=(users_list[cur_start:], return_dict, event_data, test_set))



        processes.append(p)

        cur_start = cur_stop



    for proc in processes:

        proc.start()



    for proc in processes:

        proc.join()



    return list(return_dict)



def get_train_and_test(train, test, event_data):

    start = time.time()

    print("Start get_train_and_test")



    compiled_train = []

    compiled_test = []



    user_train_list = []

    user_test_list = []



    stride_size = event_data["strides"]

    for i, (ins_id, user_sample) in tqdm(enumerate(train.groupby('installation_id', sort=False)), total=17000):

        user_train_list.append(user_sample)

        if (i + 1) % stride_size == 0:

            compiled_train += get_data_parrallel(user_train_list, event_data, False)

            del user_train_list

            user_train_list = []



    if len(user_train_list) > 0:

        compiled_train += get_data_parrallel(user_train_list, event_data, False)

        del user_train_list



    for i, (ins_id, user_sample) in tqdm(enumerate(test.groupby('installation_id', sort=False)), total=1000):

        user_test_list.append(user_sample)

        if (i + 1) % stride_size == 0:

            compiled_test += get_data_parrallel(user_test_list, event_data, True)

            del user_test_list

            user_test_list = []



    if len(user_test_list) > 0:

        compiled_test += get_data_parrallel(user_test_list, event_data, True)

        del user_test_list



    reduce_train = pd.DataFrame(compiled_train)



    reduce_test = [x[0] for x in compiled_test]



    reduce_train_from_test = []

    for i in [x[1] for x in compiled_test]:

        reduce_train_from_test += i



    reduce_test = pd.DataFrame(reduce_test)

    reduce_train_from_test = pd.DataFrame(reduce_train_from_test)

    print("End get_train_and_test, time - ", time.time() - start)

    return reduce_train, reduce_test, reduce_train_from_test



def get_train_and_test_single_proc(train, test, event_data):

    compiled_train = []

    compiled_test = []

    compiled_test_his = []

    for ins_id, user_sample in tqdm(train.groupby('installation_id', sort=False), total=17000):

        compiled_train += get_data(user_sample, event_data, False)

    for ins_id, user_sample in tqdm(test.groupby('installation_id', sort=False), total=1000):

        test_data = get_data(user_sample, event_data, True)

        compiled_test.append(test_data[0])

        compiled_test_his += test_data[1]





    reduce_train = pd.DataFrame(compiled_train)

    reduce_test = pd.DataFrame(compiled_test)

    reduce_test_his = pd.DataFrame(compiled_test_his)



    return reduce_train, reduce_test, reduce_test_his
in_kaggle = True

random.seed(42)

start_program = time.time()



event_data = {}

if in_kaggle:

    event_data["strides"] = 300

    event_data["process_numbers"] = 4

else:

    event_data["strides"] = 300

    event_data["process_numbers"] = 3
# read data

train, test, train_labels, specs, sample_submission = read_data()

# get usefull dict with maping encode

train, test, event_data_update = encode_title(train, test, train_labels)

event_data.update(event_data_update)



#reduce_train, reduce_test, reduce_train_from_test = get_train_and_test_single_proc(train, test, event_data)

reduce_train, reduce_test, reduce_train_from_test = get_train_and_test(train, test, event_data)

dels = [train, test]

del dels
gc.collect()
reduce_train.sort_values("installation_id", axis=0, ascending=True, inplace=True, na_position='last')

reduce_test.sort_values("installation_id", axis=0, ascending=True, inplace=True, na_position='last')

reduce_train = pd.concat([reduce_train, reduce_train_from_test], ignore_index=True)
reduce_train
def stract_hists(feature, train=reduce_train, test=reduce_test, adjust=False, plot=False):

    n_bins = 10

    train_data = train[feature]

    test_data = test[feature]

    if adjust:

        test_data *= train_data.mean() / test_data.mean()

    perc_90 = np.percentile(train_data, 95)

    train_data = np.clip(train_data, 0, perc_90)

    test_data = np.clip(test_data, 0, perc_90)

    train_hist = np.histogram(train_data, bins=n_bins)[0] / len(train_data)

    test_hist = np.histogram(test_data, bins=n_bins)[0] / len(test_data)

    msre = mean_squared_error(train_hist, test_hist)

    if plot:

        print(msre)

        plt.bar(range(n_bins), train_hist, color='blue', alpha=0.5)

        plt.bar(range(n_bins), test_hist, color='red', alpha=0.5)

        plt.show()

    return msre

stract_hists('Magma Peak - Level 1_2000', adjust=False, plot=True)
# call feature engineering function

features = reduce_train.loc[(reduce_train.sum(axis=1) != 0), (reduce_train.sum(axis=0) != 0)].columns # delete useless columns

features = [x for x in features if x not in ['accuracy_group', 'installation_id']]
counter = 0

to_remove = []

for feat_a in features:

    for feat_b in features:

        if feat_a != feat_b and feat_a not in to_remove and feat_b not in to_remove:

            c = np.corrcoef(reduce_train[feat_a], reduce_train[feat_b])[0][1]

            if c > 0.995:

                counter += 1

                to_remove.append(feat_b)

                print('{}: FEAT_A: {} FEAT_B: {} - Correlation: {}'.format(counter, feat_a, feat_b, c))
to_exclude = [] 

ajusted_test = reduce_test.copy()

for feature in tqdm(ajusted_test.columns):

    if feature not in ['accuracy_group', 'installation_id', 'accuracy_group', 'session_title']:

        data = reduce_train[feature]

        train_mean = data.mean()

        data = ajusted_test[feature] 

        test_mean = data.mean()

        try:

            error = stract_hists(feature, adjust=True)

            ajust_factor = train_mean / test_mean

            if ajust_factor > 10 or ajust_factor < 0.1:# or error > 0.01:

                to_exclude.append(feature)

#                 print(feature, train_mean, test_mean, error)

            else:

                ajusted_test[feature] *= ajust_factor

        except:

            to_exclude.append(feature)

#             print(feature, train_mean, test_mean)
features = [x for x in features if x not in (to_exclude + to_remove)]

reduce_train[features].shape
reduce_train.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in reduce_train.columns]

ajusted_test.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in ajusted_test.columns]

features = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in features]
#define_metric

@jit

def qwk(a1, a2):

    """

    Source: https://www.kaggle.com/c/data-science-bowl-2019/discussion/114133#latest-660168



    :param a1:

    :param a2:

    :param max_rat:

    :return:

    """

    max_rat = 3

    a1 = np.asarray(a1, dtype=int)

    a2 = np.asarray(a2, dtype=int)



    hist1 = np.zeros((max_rat + 1,))

    hist2 = np.zeros((max_rat + 1,))



    o = 0

    for k in range(a1.shape[0]):

        i, j = a1[k], a2[k]

        hist1[i] += 1

        hist2[j] += 1

        o += (i - j) * (i - j)



    e = 0

    for i in range(max_rat + 1):

        for j in range(max_rat + 1):

            e += hist1[i] * hist2[j] * (i - j) * (i - j)



    e = e / a1.shape[0]



    return 1 - o / e
class OptimizedRounder(object):

    """

    An optimizer for rounding thresholds

    to maximize Quadratic Weighted Kappa (QWK) score

    # https://www.kaggle.com/naveenasaithambi/optimizedrounder-improved

    """



    def __init__(self):

        self.coef_ = 0



    def _kappa_loss(self, coef, X, y):

        """

        Get loss according to

        using current coefficients



        :param coef: A list of coefficients that will be used for rounding

        :param X: The raw predictions

        :param y: The ground truth labels

        """

        X_p = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels=[0, 1, 2, 3])



        return -qwk(y, X_p)



    def fit(self, X, y):

        """

        Optimize rounding thresholds



        :param X: The raw predictions

        :param y: The ground truth labels

        """

        loss_partial = partial(self._kappa_loss, X=X, y=y)

        initial_coef = [1.10, 1.72, 2.25]

        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead', options={

            'maxiter': 5000})



    def predict(self, X, coef):

        """

        Make predictions with specified thresholds



        :param X: The raw predictions

        :param coef: A list of coefficients that will be used for rounding

        """

        return pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels=[0, 1, 2, 3])



    def coefficients(self):

        """

        Return the optimized coefficients

        """

        return self.coef_['x']
def soft_kappa_obj(y, p):

    y = np.asarray(y)

    p = np.asarray(p.label)

    norm = p.dot(p) + y.dot(y)



    grad = -2 * y / norm + 4 * p * np.dot(y, p) / (norm ** 2)

    hess = 8 * p * y / (norm ** 2) + 4 * np.dot(y, p) / (norm ** 2) - (16 * p ** 2 * np.dot(y, p)) / (norm ** 3)

    return grad, hess



def eval_qwk_lgb_regr_metric(y_pred, true):

    y_true=true.label



    dist = Counter(y_true)

    for k in dist:

        dist[k] /= len(y_true)



    acum = 0

    bound = {}

    for i in range(3):

        acum += dist[i]

        bound[i] = np.percentile(y_pred, acum * 100)



    def classify(x):

        if x <= bound[0]:

            return 0

        elif x <= bound[1]:

            return 1

        elif x <= bound[2]:

            return 2

        else:

            return 3



    y_pred = np.array(list(map(classify, y_pred)))



    return 'cappa', qwk(y_true, y_pred), True



def eval_qwk_lgb_regr(y_true, y_pred):

    """

    Fast cappa eval function for lgb.

    """

    dist = Counter(reduce_train['accuracy_group'])

    for k in dist:

        dist[k] /= len(reduce_train)

#     reduce_train['accuracy_group'].hist()

    

    acum = 0

    bound = {}

    for i in range(3):

        acum += dist[i]

        bound[i] = np.percentile(y_pred, acum * 100)



    def classify(x):

        if x <= bound[0]:

            return 0

        elif x <= bound[1]:

            return 1

        elif x <= bound[2]:

            return 2

        else:

            return 3



    y_pred = np.array(list(map(classify, y_pred))).reshape(y_true.shape)



    return 'cappa', cohen_kappa_score(y_true, y_pred, weights='quadratic'), True





def rmse(actual, predicted):

    return sqrt(mean_squared_error(actual, predicted))
def run_cv_model(train, test, target, model_fn, params={}, eval_fn=None, label='model'):

    kf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

#     kf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2,random_state=2020)

    fold_splits = kf.split(train, target)

    cv_scores = []

    qwk_scores = []

    pred_full_test = 0

    pred_train = np.zeros((train.shape[0], 5))

    all_coefficients = np.zeros((5, 3))

    feature_importance_df = pd.DataFrame()

    i = 1

    ind = []

    for dev_index, val_index in fold_splits:

        print('Started ' + label + ' fold ' + str(i) + '/5')

        if isinstance(train, pd.DataFrame):

            dev_X, val_X = train.iloc[dev_index], train.iloc[val_index]

            dev_y, val_y = target[dev_index], target[val_index]

        else:

            dev_X, val_X = train[dev_index], train[val_index]

            dev_y, val_y = target[dev_index], target[val_index]

        params2 = params.copy()

        #Truncated Valをいれるべきか？

        pred_val_y, pred_test_y, importances, coefficients, qwk = model_fn(dev_X, dev_y, val_X, val_y, test, params2)

        pred_full_test = pred_full_test + pred_test_y

        pred_train[val_index] = pred_val_y

        all_coefficients[i-1, :] = coefficients

        if eval_fn is not None:

            cv_score = eval_fn(val_y, pred_val_y)

            cv_scores.append(cv_score)

            qwk_scores.append(qwk[1])

            print(label + ' cv score {}: RMSE {} QWK {}'.format(i, cv_score, qwk))

        fold_importance_df = pd.DataFrame()

        fold_importance_df['feature'] = train.columns.values

        fold_importance_df['importance'] = importances

        fold_importance_df['fold'] = i

        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)        

        i += 1

    print('{} cv RMSE scores : {}'.format(label, cv_scores))

    print('{} cv mean RMSE score : {}'.format(label, np.mean(cv_scores)))

    print('{} cv QWK scores : {}'.format(label, qwk_scores))

    print('{} cv mean QWK score : {}'.format(label, np.mean(qwk_scores)))

    

    pred_full_test = pred_full_test / 5.0

    results = {'label': label,

               'train': pred_train, 

               'test': pred_full_test,

               'cv': cv_scores, 

               'qwk': qwk_scores,

               'importance': feature_importance_df,

               'coefficients': all_coefficients

              }

    return results
#一旦決め打ち

lgb_params = {'application': 'regression',

          'boosting': 'gbdt',

          'metric': 'rmse',

          'max_depth': 11,

          'learning_rate': 0.01,

          'feature_fraction': 0.8,

          'verbosity': -1,

          'lambda_l1': 1,

          'lambda_l2': 1,

          'data_random_seed': 3,

          'early_stop': 100,

          'verbose_eval': 100,

          'num_rounds': 10000

             }
target = reduce_train['accuracy_group']

categoricals = ['session_title']
def runLGB(train_X, train_y, test_X, test_y, test_X2, params):

    d_train = lgb.Dataset(train_X, label=train_y,categorical_feature=categoricals)

    d_valid = lgb.Dataset(test_X, label=test_y,categorical_feature=categoricals)

    watchlist = [d_train, d_valid]

    num_rounds = params.pop('num_rounds')

    verbose_eval = params.pop('verbose_eval')

    early_stop = None

    if params.get('early_stop'):

        early_stop = params.pop('early_stop')

    model = lgb.train(params,

                      train_set=d_train,

                      num_boost_round=num_rounds,

                      valid_sets=watchlist,

                      verbose_eval=verbose_eval,

                      early_stopping_rounds=early_stop

                     )

    

    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)

    optR = OptimizedRounder()

    coefficients = [0.5, 1.5, 2.5]

    optR.fit(pred_test_y, test_y)

    coefficients = optR.coefficients()

    pred_test_y_k = optR.predict(pred_test_y, coefficients)

    print("Valid Counts = ", Counter(test_y))

    print("Predicted Counts = ", Counter(pred_test_y_k))

    print("Coefficients = ", coefficients)

    qwk = eval_qwk_lgb_regr(test_y, pred_test_y_k)

    print("QWK = ", qwk)

    print('Predict 2/2')

    pred_test_y2 = model.predict(test_X2, num_iteration=model.best_iteration)

    return pred_test_y.reshape(-1, 1), pred_test_y2.reshape(-1, 1), model.feature_importance(), coefficients, qwk
lgb_results = run_cv_model(reduce_train[features], ajusted_test[features], target, runLGB, lgb_params, rmse, 'lgb')
optR = OptimizedRounder()

coefficients_ = np.mean(lgb_results['coefficients'], axis=0)

print(coefficients_)

lgb_train_predictions = [r[0] for r in lgb_results['train']]

lgb_train_predictions = optR.predict(lgb_train_predictions, coefficients_).astype(int)

Counter(lgb_train_predictions)
optR.predict(lgb_train_predictions, coefficients_)
optR = OptimizedRounder()

lgb_test_predictions = [r[0] for r in lgb_results['test']]

lgb_test_predictions = optR.predict(lgb_test_predictions, coefficients_).astype(int)

Counter(lgb_test_predictions)
eval_qwk_lgb_regr(target, lgb_train_predictions)
rmse(target, [r[0] for r in lgb_results['train']])
xgb_params = {

    'objective':'reg:squarederror',

    'colsample_bytree': 0.8,

    'learning_rate': 0.01,

    'max_depth': 10,

    'subsample': 1,

    'min_child_weight':3,

    'gamma':0.25,

    'n_estimators':5000,

    'num_boost_round':5000,

    'early_stopping_rounds':100

}
def runXGB(train_X, train_y, test_X, test_y, test_X2, params):

    d_train = xgb.DMatrix(train_X, label=train_y)

    d_valid = xgb.DMatrix(test_X, label=test_y)

    watchlist = [d_train, d_valid]

    num_rounds = params.pop('num_boost_round')

    verbose_eval = 100

    early_stop = None

    if params.get('early_stopping_rounds'):

        early_stop = params.pop('early_stopping_rounds')

    model = xgb.train(params,

                      d_train,

                      num_boost_round=num_rounds,

                      evals=[(d_train, 'train'), (d_valid, 'val')],

                      verbose_eval=verbose_eval,

                      early_stopping_rounds=early_stop

                     )

    

    pred_test_y = model.predict(xgb.DMatrix(test_X),ntree_limit=model.best_ntree_limit)

    optR = OptimizedRounder()

    coefficients = [0.5, 1.5, 2.5]

    optR.fit(pred_test_y, test_y)

    coefficients = optR.coefficients()

    pred_test_y_k = optR.predict(pred_test_y, coefficients)

    print("Valid Counts = ", Counter(test_y))

    print("Predicted Counts = ", Counter(pred_test_y_k))

    print("Coefficients = ", coefficients)

    qwk = eval_qwk_lgb_regr(test_y, pred_test_y_k)

    print("QWK = ", qwk)

    print('Predict 2/2')

    pred_test_y2 = model.predict(xgb.DMatrix(test_X2),ntree_limit=model.best_ntree_limit)

    return pred_test_y.reshape(-1, 1), pred_test_y2.reshape(-1, 1), features,coefficients, qwk
xgb_results = run_cv_model(reduce_train[features], ajusted_test[features], target, runXGB, xgb_params, rmse, 'xgb')
optR = OptimizedRounder()

coefficients_ = np.mean(xgb_results['coefficients'], axis=0)

print(coefficients_)

xgb_train_predictions = [r[0] for r in xgb_results['train']]

xgb_train_predictions = optR.predict(xgb_train_predictions, coefficients_).astype(int)

Counter(xgb_train_predictions)
optR.predict(xgb_train_predictions, coefficients_)
optR = OptimizedRounder()

xgb_test_predictions = [r[0] for r in xgb_results['test']]

xgb_test_predictions = optR.predict(xgb_test_predictions, coefficients_).astype(int)

Counter(xgb_test_predictions)
eval_qwk_lgb_regr(target, xgb_train_predictions)
rmse(target, [r[0] for r in xgb_results['train']])
cat_params = {

          'depth': 9,

          'eta': 0.05,

          'random_strength': 1.5,

          'one_hot_max_size': 2,

          'reg_lambda': 6,

          'od_type': 'Iter',

          'fold_len_multiplier': 2,

          'bootstrap_type' : "Bayesian",

          'bagging_temperature': 1,

          'random_seed': 217,

          'early_stopping_rounds':100, 

          'num_boost_round': 2500

}
def runCAT(train_X, train_y, test_X, test_y, test_X2, params):

    watchlist = (test_X, test_y)

    verbose_eval = 100

    early_stop = None

    if params.get('early_stopping_rounds'):

        early_stop = params.pop('early_stopping_rounds')

    model = CatBoostRegressor(cat_features=categoricals, **params)

    model.fit(train_X, train_y, eval_set=watchlist, verbose=verbose_eval)

    

    pred_test_y = model.predict(test_X)

    optR = OptimizedRounder()

    coefficients = [0.5, 1.5, 2.5]

    optR.fit(pred_test_y, test_y)

    coefficients = optR.coefficients()

    pred_test_y_k = optR.predict(pred_test_y, coefficients)

    print("Valid Counts = ", Counter(test_y))

    print("Predicted Counts = ", Counter(pred_test_y_k))

    print("Coefficients = ", coefficients)

    qwk = eval_qwk_lgb_regr(test_y, pred_test_y_k)

    print("QWK = ", qwk)

    print('Predict 2/2')

    pred_test_y2 = model.predict(test_X2)

    return pred_test_y.reshape(-1, 1), pred_test_y2.reshape(-1, 1), features,coefficients, qwk
cat_results = run_cv_model(reduce_train[features], ajusted_test[features], target, runCAT, cat_params, rmse, 'cat')
optR = OptimizedRounder()

coefficients_ = np.mean(cat_results['coefficients'], axis=0)

print(coefficients_)

cat_train_predictions = [r[0] for r in cat_results['train']]

cat_train_predictions = optR.predict(cat_train_predictions, coefficients_).astype(int)

Counter(cat_train_predictions)
optR.predict(cat_train_predictions, coefficients_)
optR = OptimizedRounder()

cat_test_predictions = [r[0] for r in cat_results['test']]

cat_test_predictions = optR.predict(cat_test_predictions, coefficients_).astype(int)

Counter(cat_test_predictions)
eval_qwk_lgb_regr(target, cat_train_predictions)
rmse(target, [r[0] for r in cat_results['train']])
oof_models = [lgb_train_predictions,xgb_train_predictions,cat_train_predictions]

test_models = [lgb_test_predictions, xgb_test_predictions,cat_test_predictions]
from sklearn.linear_model import LinearRegression, Ridge

lr = Ridge(fit_intercept=False)

lr.fit(np.array(oof_models).T, target)

print(lr.coef_)

lr.coef_ = lr.coef_ * 1/(sum(lr.coef_))

print(lr.coef_)

oof_lr = lr.predict(np.array(oof_models).T)

test_preds_lr = lr.predict(np.array(test_models).T)

#lr of nn and lgb and xgb

optR = OptimizedRounder()

optR.fit(oof_lr, target)

coefficients = optR.coefficients()

print(coefficients)

oof_rounded = optR.predict(oof_lr, coefficients)

print(eval_qwk_lgb_regr(target, oof_rounded))

test_rounded_lr = optR.predict(test_preds_lr, coefficients)
sample_submission['accuracy_group'] = test_rounded_lr.astype(int)
sample_submission.to_csv('submission.csv', index=False)

sample_submission['accuracy_group'].value_counts(normalize=True)