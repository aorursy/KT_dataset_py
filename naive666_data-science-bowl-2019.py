# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import xgboost as xgb

from xgboost import XGBClassifier, XGBRegressor

from xgboost import plot_importance

from matplotlib import pyplot

# import shap



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

from time import time

from tqdm import tqdm_notebook as tqdm

from collections import Counter

from scipy import stats

import lightgbm as lgb

from sklearn.metrics import cohen_kappa_score

from sklearn.model_selection import KFold, StratifiedKFold

import gc

import json

pd.set_option('display.max_columns', 1000)
Id = "installation_id"

target = "accuracy_group"
import pandas as pd

sample_submission = pd.read_csv("../input/data-science-bowl-2019/sample_submission.csv")

specs = pd.read_csv("../input/data-science-bowl-2019/specs.csv")

test = pd.read_csv("../input/data-science-bowl-2019/test.csv")

train = pd.read_csv("../input/data-science-bowl-2019/train.csv")

train_labels = pd.read_csv("../input/data-science-bowl-2019/train_labels.csv")
def encode_title(train, test, train_labels):

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

    

    

    return train, test, train_labels, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code, activities_map, all_type_world
train, test, train_labels, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code, activities_map, all_type_world = encode_title(train, test, train_labels)
def cnt_miss(df):

    cnt = 0

    for e in range(len(df)):

        x = df['event_data'].iloc[e]

        y = json.loads(x)['misses']

        cnt += y

    return cnt



def get_4020_acc(df,counter_dict):

    

    for e in ['Cauldron Filler (Assessment)', 'Bird Measurer (Assessment)', 

              'Mushroom Sorter (Assessment)','Chest Sorter (Assessment)']:

        

        Assess_4020 = df[(df.event_code == 4020) & (df.title==activities_map[e])]   

        true_attempts_ = Assess_4020['event_data'].str.contains('true').sum()

        false_attempts_ = Assess_4020['event_data'].str.contains('false').sum()



        measure_assess_accuracy_ = true_attempts_/(true_attempts_+false_attempts_) if (true_attempts_+false_attempts_) != 0 else 0

        counter_dict[e+"_4020_accuracy"] += (counter_dict[e+"_4020_accuracy"] + measure_assess_accuracy_) / 2.0

    

    return counter_dict
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

    

    assess_4020_acc_dict = {'Cauldron Filler (Assessment)_4020_accuracy': 0, 

                            'Mushroom Sorter (Assessment)_4020_accuracy': 0, 

                            'Bird Measurer (Assessment)_4020_accuracy': 0, 

                            'Chest Sorter (Assessment)_4020_accuracy': 0}

    

    game_time_dict = {'Clip_gametime': 0, 'Game_gametime': 0, 

                      'Activity_gametime': 0, 'Assessment_gametime': 0}

    

    last_session_time_sec = 0

    accuracy_groups = {0:0, 1:0, 2:0, 3:0}

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

    last_accuracy_title = {'acc_' + title: -1 for title in assess_titles}

    last_game_time_title = {'lgt_' + title: 0 for title in assess_titles}

    ac_game_time_title = {'agt_' + title: 0 for title in assess_titles}

    ac_true_attempts_title = {'ata_' + title: 0 for title in assess_titles}

    ac_false_attempts_title = {'afa_' + title: 0 for title in assess_titles}

    event_code_count: Dict[str, int] = {ev: 0 for ev in list_of_event_code}

    event_id_count: Dict[str, int] = {eve: 0 for eve in list_of_event_id}

    title_count: Dict[str, int] = {eve: 0 for eve in activities_labels.values()} 

    title_event_code_count: Dict[str, int] = {t_eve: 0 for t_eve in all_title_event_code}

    type_world_count: Dict[str, int] = {w_eve: 0 for w_eve in all_type_world}

    session_count = 0

    

    # itarates through each session of one instalation_id

    for i, session in user_sample.groupby('game_session', sort=False):

        # i = game_session_id

        # session is a DataFrame that contain only one game_session

        

        # get some sessions information

        session_type = session['type'].iloc[0]

        session_title = session['title'].iloc[0]

        session_title_text = activities_labels[session_title]

        

        if session_type == "Activity":

            Activity_mean_event_count = (Activity_mean_event_count + session['event_count'].iloc[-1])/2.0

            

        if session_type == "Game":

            Game_mean_event_count = (Game_mean_event_count + session['event_count'].iloc[-1])/2.0

            

            game_s = session[session.event_code == 2030]

            misses_cnt = cnt_miss(game_s)

            accumulated_game_miss += misses_cnt

            

            try:

                game_round = json.loads(session['event_data'].iloc[-1])["round"]

                mean_game_round =  (mean_game_round + game_round)/ 2.0

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

            durations.append((session.iloc[-1, 2] - session.iloc[0, 2] ).seconds)

            

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

            features['accumulated_accuracy'] = accumulated_accuracy/counter if counter > 0 else 0

            # --------------------------

            features['Cauldron_Filler_4025'] = Cauldron_Filler_4025/counter if counter > 0 else 0

            Assess_4025 = session[(session.event_code == 4025) & (session.title=='Cauldron Filler (Assessment)')]

            true_attempts_ = Assess_4025['event_data'].str.contains('true').sum()

            false_attempts_ = Assess_4025['event_data'].str.contains('false').sum()

            

            cau_assess_accuracy_ = true_attempts_/(true_attempts_+false_attempts_) if (true_attempts_+false_attempts_) != 0 else 0

            Cauldron_Filler_4025 += cau_assess_accuracy_

            

            chest_assessment_uncorrect_sum += len(session[session.event_id=="df4fe8b6"])

            

            Assessment_mean_event_count = (Assessment_mean_event_count + session['event_count'].iloc[-1])/2.0

            # ----------------------------

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

            

        if session_type == 'Game':

            durations_game.append((session.iloc[-1, 2] - session.iloc[0, 2] ).seconds)

            

        if session_type == 'Activity':

            durations_activity.append((session.iloc[-1, 2] - session.iloc[0, 2] ).seconds)

                

        

        session_count += 1

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

        type_world_count = update_counters(type_world_count, 'type_world')

        

        assess_4020_acc_dict = get_4020_acc(session , assess_4020_acc_dict)

        game_time_dict[session_type+'_gametime'] = (game_time_dict[session_type+'_gametime'] + (session['game_time'].iloc[-1]/1000.0))/2.0



        # counts how many actions the player has done so far, used in the feature of the same name

        accumulated_actions += len(session)

        if last_activity != session_type:

            user_activities_count[session_type] += 1

            last_activitiy = session_type 

                        

    # if it't the test_set, only the last assessment must be predicted, the previous are scraped

    if test_set:

        return all_assessments[-1]

    return all_assessments
def get_train_and_test(train, test):

    compiled_train = []

    compiled_test = []

    compiled_test_his = []

    for i, (ins_id, user_sample) in tqdm(enumerate(train.groupby('installation_id', sort = False)), total = 17000):

        compiled_train += get_data(user_sample)

    for ins_id, user_sample in tqdm(test.groupby('installation_id', sort = False), total = 1000):

        test_data = get_data(user_sample, test_set = True)

        compiled_test.append(test_data)

    for i, (ins_id, user_sample) in tqdm(enumerate(test.groupby('installation_id', sort = False)), total = 1000):

        compiled_test_his += get_data(user_sample)

    reduce_train = pd.DataFrame(compiled_train)

    reduce_test = pd.DataFrame(compiled_test)

    reduce_test_his = pd.DataFrame(compiled_test_his)

    

    return reduce_train, reduce_test, reduce_test_his

# tag = 'encode_title'

# # train

# train_result_path = 'train_' + tag + '.pkl'

# new_train.to_pickle(train_result_path)

# # test

# test_result_path = '.test_' + tag + '.pkl'

# new_test.to_pickle(test_result_path)
# tranform function to get the train and test set

reduce_train, reduce_test, reduce_test_his = get_train_and_test(train, test)
def preprocess(reduce_train, reduce_test):

    for df in [reduce_train, reduce_test]:

        df['installation_session_count'] = df.groupby(['installation_id'])['Clip'].transform('count')

        df['installation_duration_mean'] = df.groupby(['installation_id'])['duration_mean'].transform('mean')

        df['installation_duration_std'] = df.groupby(['installation_id'])['duration_mean'].transform('std')

        df['installation_title_nunique'] = df.groupby(['installation_id'])['session_title'].transform('nunique')

        

        df['sum_event_code_count'] = df[[2050, 4100, 4230, 5000, 4235, 2060, 4110, 5010, 2070, 2075, 2080, 2081, 2083, 3110, 4010, 3120, 3121, 4020, 4021, 

                                        4022, 4025, 4030, 4031, 3010, 4035, 4040, 3020, 3021, 4045, 2000, 4050, 2010, 2020, 4070, 2025, 2030, 4080, 2035, 

                                        2040, 4090, 4220, 4095]].sum(axis = 1)

        

        df['installation_event_code_count_mean'] = df.groupby(['installation_id'])['sum_event_code_count'].transform('mean')

        df['installation_event_code_count_std'] = df.groupby(['installation_id'])['sum_event_code_count'].transform('std')

        

    features = reduce_train.loc[(reduce_train.sum(axis=1) != 0), (reduce_train.sum(axis=0) != 0)].columns # delete useless columns

    features = [x for x in features if x not in ['accuracy_group', 'installation_id']] + ['acc_' + title for title in assess_titles]

   

    return reduce_train, reduce_test, features



# call feature engineering function

reduce_train, reduce_test, features = preprocess(reduce_train, reduce_test)
reduce_train.shape, reduce_test.shape, reduce_test_his.shape
len(features)
reduce_train.head()
categoricals = ['session_title']
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
len(reduce_train.columns)
# This function return the remaining valid features after deleting the 0 features.

def delete_zero_columns(reduce_train):

    features = []

    for column_name in reduce_train.columns:

        if column_name not in ['accuracy_group', 'installation_id']:

            if np.sum(reduce_train[column_name], axis = 0) != 0:

                features.append(column_name)

    return features
del_features = delete_zero_columns(reduce_train)
len(del_features)
len(features)
features = set(del_features).union(set(features))
len(features)
from scipy.stats import ks_2samp

from sklearn.preprocessing import power_transform
def power_transform_data(reduce_train, reduce_test, features):

    train_length = len(reduce_train)

    train_test = pd.concat([reduce_train[features], reduce_test[features]], axis = 0)

#     new_reduce_train = reduce_train.copy()

#     new_reduce_test = reduce_test.copy()

    for feature in features:

#         new_reduce_train[feature] = power_transform((new_reduce_train[feature].values).reshape(-1,1), method = 'yeo-johnson').reshape(-1)

#         new_reduce_test[feature] = power_transform((new_reduce_test[feature].values).reshape(-1,1), method = 'yeo-johnson').reshape(-1)

        train_test[feature] = power_transform((train_test[feature].values).reshape(-1,1), method = 'yeo-johnson').reshape(-1)

        new_reduce_train = train_test.iloc[:train_length, :]

        new_reduce_test = train_test.iloc[train_length:, :]

    return new_reduce_train, new_reduce_test
# This function drop the columns with different distribution in train and test set

def check_distribution(reduce_train, reduce_test, features):

    to_exclude = []

    for feature in features:

        train_mean = reduce_train[feature].mean()

        train_std = reduce_train[feature].std()

        test_mean = reduce_test[feature].mean()

        test_std = reduce_test[feature].std()

        #print('train_mean: {}, test_mean: {}, train_std: {}, test_std: {}'.format(train_mean, test_mean, train_std, test_std))

        if test_mean != 0:

            if abs(train_mean / test_mean) > 10 or abs(train_mean / test_mean) < 0.1:

                print('**************************')

                print('Feature: {}, train_mean: {}, test_mean: {}'.format(feature, train_mean, test_mean))

                to_exclude.append(feature)

        else:

            if abs(train_mean) > 10:

                print('**************************')

                print('Feature: {}, train_mean: {}, test_mean: {}'.format(feature, train_mean, test_mean))

                to_exclude.append(feature)

    return to_exclude
to_exclude = check_distribution(reduce_train, reduce_test, features)
new_features = list(set(features) - set(to_exclude))
len(new_features)
unique_id = list(reduce_train['installation_id'].unique())
#reduce_train.loc[reduce_train['installation_id'] == unique_id[0], :]
# debug for "LightGBMError: Do not support special JSON characters in feature name."

def change_json(df):

    df.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in df.columns]

    return df
#  params =  {'num_leaves': 61,  

#            'min_child_weight': 0.03454472573214212,

#            'feature_fraction': 0.3797454081646243,

#            'bagging_fraction': 0.4181193142567742,

#            'min_data_in_leaf': 96,  

#            'objective': 'regression',

#            "metric": 'rmse',

#            'learning_rate': 0.1, 

#            "boosting_type": "gbdt",

#            "bagging_seed": 11,

#            "verbosity": -1,

#            'reg_alpha': 0.3899927210061127,

#            'reg_lambda': 0.6485237330340494,

#            'random_state': 46,

#            'num_threads': 16,

#            'lambda_l1': 1,  

#            'lambda_l2': 1,

#            'n_estimators': 8000,

#            'early_stopping': 150

#     }

def run_lgb_regression(reduce_train, reduce_test, useful_features, n_splits, depth, params):

    loss_scores = []

    useful_features = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in useful_features]

    reduce_train = change_json(reduce_train)

    reduce_test = change_json(reduce_test)

    feature_importance = {}

    kf = StratifiedKFold(n_splits = 5, random_state = 42, shuffle = True)

    oof_predict = np.zeros((len(reduce_train), ))

    y_pred = np.zeros((len(reduce_test), ))

    for fold, (train_index, test_index) in enumerate(kf.split(reduce_train, reduce_train[target])):

        print('Fold: {}'.format(fold + 1))

        X_train = reduce_train[useful_features].iloc[train_index]

        X_val = reduce_train[useful_features].iloc[test_index]

        y_train = reduce_train[target].iloc[train_index]

        y_val = reduce_train[target].iloc[test_index]

        

        train_set = lgb.Dataset(X_train, y_train, categorical_feature = categoricals)

        val_set = lgb.Dataset(X_val, y_val, categorical_feature = categoricals)

        

        model = lgb.train(params, train_set, num_boost_round = params['n_estimators'], valid_sets = [train_set, val_set],

                         early_stopping_rounds = params['early_stopping'])

        oof_predict[test_index] = model.predict(X_val)

        y_pred += model.predict(reduce_test[useful_features]) / n_splits

        #_, loss_score, _ = eval_qwk_lgb_regr(reduce_train.loc[test_index, target], oof_predict) 

        #loss_scores.append(loss_score)

        #print('The cohen_kappa score for folder_{} is: {}'.format(fold + 1, loss_score))

        

        feature_importance['fold_{}'.format(fold + 1)] = model.feature_importance()

        print(feature_importance['fold_{}'.format(fold + 1)][:5])

        

    return y_pred, oof_predict, feature_importance
#y_pred, oof_predict, feature_importance = run_lgb_regression(reduce_train, reduce_test, new_features, 5, 10)
#eval_qwk_lgb_regr(reduce_train[target], oof_predict)
import seaborn as sns
def draw_feature_importance(feature_importance, features):

    feature_imp = pd.DataFrame(zip(feature_importance, features), columns=['Value','Feature'])

    plt.figure(figsize=(20, 500))

    sns.barplot(x="Value", y="Feature", data = feature_imp.sort_values(by = "Value", ascending = False))

    plt.title('LightGBM Features (avg over folds)')

    plt.tight_layout()

    plt.show()
def get_important_features(feature_importance, feature):

    feature_imp = pd.DataFrame(zip(feature_importance, features), columns=['Value','Feature'])

    feature_imp = feature_imp.sort_values(by = 'Value', ascending = False)

    return feature_imp
def check_top_features(top_n, feature_imp_fold_list):

    set1 = set(list(feature_imp_fold_list[0].head(top_n)['Feature'].values))

    set2 = set(list(feature_imp_fold_list[1].head(top_n)['Feature'].values))

    set3 = set(list(feature_imp_fold_list[2].head(top_n)['Feature'].values))

    set4 = set(list(feature_imp_fold_list[3].head(top_n)['Feature'].values))

    set5 = set(list(feature_imp_fold_list[3].head(top_n)['Feature'].values))

    top_features = set.intersection(set1, set2, set3, set4, set5)

    return top_features
#top_features = check_top_features(500, feature_imp_fold_list)
#len(top_features)
from bayes_opt import BayesianOptimization

from sklearn.metrics import mean_squared_error
def model(reduce_train, reduce_test, useful_features, n_splits, num_leaves, max_depth, min_child_weight, feature_fraction, lambda_l1, lambda_l2, 

          bagging_fraction, min_data_in_leaf, learning_rate, reg_alpha, reg_lambda, n_estimators):

     

        params =  {'num_leaves': int(num_leaves),  

            'max_depth' : int(max_depth),

           'min_child_weight': min_child_weight,

           'feature_fraction': feature_fraction,

           'bagging_fraction': bagging_fraction,

           'min_data_in_leaf': int(min_data_in_leaf), 

           'objective': 'regression',

           "metric": 'rmse',

           'learning_rate': learning_rate, 

           "boosting_type": "gbdt",

           "bagging_seed": 11,

           "verbosity": -1,

           'reg_alpha': reg_alpha,

           'reg_lambda': reg_lambda,

           'random_state': 46,

           'num_threads': 16,

           'lambda_l1': lambda_l1,  

           'lambda_l2': lambda_l2, 

           'n_estimators': int(n_estimators),

           'early_stopping': 150

    }

        def run_lgb(reduce_train, reduce_test, useful_features, n_splits = n_splits):

            #useful_features.remove('installation_id')

            rmse_score_list = []

            useful_features = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in useful_features]

            reduce_train = change_json(reduce_train)

            reduce_test = change_json(reduce_test)

            kf = StratifiedKFold(n_splits = n_splits, random_state = 42, shuffle = True)

            oof_predict = np.zeros((len(reduce_train), ))

            y_pred = np.zeros((len(reduce_test), ))

            for fold, (train_index, test_index) in enumerate(kf.split(reduce_train, reduce_train[target])):

                X_train = reduce_train[useful_features].iloc[train_index]

                X_val = reduce_train[useful_features].iloc[test_index]

                y_train = reduce_train[target].iloc[train_index]

                y_val = reduce_train[target].iloc[test_index]

                train_set = lgb.Dataset(X_train, y_train, categorical_feature = categoricals)

                val_set = lgb.Dataset(X_val, y_val, categorical_feature = categoricals)

                lgb_model = lgb.train(params, train_set, num_boost_round = params['n_estimators'], valid_sets = [train_set, val_set],

                             early_stopping_rounds = params['early_stopping'])

                val_predict = lgb_model.predict(X_val)

                rmse_score = np.sqrt(mean_squared_error(val_predict, y_val))

                rmse_score_list.append(rmse_score)

            return -np.mean(rmse_score_list)

        

        return run_lgb(reduce_train, reduce_test, useful_features)
reduce_train.shape
from functools import partial

partial_model = partial(model, reduce_train, reduce_test, new_features, n_splits = 2)
bounds_LGB = {

    'num_leaves' : (50, 100),

    'max_depth': (8, 30),

    'min_child_weight' : (0.01, 0.6),

    'min_data_in_leaf' : (80, 120),

    'feature_fraction' : (0.1, 0.8),

    'lambda_l1': (0, 10),

    'lambda_l2': (0, 10),

    'bagging_fraction': (0.2, 1),

    'learning_rate': (0.01, 0.8),

    'reg_alpha' : (0.1 , 5), 

    'reg_lambda' : (0.1, 5),

    'n_estimators' : (5000,8000)

}
import warnings
# init_points = 16

# n_iter = 16

# LGB_BO = BayesianOptimization(partial_model, bounds_LGB, random_state=1029)

# with warnings.catch_warnings():

#     warnings.filterwarnings('ignore')

#     LGB_BO.maximize(init_points = init_points, n_iter = n_iter, acq='ei', alpha=1e-6)
#best_LGB_BO_params = LGB_BO.max['params']
#LGB_BO.max
#best_LGB_BO_params
bayesian_params =  {'num_leaves': 50,  

            'max_depth' : 30,

           'min_child_weight': 0.01,

           'feature_fraction': 0.8,

           'bagging_fraction': 0.2,

           'min_data_in_leaf': 80, 

           'objective': 'regression',

           "metric": 'rmse',

           'learning_rate': 0.01, 

           "boosting_type": "gbdt",

           "bagging_seed": 11,

           "verbosity": -1,

           'reg_alpha': 50,

           'reg_lambda': 0.1,

           'random_state': 46,

           'num_threads': 16,

           'lambda_l1': 10,  

           'lambda_l2': 0, 

           'n_estimators': 5149,

           'early_stopping': 150

    }
y_pred_bayes, oof_predict_bayes, feature_importance_bayes = run_lgb_regression(reduce_train, reduce_test, new_features, 5, 10, bayesian_params)
oof_predict_bayes
eval_qwk_lgb_regr(reduce_train[target], oof_predict_bayes)
feature_imp_fold_1 = get_important_features(feature_importance_bayes['fold_1'], new_features)

feature_imp_fold_2 = get_important_features(feature_importance_bayes['fold_2'], new_features)

feature_imp_fold_3 = get_important_features(feature_importance_bayes['fold_3'], new_features)

feature_imp_fold_4 = get_important_features(feature_importance_bayes['fold_4'], new_features)

feature_imp_fold_5 = get_important_features(feature_importance_bayes['fold_5'], new_features)

feature_imp_fold_list = [feature_imp_fold_1, feature_imp_fold_2, feature_imp_fold_3, feature_imp_fold_4, feature_imp_fold_5]
def merge_feature_imp(feature_imp_fold_list):

    feature_imp_fold_1 = feature_imp_fold_list[0].set_index('Feature')

    feature_imp_fold_2 = feature_imp_fold_list[1].set_index('Feature')

    feature_imp_fold_3 = feature_imp_fold_list[2].set_index('Feature')

    feature_imp_fold_4 = feature_imp_fold_list[3].set_index('Feature')

    feature_imp_fold_5 = feature_imp_fold_list[4].set_index('Feature')

    df1 = pd.merge(feature_imp_fold_1, feature_imp_fold_2, how = 'inner', left_index = True, right_index = True)

    df2 = df1.merge(feature_imp_fold_3, how = 'inner', left_index = True, right_index = True)

    df3 = df2.merge(feature_imp_fold_4, how = 'inner', left_index = True, right_index = True)

    final_df = df3.merge(feature_imp_fold_5, how = 'inner', left_index = True, right_index = True)

    final_df.columns = ['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5']

    final_df['average'] = final_df[['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5']].mean(axis = 1)

    return final_df
feature_importance_from_all_folders = merge_feature_imp(feature_imp_fold_list)
feature_importance_from_all_folders = feature_importance_from_all_folders.sort_values('average', ascending = False)
top_features = feature_importance_from_all_folders.loc[feature_importance_from_all_folders['average'] > 5, :].index
top_features = list(top_features.values)
if 'session_title' not in top_features:

    top_features.append('session_title')
#partial_model_top_features = partial(model, reduce_train, reduce_test, top_features, 5)
# init_points = 16

# n_iter = 16

# LGB_BO_top_features = BayesianOptimization(partial_model_top_features, bounds_LGB, random_state = 1029)

# with warnings.catch_warnings():

#     warnings.filterwarnings('ignore')

#     LGB_BO_top_features.maximize(init_points = init_points, n_iter = n_iter, acq='ei', alpha=1e-6)
#best_LGB_BO_top_features_params = LGB_BO_top_features.max['params']
#LGB_BO_top_features.max
#best_LGB_BO_top_features_params
bayesian_params_top_features =  {'num_leaves': 64,  

            'max_depth' : 17,

           'min_child_weight': 0.39533312446497537,

           'feature_fraction': 0.5373674618462821,

           'bagging_fraction': 0.2357059531074505,

           'min_data_in_leaf': 112, 

           'objective': 'regression',

           "metric": 'rmse',

           'learning_rate': 0.018024583616814218, 

           "boosting_type": "gbdt",

           "bagging_seed": 11,

           "verbosity": -1,

           'reg_alpha': 1.6071134326080774,

           'reg_lambda': 1.6430470013389429,

           'random_state': 46,

           'num_threads': 16,

           'lambda_l1': 3.0475007160079546,  

           'lambda_l2': 4.476200330834915, 

           'n_estimators': 5126,

           'early_stopping': 150

    }
y_pred_bayes_top_features, oof_predict_bayes_top_features, feature_importance_bayes_top_features = run_lgb_regression(reduce_train, reduce_test, 

                                                                                             top_features, 5, 10, bayesian_params_top_features)
oof_predict_bayes_top_features
eval_qwk_lgb_regr(reduce_train[target], oof_predict_bayes_top_features)
def reg_to_cat(y_regress):

    dist = Counter(reduce_train['accuracy_group'])

    for k in dist:

        dist[k] /= len(reduce_train)

    acum = 0

    bound = {}

    for i in range(3):

        acum += dist[i]

        bound[i] = np.percentile(y_regress, acum * 100)



    def classify(x):

        if x <= bound[0]:

            return 0

        elif x <= bound[1]:

            return 1

        elif x <= bound[2]:

            return 2

        else:

            return 3



    y_pred = np.array(list(map(classify, y_regress)))

    return y_pred

y_pred = reg_to_cat(y_pred_bayes_top_features)
sample_submission['accuracy_group'] = y_pred.astype(int)

sample_submission.to_csv('./submission.csv', index=False)

sample_submission['accuracy_group'].value_counts(normalize=True)