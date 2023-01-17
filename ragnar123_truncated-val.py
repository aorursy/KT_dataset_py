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

from time import time

from tqdm import tqdm_notebook as tqdm

from collections import Counter

from scipy import stats

import lightgbm as lgb

from sklearn.metrics import cohen_kappa_score, mean_squared_error

from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

import gc

import json

pd.set_option('display.max_columns', 1000)

from sklearn import metrics

import warnings

warnings.filterwarnings('ignore')

import random
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

    # in the train_set, all assessments goes to the dataset

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
# thank to Bruno

def eval_qwk_lgb_regr(y_pred, train_t):

    """

    Fast cappa eval function for lgb.

    """

    dist = Counter(train_t['accuracy_group'])

    for k in dist:

        dist[k] /= len(train_t)

    

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

    

    return y_pred



def predict(sample_submission, y_pred):

    sample_submission['accuracy_group'] = y_pred

    sample_submission['accuracy_group'] = sample_submission['accuracy_group'].astype(int)

    sample_submission.to_csv('submission.csv', index = False)

    print(sample_submission['accuracy_group'].value_counts(normalize = True))
def get_random_assessment(reduce_train):

    used_idx = []

    for iid in tqdm(set(reduce_train['installation_id'])):

        list_ = list(reduce_train[reduce_train['installation_id']==iid].index)

        cur = random.choices(list_, k = 1)[0]

        used_idx.append(cur)

    reduce_train_t = reduce_train.loc[used_idx]

    return reduce_train_t, used_idx
# for each validation fold extract one random observation for each installation id to simulate the test set

def run_lgb(reduce_train, reduce_test, features):

    # features found in initial bayesian optimization

    params = {'boosting_type': 'gbdt', 

              'metric': 'rmse', 

              'objective': 'regression', 

              'eval_metric': 'cappa', 

              'n_jobs': -1, 

              'seed': 42, 

              'num_leaves': 26, 

              'learning_rate': 0.077439684887749, 

              'max_depth': 33, 

              'lambda_l1': 3.27791989030057, 

              'lambda_l2': 1.3047627805931334, 

              'bagging_fraction': 0.896924978584253, 

              'bagging_freq': 1, 

              'colsample_bytree': 0.8710772167017853}



    kf = GroupKFold(n_splits = 5)

    target = 'accuracy_group'

    oof_pred = np.zeros(len(reduce_train))

    y_pred = np.zeros(len(reduce_test))

    ind = []



    for fold, (tr_ind, val_ind) in enumerate(kf.split(reduce_train, groups = reduce_train['installation_id'])):

        print('Fold:', fold + 1)

        x_train, x_val = reduce_train[features].iloc[tr_ind], reduce_train[features].iloc[val_ind]

        y_train, y_val = reduce_train[target][tr_ind], reduce_train[target][val_ind]

        x_train.drop('installation_id', inplace = True, axis = 1)

        train_set = lgb.Dataset(x_train, y_train, categorical_feature = ['session_title'])





        x_val, idx_val = get_random_assessment(x_val)

        ind.extend(idx_val)

        x_val.drop('installation_id', inplace = True, axis = 1)

        y_val = y_val.loc[idx_val]

        val_set = lgb.Dataset(x_val, y_val, categorical_feature = ['session_title'])



        model = lgb.train(params, train_set, num_boost_round = 100000, early_stopping_rounds = 100, 

                         valid_sets = [train_set, val_set], verbose_eval = 100)



        oof_pred[idx_val] = model.predict(x_val)

        y_pred += model.predict(reduce_test[[x for x in features if x not in ['installation_id']]]) / kf.n_splits

    oof_rmse_score = np.sqrt(mean_squared_error(reduce_train[target][ind], oof_pred[ind]))

    oof_cohen_score = cohen_kappa_score(reduce_train[target][ind], eval_qwk_lgb_regr(oof_pred[ind], reduce_train), weights = 'quadratic')

    print('Our oof rmse score is:', oof_rmse_score)

    print('Our oof cohen kappa score is:', oof_cohen_score)

    return y_pred, oof_rmse_score, oof_cohen_score
# read data

train, test, train_labels, specs, sample_submission = read_data()

# get usefull dict with maping encode

train, test, train_labels, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code, activities_map, all_type_world = encode_title(train, test, train_labels)

# tranform function to get the train and test set

reduce_train, reduce_test, reduce_test_his = get_train_and_test(train, test)

# delete train and test to release memory

del train, test
old_features = list(reduce_train.columns[0:99]) + list(reduce_train.columns[886:])

el_features = ['accuracy_group', 'accuracy', 'installation_id']

old_features = [col for col in old_features if col not in el_features]

event_id_features = list(reduce_train.columns[99:483])

title_event_code_cross = list(reduce_train.columns[483:886])

features = old_features + event_id_features + title_event_code_cross
def remove_correlated_features(reduce_train):

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

    return to_remove

to_remove = remove_correlated_features(reduce_train)

features = [col for col in features if col not in to_remove]

features = [col for col in features if col not in ['Heavy, Heavier, Heaviest_2000', 'Heavy, Heavier, Heaviest']]

features.append('installation_id')

print('Training with {} features'.format(len(features)))
# function to exclude columns from the train and test set if the mean is different, also adjust test column by a factor to simulate the same distribution

def exclude(reduce_train, reduce_test, features):

    to_exclude = [] 

    ajusted_test = reduce_test.copy()

    for feature in features:

        if feature not in ['accuracy_group', 'installation_id', 'session_title']:

            data = reduce_train[feature]

            train_mean = data.mean()

            data = ajusted_test[feature] 

            test_mean = data.mean()

            try:

                ajust_factor = train_mean / test_mean

                if ajust_factor > 10 or ajust_factor < 0.1:# or error > 0.01:

                    to_exclude.append(feature)

                    print(feature)

                else:

                    ajusted_test[feature] *= ajust_factor

            except:

                to_exclude.append(feature)

                print(feature)

    return to_exclude, ajusted_test



to_exclude, ajusted_test = exclude(reduce_train, reduce_test, features)

features = [col for col in features if col not in to_exclude]
# train 5 times because the evaluation and training data change with the randomness

y_pred_1, oof_rmse_score_1, oof_cohen_score_1 = run_lgb(reduce_train, ajusted_test, features)

y_pred_2, oof_rmse_score_2, oof_cohen_score_2 = run_lgb(reduce_train, ajusted_test, features)

y_pred_3, oof_rmse_score_3, oof_cohen_score_3 = run_lgb(reduce_train, ajusted_test, features)

y_pred_4, oof_rmse_score_4, oof_cohen_score_4 = run_lgb(reduce_train, ajusted_test, features)

y_pred_5, oof_rmse_score_5, oof_cohen_score_5 = run_lgb(reduce_train, ajusted_test, features)

mean_rmse = (oof_rmse_score_1 + oof_rmse_score_2 + oof_rmse_score_3 + oof_rmse_score_4 + oof_rmse_score_5) / 5

mean_cohen_kappa = (oof_cohen_score_1 + oof_cohen_score_2 + oof_cohen_score_3 + oof_cohen_score_4 + oof_cohen_score_5) / 5

print('Our mean rmse score is: ', mean_rmse)

print('Our mean cohen kappa score is: ', mean_cohen_kappa)

y_final = (y_pred_1 + y_pred_2 + y_pred_3 + y_pred_4 + y_pred_5) / 5

y_final = eval_qwk_lgb_regr(y_final, reduce_train)

predict(sample_submission, y_final)