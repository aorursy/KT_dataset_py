import numpy as np  # linear algebra

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import xgboost as xgb

from xgboost import XGBClassifier, XGBRegressor

from xgboost import plot_importance

from catboost import CatBoostRegressor

from matplotlib import pyplot

import shap



import os

# Any results you write to the current directory are saved as output.

from time import time

from tqdm import tqdm

from collections import Counter

from scipy import stats

import lightgbm as lgb

from sklearn.metrics import cohen_kappa_score, mean_squared_error

from typing import Dict

from sklearn.model_selection import KFold, StratifiedKFold

import gc

import json



pd.set_option('display.max_columns', 1000)



local = False





def eval_qwk_lgb_regr(y_true, y_pred):

    """

    Fast cappa eval function for lgb.

    """

    dist = Counter(reduce_train['accuracy_group'])

    for k in dist:

        dist[k] /= len(reduce_train)

    reduce_train['accuracy_group'].hist()



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





def read_data():

    if local:

        print('Reading train.csv file....')

        train = pd.read_pickle('../data/train.pickle')

        print('Training.csv file have {} rows and {} columns'.format(train.shape[0], train.shape[1]))



        print('Reading test.csv file....')

        test = pd.read_pickle('../data/test.pickle')

        print('Test.csv file have {} rows and {} columns'.format(test.shape[0], test.shape[1]))



        print('Reading train_labels.csv file....')

        train_labels = pd.read_pickle('../data/train_labels.pickle')

        print('Train_labels.csv file have {} rows and {} columns'.format(train_labels.shape[0], train_labels.shape[1]))



        print('Reading specs.csv file....')

        specs = pd.read_pickle('../data/specs.pickle')

        print('Specs.csv file have {} rows and {} columns'.format(specs.shape[0], specs.shape[1]))



        print('Reading sample_submission.csv file....')

        sample_submission = pd.read_csv('../data/sample_submission.csv')

        print('Sample_submission.csv file have {} rows and {} columns'.format(sample_submission.shape[0],

                                                                              sample_submission.shape[1]))

        return train, test, train_labels, specs, sample_submission

    else:

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

        print('Sample_submission.csv file have {} rows and {} columns'.format(sample_submission.shape[0],

                                                                              sample_submission.shape[1]))

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



    user_activities_count = {'Clip': 0, 'Activity': 0, 'Assessment': 0, 'Game': 0}



    # new features: time spent in each activity

    last_session_time_sec = 0

    accuracy_groups = {0: 0, 1: 0, 2: 0, 3: 0}

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



    # last features

    sessions_count = 0



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

        if (session_type == 'Assessment') & (test_set or len(session) > 1):

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

            features['installation_session_count'] = sessions_count



            variety_features = [('var_event_code', event_code_count),

                                ('var_event_id', event_id_count),

                                ('var_title', title_count),

                                ('var_title_event_code', title_event_code_count)]



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

            # the time spent in the app so far

            if durations == []:

                features['duration_mean'] = 0

                features["duration_std"] = 0

                features["duration_med"] = 0

                features["duration_max"] = 0



            else:

                features['duration_mean'] = np.mean(durations)

                features["duration_std"] = np.std(durations)

                features["duration_med"] = np.median(durations)

                features["duration_max"] = np.max(durations)

            durations.append((session.iloc[-1, 2] - session.iloc[0, 2]).seconds)

            # the accurace is the all time wins divided by the all time attempts

            features['accumulated_accuracy'] = accumulated_accuracy / counter if counter > 0 else 0

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



            if test_set:

                all_assessments.append(features)

            elif true_attempts + false_attempts > 0:

                all_assessments.append(features)



            counter += 1



        sessions_count += 1



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

    for i, (ins_id, user_sample) in tqdm(enumerate(train.groupby('installation_id', sort=False)), total=17000):

        compiled_train += get_data(user_sample)

    for ins_id, user_sample in tqdm(test.groupby('installation_id', sort=False), total=1000):

        test_data = get_data(user_sample, test_set=True)

        compiled_test.append(test_data)

    reduce_train = pd.DataFrame(compiled_train)

    reduce_test = pd.DataFrame(compiled_test)

    categoricals = ['session_title']

    return reduce_train, reduce_test, categoricals
# read data

train, test, train_labels, specs, sample_submission = read_data()

# get usefull dict with maping encode

train, test, train_labels, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code = encode_title(

    train, test, train_labels)

# tranform function to get the train and test set

reduce_train, reduce_test, categoricals = get_train_and_test(train, test)





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



plt.show()



reduce_train.columns = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in reduce_train.columns]

reduce_test.columns = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in reduce_test.columns]

features = reduce_train.columns  # delete useless columns

features = [x for x in features if x not in ['accuracy_group', 'installation_id']]

ajusted_test = reduce_test.copy()
from sklearn.model_selection import train_test_split

import eli5

from eli5.sklearn import PermutationImportance



clf = xgb.XGBRegressor(

    learning_rate=0.05,

    max_depth=6,

    metric="rmse",

    n_estimators=200

)

y = reduce_train["accuracy_group"]



# hold out(I think kfold is better than hold out)

tr_tr, tr_val, y_tr, y_val = train_test_split(reduce_train[features], y, random_state=0, train_size=0.8)

clf.fit(tr_tr, y_tr)



perm = PermutationImportance(clf, random_state=0).fit(tr_val, y_val)

eli5.show_weights(perm, feature_names=tr_tr.columns.tolist(), top=900)
th_list = [-10000, 0.00001,0.0001, 0.0002,0.0003]
len(features)
class Base_Model(object):



    def __init__(self, train_df, test_df, features, categoricals=[], n_splits=5, verbose=True):

        self.train_df = train_df

        self.test_df = test_df

        self.features = features

        self.n_splits = n_splits

        self.categoricals = categoricals

        self.target = 'accuracy_group'

        self.cv = self.get_cv()

        self.verbose = verbose

        self.params = self.get_params()

        self.y_pred, self.score, self.model = self.fit()



    def train_model(self, train_set, val_set):

        raise NotImplementedError



    def get_cv(self):

        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        return cv.split(self.train_df, self.train_df[self.target])



    def get_params(self):

        raise NotImplementedError



    def convert_dataset(self, x_train, y_train, x_val, y_val):

        raise NotImplementedError



    def convert_x(self, x):

        return x



    def fit(self):

        oof_pred = np.zeros((len(reduce_train),))

        y_pred = np.zeros((len(reduce_test),))

        for fold, (train_idx, val_idx) in enumerate(self.cv):

            x_train, x_val = self.train_df[self.features].iloc[train_idx], self.train_df[self.features].iloc[val_idx]

            y_train, y_val = self.train_df[self.target][train_idx], self.train_df[self.target][val_idx]

            train_set, val_set = self.convert_dataset(x_train, y_train, x_val, y_val)

            model = self.train_model(train_set, val_set)

            conv_x_val = self.convert_x(x_val)

            oof_pred[val_idx] = model.predict(conv_x_val).reshape(oof_pred[val_idx].shape)

            x_test = self.convert_x(self.test_df[self.features])

            y_pred += model.predict(x_test).reshape(y_pred.shape) / self.n_splits

            print('Partial score of fold {} is: {}'.format(fold, eval_qwk_lgb_regr(y_val, oof_pred[val_idx])[1]))

        _, loss_score, _ = eval_qwk_lgb_regr(self.train_df[self.target], oof_pred)

        if self.verbose:

            print('Our oof cohen kappa score is: ', loss_score)

        return y_pred, loss_score, model



class Xgb_Model(Base_Model):



    def train_model(self, train_set, val_set):

        verbosity = 200 if self.verbose else 0

        return xgb.train(self.params, train_set,

                         num_boost_round=3000, evals=[(train_set, 'train'), (val_set, 'val')],

                         verbose_eval=verbosity, early_stopping_rounds=150)



    def convert_dataset(self, x_train, y_train, x_val, y_val):

        train_set = xgb.DMatrix(x_train, y_train)

        val_set = xgb.DMatrix(x_val, y_val)

        return train_set, val_set



    def convert_x(self, x):

        return xgb.DMatrix(x)



    def get_params(self):

        params = {'colsample_bytree': 0.8,

                  'learning_rate': 0.01,

                  'max_depth': 8,

                  'subsample': 1,

                  'objective': 'reg:squarederror',

                  # 'eval_metric':'rmse',

                  'min_child_weight': 3,

                  'gamma': 0.25,

                  'n_estimators': 5000}

        if local:

            params["gpu_hist"] = True



        return params
from sklearn.feature_selection import SelectFromModel

categoricals = ['session_title']
score_list = []

for i, x in enumerate(th_list):

    print("---------------------\nThreshold is " + str(x))

    sel = SelectFromModel(perm, threshold=th_list[i], prefit=True)

    # new_features = sel.transform(tr_tr)

    features_idx = sel.get_support()

    

    new_features = reduce_train[features].columns[features_idx]

    

    print("Training using {} features. ({}%) \n-----------------------".format(len(new_features), len(new_features)*100 / len(features)))



    xgb_model = Xgb_Model(reduce_train, ajusted_test, new_features, categoricals=categoricals)

    score_list.append(xgb_model.score)
for (th, sc) in zip(th_list, score_list):

    print("Threshold is {} and Kappa score is {}".format(th, sc))