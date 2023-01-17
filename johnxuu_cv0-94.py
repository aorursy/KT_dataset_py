import numpy as np

import pandas as pd

# import datetime

# from catboost import CatBoostClassifier

# from time import time

from tqdm import tqdm_notebook as tqdm

from collections import Counter

# from scipy import stats
from sklearn.metrics import confusion_matrix

# this function is the quadratic weighted kappa (the metric used for the competition submission)

def qwk(act,pred,n=4,hist_range=(0,3)):

    

    # Calculate the percent each class was tagged each label

    O = confusion_matrix(act,pred)

    # normalize to sum 1

    O = np.divide(O,np.sum(O))

    

    # create a new matrix of zeroes that match the size of the confusion matrix

    # this matriz looks as a weight matrix that give more weight to the corrects

    W = np.zeros((n,n))

    for i in range(n):

        for j in range(n):

            # makes a weird matrix that is bigger in the corners top-right and botton-left (= 1)

            W[i][j] = ((i-j)**2)/((n-1)**2)

            

    # make two histograms of the categories real X prediction

    act_hist = np.histogram(act,bins=n,range=hist_range)[0]

    prd_hist = np.histogram(pred,bins=n,range=hist_range)[0]

    

    # multiply the two histograms using outer product

    E = np.outer(act_hist,prd_hist)

    E = np.divide(E,np.sum(E)) # normalize to sum 1

    

    # apply the weights to the confusion matrix

    num = np.sum(np.multiply(W,O))

    # apply the weights to the histograms

    den = np.sum(np.multiply(W,E))

    

    return 1-np.divide(num,den)

    
filepath = '/kaggle/input/data-science-bowl-2019/'

train = pd.read_csv(filepath+'train.csv')

train_labels = pd.read_csv(filepath+'train_labels.csv')

specs = pd.read_csv(filepath+'specs.csv')

test = pd.read_csv(filepath+'test.csv')

submission = pd.read_csv(filepath+'sample_submission.csv')
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

assess_titles = list(set(train[train['type'] == 'Assessment']['title'].value_counts().index)\

                     .union(set(test[test['type'] == 'Assessment']['title'].value_counts().index)))

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
def type_duration_features(type_durations, type_str):

    """

        type_durations: clip_durations, activity_durations, game_durations, assessment_durations

        type_str: "clip", "activity", "game", "assessment"

    """

    features = {}

    if type_durations == []:

        features[type_str+'_duration_mean'] = 0

        features[type_str+'_duration_last'] = 0

        features[type_str+'_duration_max'] = 0

        features[type_str+'_duration_min'] = 0

        features[type_str+'_duration_percentil_10'] = 0

        features[type_str+'_duration_percentil_20'] = 0

        features[type_str+'_duration_percentil_30'] = 0

        features[type_str+'_duration_percentil_40'] = 0

        features[type_str+'_duration_percentil_50'] = 0

        features[type_str+'_duration_percentil_60'] = 0

        features[type_str+'_duration_percentil_70'] = 0

        features[type_str+'_duration_percentil_80'] = 0

        features[type_str+'_duration_percentil_90'] = 0

        

        features[type_str+'_duration_percentil_1'] = 0

        features[type_str+'_duration_percentil_5'] = 0

        features[type_str+'_duration_percentil_15'] = 0

        features[type_str+'_duration_percentil_25'] = 0

        features[type_str+'_duration_percentil_35'] = 0

        features[type_str+'_duration_percentil_45'] = 0

        features[type_str+'_duration_percentil_55'] = 0

        features[type_str+'_duration_percentil_65'] = 0

        features[type_str+'_duration_percentil_75'] = 0

        features[type_str+'_duration_percentil_85'] = 0

        features[type_str+'_duration_percentil_95'] = 0

        features[type_str+'_duration_percentil_99'] = 0

        

        features[type_str+'_duration_std'] = 0

        features[type_str+'_duration_var'] = 0

        features[type_str+'_duration_std_top'] = 0

        features[type_str+'_duration_std_bottom'] = 0

        features[type_str+'_duration_max_range'] = 0

        for i in range(21):

            features[type_str+'_duration_relative_percentile_{}'.format(i)] = 0

    else:

        features[type_str+'_duration_std'] = np.std(type_durations)

        features[type_str+'_duration_var'] = np.var(type_durations)

        features[type_str+'_duration_std_top'] = np.mean(type_durations) + np.std(type_durations) 

        features[type_str+'_duration_std_bottom'] = np.mean(type_durations) - np.std(type_durations)

        features[type_str+'_duration_max_range'] = max(type_durations) - min(type_durations)

        

        features[type_str+'_duration_mean'] = np.mean(type_durations)

        features[type_str+'_duration_last'] = type_durations[-1]

        features[type_str+'_duration_max'] = max(type_durations)

        features[type_str+'_duration_min'] = min(type_durations)

        percentil_calc = np.percentile(type_durations, [10, 20, 30, 40, 50, 60, 70, 80, 90])

        features[type_str+'_duration_percentil_10'] = percentil_calc[0]

        features[type_str+'_duration_percentil_20'] = percentil_calc[1]

        features[type_str+'_duration_percentil_30'] = percentil_calc[2]

        features[type_str+'_duration_percentil_40'] = percentil_calc[3]

        features[type_str+'_duration_percentil_50'] = percentil_calc[4]

        features[type_str+'_duration_percentil_60'] = percentil_calc[5]

        features[type_str+'_duration_percentil_70'] = percentil_calc[6]

        features[type_str+'_duration_percentil_80'] = percentil_calc[7]

        features[type_str+'_duration_percentil_90'] = percentil_calc[8]

        for i in range(9):

            features[type_str+'_duration_relative_percentile_{}'.format(i)] = percentil_calc[i] - np.mean(type_durations)

        

        percentil_calc = np.percentile(type_durations, [1, 5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 99])

        features[type_str+'_duration_percentil_1'] = percentil_calc[0]

        features[type_str+'_duration_percentil_5'] = percentil_calc[1]

        features[type_str+'_duration_percentil_15'] = percentil_calc[2]

        features[type_str+'_duration_percentil_25'] = percentil_calc[3]

        features[type_str+'_duration_percentil_35'] = percentil_calc[4]

        features[type_str+'_duration_percentil_45'] = percentil_calc[5]

        features[type_str+'_duration_percentil_55'] = percentil_calc[6]

        features[type_str+'_duration_percentil_65'] = percentil_calc[7]

        features[type_str+'_duration_percentil_75'] = percentil_calc[8]

        features[type_str+'_duration_percentil_85'] = percentil_calc[9]

        features[type_str+'_duration_percentil_95'] = percentil_calc[10]

        features[type_str+'_duration_percentil_99'] = percentil_calc[11]

        for i in range(9,21):

            features[type_str+'_duration_relative_percentile_{}'.format(i)] = percentil_calc[i-9] - np.mean(type_durations)

        

    return features
def get_data(user_sample, test_set=False):

    last_activity = 0

    user_activities_count = {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}

    

    time_spent_each_act = {actv: 0 for actv in list_of_user_activities}

    event_code_count = {eve: 0 for eve in list_of_event_code}

    

    last_accuracy_title = {'acc_' + title: -1 for title in assess_titles}

    event_id_count: Dict[str, int] = {eve: 0 for eve in list_of_event_id}

    title_count: Dict[str, int] = {eve: 0 for eve in activities_labels.values()} 

    title_event_code_count: Dict[str, int] = {t_eve: 0 for t_eve in all_title_event_code}

    

    accuracy_groups = {0:0, 1:0, 2:0, 3:0}

    all_assessments = []

    accumulated_accuracy_group = 0

    accumulated_accuracy = 0

    accumulated_correct_attempts = 0 

    accumulated_uncorrect_attempts = 0 

    accumulated_actions = 0

    counter = 0

    assessment_durations = []

    

    features = {}

    

    for _, session in user_sample.groupby('game_session', sort=False):

        session_type = session['type'].iloc[0]

        session_title = session['title'].iloc[0]

        session_title_text = activities_labels[session_title]

        

        # get current session time in seconds

        if session_type != 'Assessment':

            time_spent = int(session['game_time'].iloc[-1] / 1000)

            time_spent_each_act[activities_labels[session_title]] += time_spent

        

        if (session_type == 'Assessment') & (test_set or len(session)>1):

            # search for event_code 4100, that represents the assessments trial

            all_attempts = session.query(f'event_code == {win_code[session_title]}')

            # then, check the numbers of wins and the number of losses

            true_attempts = all_attempts['event_data'].str.contains('true').sum()

            false_attempts = all_attempts['event_data'].str.contains('false').sum()

            # copy a dict to use as feature template, it's initialized with some itens: 

            # {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}

            features.update(user_activities_count.copy())

            features.update(time_spent_each_act.copy())

            

            features.update(event_code_count.copy())

            features.update(last_accuracy_title.copy())

            features.update(event_id_count.copy())

            features.update(title_count.copy())

            features.update(title_event_code_count.copy())

            

            # add title as feature, remembering that title represents the name of the game

            features['session_title'] = session_title

            features['session_world'] = session['world'].iloc[0]

            features['installation_id'] = session['installation_id'].iloc[-1]

            # the 4 lines below add the feature of the history of the trials of this player

            # this is based on the all time attempts so far, at the moment of this assessment

            features['accumulated_correct_attempts'] = accumulated_correct_attempts

            features['accumulated_uncorrect_attempts'] = accumulated_uncorrect_attempts

            accumulated_correct_attempts += true_attempts 

            accumulated_uncorrect_attempts += false_attempts

            # the time spent in the app so far

            ass_duration_fea = type_duration_features(assessment_durations, 'assessment')

            features.update(ass_duration_fea)

            assessment_durations.append((session.iloc[-1,2]-session.iloc[0,2]).seconds)

            

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

            

    if test_set:

        return all_assessments[-1], all_assessments[:-1]

    # in the train_set, all assessments goes to the dataset

    return all_assessments
compiled_data = []

for _, user_sample in tqdm(train.groupby('installation_id', sort=False), total=17000):

    compiled_data += get_data(user_sample)
new_train = pd.DataFrame(compiled_data)

del compiled_data

new_train.shape
# function that creates more features

def preprocess(df):

    df['installation_session_count'] = df.groupby(['installation_id'])['Clip'].transform('count')

    df['installation_duration_mean'] = df.groupby(['installation_id'])['assessment_duration_mean'].transform('mean')

    df['installation_duration_std'] = df.groupby(['installation_id'])['assessment_duration_mean'].transform('std')

    df['installation_title_nunique'] = df.groupby(['installation_id'])['session_title'].transform('nunique')



    df['sum_event_code_count'] = df[[2050, 4100, 4230, 5000, 4235, 2060, 4110, 5010, \

                                     2070, 2075, 2080, 2081, 2083, 3110, 4010, 3120, 3121, 4020, 4021, 

                                    4022, 4025, 4030, 4031, 3010, 4035, 4040, 3020, \

                                     3021, 4045, 2000, 4050, 2010, 2020, 4070, 2025, 2030, 4080, 2035, 

                                    2040, 4090, 4220, 4095]].sum(axis = 1)



    df['installation_event_code_count_mean'] = df.groupby(['installation_id'])['sum_event_code_count'].transform('mean')

    df['installation_event_code_count_std'] = df.groupby(['installation_id'])['sum_event_code_count'].transform('std')

        

    return df
new_train = preprocess(new_train)

new_train.shape
all_features = [x for x in new_train.columns if x not in ['accuracy_group', 'installation_id', 4070, 'Clip']]

cat_features = ['session_title','session_world']

X, y = new_train[all_features], new_train['accuracy_group']
X_train, y_train = X, y
import lightgbm as lgb

from bayes_opt import BayesianOptimization

import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold
nfold = 5

skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=42)
rate_0 = train_labels.query("accuracy_group==0")['accuracy_group'].count() / len(train_labels)

rate_1 = train_labels.query("accuracy_group==1")['accuracy_group'].count() / len(train_labels)

rate_2 = train_labels.query("accuracy_group==2")['accuracy_group'].count() / len(train_labels)

rate_3 = train_labels.query("accuracy_group==3")['accuracy_group'].count() / len(train_labels)
new_test = []

test_history = []

for _, user_sample in tqdm(test.groupby('installation_id', sort=False), total=1000):

    a, history = get_data(user_sample, test_set=True)

    new_test.append(a)

    test_history += history

X_test = pd.DataFrame(new_test)

test_his = pd.DataFrame(test_history)

X_test = preprocess(X_test.append(test_his)).iloc[:len(X_test)]

X_test = X_test.loc[:, all_features]
def LGB_bayesian(**var_dict):



    var_dict['num_leaves'] = int(var_dict['num_leaves'])

    var_dict['min_data_in_leaf'] = int(var_dict['min_data_in_leaf'])

    var_dict['max_depth'] = int(var_dict['max_depth'])



    assert type(var_dict['num_leaves']) == int

    assert type(var_dict['min_data_in_leaf']) == int

    assert type(var_dict['max_depth']) == int

    

    param = {

              'num_leaves': var_dict['num_leaves'], 

              'min_child_samples': var_dict['min_child_samples'], 

              'min_data_in_leaf': var_dict['min_data_in_leaf'],

              'min_child_weight': var_dict['min_child_weight'],

              'bagging_fraction' : var_dict['bagging_fraction'],

              'feature_fraction' : var_dict['feature_fraction'],

              'learning_rate' : var_dict['learning_rate'],

              'subsample': var_dict['subsample'], 

              'max_depth': var_dict['max_depth'],

              'colsample_bytree': var_dict['colsample_bytree'],

              'reg_alpha': var_dict['reg_alpha'],

              'reg_lambda': var_dict['reg_lambda'],

              'objective': 'multiclass',

              'save_binary': True,

              'seed': 1337,

              'feature_fraction_seed': 1337,

              'bagging_seed': 1337,

              'drop_seed': 1337,

              'data_random_seed': 1337,

              'boosting_type': 'gbdt',

              'verbose': 1,

              'is_unbalance': True,

              'boost_from_average': True,

              'metric':'multi_logloss',

              'num_class': 4,

              'device': 'cpu'}    

    

    oof = np.zeros((len(X_train),4))

    score = 0

    for train_idx, valid_idx in skf.split(X_train, y_train.values):

        trn_data = lgb.Dataset(X_train.loc[train_idx, all_features].values,

                                       label=y_train.loc[train_idx].values

                                       )

        val_data = lgb.Dataset(X_train.loc[valid_idx, all_features].values,

                                       label=y_train.loc[valid_idx].values

                                       ) 

        clf = lgb.train(param, trn_data,  num_boost_round=50, valid_sets = [trn_data, val_data], verbose_eval=0, \

                        early_stopping_rounds = 50)

    

        oof[valid_idx]  = clf.predict(X_train.loc[valid_idx, all_features].values, \

                                      num_iteration=clf.best_iteration)

        

    result = np.argmax(oof, axis=1)

    score = qwk(y_train.values, result)

    

    trn_data = lgb.Dataset(X_train.loc[:,all_features].values, label=y_train.values)

    clf = lgb.train(param, trn_data, num_boost_round=50, verbose_eval=0)

    

    preds = np.round(np.argmax(clf.predict(X_test.values), axis=1)).astype('int')

    

    return score - ( 

                        abs(preds.tolist().count(0)/1000 - rate_0) \

                      + abs(preds.tolist().count(1)/1000 - rate_1) \

                      + abs(preds.tolist().count(2)/1000 - rate_2) \

                      + abs(preds.tolist().count(3)/1000 - rate_3)

                    )
# Bounded region of parameter space

bounds_LGB = {

    'num_leaves': (10, 900), 

    'min_data_in_leaf': (2, 900),

    'bagging_fraction' : (0.1, 0.99999999),

    'feature_fraction' : (0.0001, 0.99),

    'learning_rate': (0.01, 0.99999999),

    'min_child_weight': (0.000001, 0.05),   

    'min_child_samples':(10, 900), 

    'subsample': (0.001, 0.8),

    'colsample_bytree': (0.0001, 0.99), 

    'reg_alpha': (0.1, 3), 

    'reg_lambda': (1, 5),

    'max_depth': (-2, 90),

}
LGB_BO = BayesianOptimization(LGB_bayesian, bounds_LGB, random_state=42)
print('-' * 30)

with warnings.catch_warnings():

    warnings.filterwarnings('ignore')

    LGB_BO.maximize(init_points=10, n_iter=20)
LGB_BO.max['target']
LGB_BO.max['params']
BO = {'bagging_fraction': 0.99999999,

 'colsample_bytree': 0.99,

 'feature_fraction': 0.99,

 'learning_rate': 0.99999999,

 'max_depth': 90,

 'min_child_samples': 10.0,

 'min_child_weight': 0.05,

 'min_data_in_leaf': 351,

 'num_leaves': 900,

 'reg_alpha': 3.0,

 'reg_lambda': 5.0,

 'subsample': 0.8}
param_lgb = {

        'objective': 'multiclass',

        'save_binary': True,

        'seed': 1337,

        'feature_fraction_seed': 1337,

        'bagging_seed': 1337,

        'drop_seed': 1337,

        'data_random_seed': 1337,

        'boosting_type': 'gbdt',

        'verbose': 1,

        'is_unbalance': True,

        'boost_from_average': True,

        'metric':'multi_logloss',

        'num_class': 4,

        'device': 'CPU'

    }

param_lgb.update(BO)
# CV 

oof = np.zeros((len(X_train),4))

oof_train = np.zeros((len(X_train),4))

score = 0

feature_importance_df = pd.DataFrame()

for fold, (train_idx, valid_idx) in enumerate(skf.split(X_train, y_train.values)):

    print('-'*30)

    print(f'fold {fold+1}')

    print('-'*30)

    trn_data = lgb.Dataset(X_train.loc[train_idx, all_features].values,

                                   label=y_train.loc[train_idx].values

                                   )

    val_data = lgb.Dataset(X_train.loc[valid_idx, all_features].values,

                                   label=y_train.loc[valid_idx].values

                                   ) 

    clf = lgb.train(param_lgb, trn_data,  num_boost_round=50, valid_sets = [trn_data, val_data], verbose_eval=10, \

                    early_stopping_rounds = 50)



    oof[valid_idx]  = clf.predict(X_train.loc[valid_idx, all_features].values, \

                                  num_iteration=clf.best_iteration)

    oof_train[train_idx] += clf.predict(X_train.loc[train_idx, all_features].values, \

                                  num_iteration=clf.best_iteration) / (nfold-1)

    

    # Features imp

    fold_importance_df = pd.DataFrame()

    fold_importance_df["Feature"] = all_features

    fold_importance_df["importance"] = clf.feature_importance()

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    

result_train = np.argmax(oof_train, axis=1)    

score_train = qwk(y_train.values, result_train)

result = np.argmax(oof, axis=1)

score = qwk(y_train.values, result)

print(f'cv train qwk: {score_train}')

print(f'cv valid qwk: {score}')
# featrue importance plot 

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('dark_background')

plt.figure(figsize=(10,10))

sns.barplot(x="importance", y="Feature", 

            data=pd.DataFrame(feature_importance_df.groupby('Feature')['importance'].mean().sort_values(ascending=False))\

            .reset_index().iloc[:30],

        edgecolor=('white'), linewidth=2, palette="rocket")

plt.title('LGB Features importance (averaged/folds)', fontsize=18)

plt.tight_layout()



# useful features : 4035, 'session_title', 'Sandcastle Builder (Activity)', 2000, 4025, 4030, 4020, \

# 'assessment_duration_last', 'accumulated_actions', 'Chow Time', 4040, 3120, 'session_world', 'installation_session_count', \

# 'installation_duration_mean'
# train on all data once

trn_data = lgb.Dataset(X_train.loc[:,all_features].values, label=y_train.values)

clf = lgb.train(param_lgb, trn_data, num_boost_round=50, verbose_eval=0)
preds = np.argmax(clf.predict(X_test.values), axis=1)

preds.shape
submission['accuracy_group'] = np.round(preds).astype('int')

submission.to_csv('submission.csv', index=None)

submission.head()
submission['accuracy_group'].plot(kind='hist')
train_labels['accuracy_group'].plot(kind='hist')
pd.Series(result).plot(kind='hist')