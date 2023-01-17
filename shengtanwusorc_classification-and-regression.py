# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import copy

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from tqdm import tqdm_notebook

from sklearn.preprocessing import StandardScaler

from sklearn.svm import NuSVR, SVR

from sklearn.metrics import mean_absolute_error

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



from sklearn.model_selection import GroupKFold

from typing import Any

from numba import jit

import lightgbm as lgb

import xgboost as xgb

from sklearn import metrics

from itertools import product

import copy

import time

import os

import datetime

from time import time

from tqdm import tqdm_notebook as tqdm

from collections import Counter

from scipy import stats

from bayes_opt import BayesianOptimization

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import warnings

warnings.filterwarnings('ignore')

        

from IPython.core.display import display, HTML

display(HTML("<style>div.output_scroll { height: 88em; }</style>"))     

# Any results you write to the current directory are saved as output.



# set maximum rows to show

pd.set_option('display.max_rows', 1000)

pd.set_option('display.max_columns', 1000)
# helper function for coefficient rounder

from functools import partial

import scipy as sp

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

        X_p = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3])



        return -qwk(y, X_p)



    def fit(self, X, y):

        """

        Optimize rounding thresholds

        

        :param X: The raw predictions

        :param y: The ground truth labels

        """

        loss_partial = partial(self._kappa_loss, X=X, y=y)

        initial_coef = [0.5, 1.5, 2.5]

        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')



    def predict(self, X, coef):

        """

        Make predictions with specified thresholds

        

        :param X: The raw predictions

        :param coef: A list of coefficients that will be used for rounding

        """

        return pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3])





    def coefficients(self):

        """

        Return the optimized coefficients

        """

        return self.coef_['x']
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



def data_stats(df):

    df.describe()

    for column in df.columns:

        print('-'*10)

        print(column +': ' + str(df[column].nunique()) + ' unique values.')

        print(column + ' contains: ' + str(df[column].isnull().sum()) + ' na entries')

    print('-'*50)
train, test, train_labels, specs, sample_submission = read_data()
def encode_labels(train, test, train_labels):

        # keep installation ids with assessment taken only

    keep_id = train[train.type == "Assessment"][['installation_id']].drop_duplicates()

    train = pd.merge(train, keep_id, on="installation_id", how="inner") 

    del keep_id

    print('Totally ' + str(train['installation_id'].nunique()) +' players.')

    # get title set and map title

    train['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), train['title'], train['event_code']))

    test['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), test['title'], test['event_code']))

    title_event_code_set = list(set(train['title_event_code'].unique()).union(set(test['title_event_code'].unique())))

    title_set = list(set(train['title'].unique()).union(set(test['title'].unique())))

    title_map = dict(zip(title_set, np.arange(len(title_set))))

    title_labels =  dict(zip(np.arange(len(title_set)), title_set))

    # label encoding titles

    train['title'] = train['title'].map(title_map)

    test['title'] = test['title'].map(title_map)

    train_labels['title'] = train_labels['title'].map(title_map)

    # get event code and event id set.

    event_code_set = list(set(train['event_code'].unique()).union(set(test['event_code'].unique())))

    event_id_set = list(set(train['event_id'].unique()).union(set(test['event_id'].unique())))

    # get world_set and map world

    world_set = list(set(train['world'].unique()).union(set(test['world'].unique())))

    world_map = dict(zip(world_set, np.arange(len(world_set))))

    assess_title_set = list(set(train[train['type'] == 'Assessment']['title'].unique()).union(set(test[test['type'] == 'Assessment']['title'].unique())))

    # label encoding world 

    train['world'] = train['world'].map(world_map)

    test['world'] = test['world'].map(world_map)

    # get attempts labels, all attempts are encoded as 4100

    win_code = dict(zip(title_map.values(), (4100*np.ones(len(title_map))).astype('int')))

    # then, it set one element, the 'Bird Measurer (Assessment)' as 4110, 10 more than the rest

    win_code[title_map['Bird Measurer (Assessment)']] = 4110

    # transform timestamp to datetime type

    train['timestamp'] = pd.to_datetime(train['timestamp'])

    test['timestamp'] = pd.to_datetime(test['timestamp'])

    return train, test, train_labels, event_code_set, event_id_set, title_labels, assess_title_set,title_event_code_set, win_code
train, test, train_labels, event_code_set, event_id_set, title_labels, assess_title_set, title_event_code_set, attempt_code = encode_labels(train, test, train_labels)
train
def get_group(accuracy):

    if accuracy <= 0:

        return 0

    elif accuracy >=1:

        return 3

    elif accuracy >= 0.5:

        return 2

    else:

        return 1

    

def init_user_activities_features():

    user_act_feats = {}

    user_act_feats['user_activities_count'] = {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}

    user_act_feats['time_spent_activities'] = {'ClipTime':0, 'ActivityTime':0, 'AssessmentTime':0, 'GameTime':0}

    user_act_feats['career_activities_count'] = {'CareerClip':0, 'CareerActivity': 0, 'CareerAssessment': 0, 'CareerGame':0}

    user_act_feats['career_time_spent_activities'] = {'CareerClipTime':0, 'CareerActivityTime':0, 'CareerAssessmentTime':0, 'CareerGameTime':0}

    user_act_feats['career_avg_time_spent_activities'] = {'CareerAVGClipTime':0,

                                        'CareerAVGActivityTime':0,

                                        'CareerAVGAssessmentTime':0,

                                        'CareerAVGGameTime':0}

    return user_act_feats



def init_event_features():

    event_feats = {}

    event_feats['event_code_count'] = {ev: 0 for ev in event_code_set}

    event_feats['event_id_count'] = {eve:0 for eve in event_id_set}

    event_feats['title_count'] = {eve:0 for eve in title_labels.values()}

    event_feats['title_event_code_count'] ={t_eve: 0 for t_eve in title_event_code_set}

    event_feats['last_accuracy_title'] = {'acc_' + title: -1 for title in [title_labels[i] for i in assess_title_set]}

    return event_feats



def update_user_activities_features(features, user_act_feats):

    features.update(user_act_feats['career_activities_count'].copy()) 

    features.update(user_act_feats['career_time_spent_activities'].copy())

    

def update_event_features(features, event_feats):

    features.update(event_feats['event_code_count'].copy())

    features.update(event_feats['event_id_count'].copy())

    features.update(event_feats['title_count'].copy())

    features.update(event_feats['title_event_code_count'].copy())

    features.update(event_feats['last_accuracy_title'].copy())

    

def get_data(user_sample, test_set = None, prior_accuracy = None):

    # initialize 5 dicts of user activities features

    user_act_feats = init_user_activities_features()

    event_feats = init_event_features()

    # initialize counters ans scalars

    accuracy_groups = {0:0, 1:0, 2:0, 3:0}

    accumulated_true_attempts = 0

    accumulated_false_attempts = 0

    accumulated_accuracy = 0

    # feature rows to return

    all_assessments = []

    # initialization flag

    init = 1

    

    # starting collecting by sessions. 

    for i, session in user_sample.groupby('game_session', sort = False):

        # get session info

        session_type, session_title  = session['type'].iloc[0], session['title'].iloc[0]

        session_title_text = title_labels[session_title]

        # get session elapse time 

        session_time = (session.iloc[-1,2] - session.iloc[0,2]).seconds

        

        # session_time filter

        #if session_time <= 0 or session_time > 36000:

            #if (test_set and session_type!= 'Assessment') or not test_set:

                #continue

                

        # init if a new cycle starts.

        if init == 1:

            features = user_act_feats['user_activities_count'].copy()

            features.update(user_act_feats['time_spent_activities'].copy())

            accuracy_groups = {0:0, 1:0, 2:0, 3:0}

            init = 0 # turn init off between regular sessions.

        features[session_type] +=1

        features[session_type + 'Time'] += session_time

        # update activities count, time, and avgtime

        user_act_feats['career_activities_count']['Career'+str(session_type)] +=1

        if session_type is not 'Assessment':

            user_act_feats['career_time_spent_activities']['Career' + str(session_type) + 'Time'] += session_time

        if user_act_feats['career_activities_count']['Career'+str(session_type)]!=0:

            user_act_feats['career_avg_time_spent_activities']['CareerAVG'+str(session_type) +'Time'] = user_act_feats['career_time_spent_activities']['Career' + str(session_type) + 'Time']/user_act_feats['career_activities_count']['Career'+str(session_type)]

        if (session_type == 'Assessment') & (test_set or len(session) >1):

            # add columns to features dictionry

            update_user_activities_features(features, user_act_feats)

            update_event_features(features, event_feats)

            features['installation_id'] = session['installation_id'].iloc[-1]

            features['session_title'] = session['title'].iloc[0]

            # update assessment time afterwards

            user_act_feats['career_time_spent_activities']['Career' + str(session_type) + 'Time'] += session_time

            # collect all attempts and calculate accuracy in current session

            all_attempts = session.query(f'event_code == {attempt_code[session_title]}')

            true_attempts = all_attempts['event_data'].str.contains('true').sum()

            false_attempts = all_attempts['event_data'].str.contains('false').sum()

            features['accumulated_true_attempts'] = accumulated_true_attempts

            features['accumulated_false_attempts'] = accumulated_false_attempts

            accumulated_true_attempts += true_attempts

            accumulated_false_attempts += false_attempts

            accuracy = true_attempts/(true_attempts + false_attempts) if (true_attempts + false_attempts) != 0 else 0

            features['accuracy'] = accuracy

            features['accuracy_group'] = get_group(accuracy)

            # calculated average accumulated accuracy so far

            num = user_act_feats['career_activities_count']['CareerAssessment']

            features['accumulated_accuracy'] = accumulated_accuracy/(num-1) if num > 1 else 0

            accumulated_accuracy += accuracy

            accuracy_groups[features['accuracy_group']] = 1

            # update one-hot encoded accuracy

            features.update(accuracy_groups)

            # only assessments with attempts will be included

            if test_set or true_attempts + false_attempts > 0:

                all_assessments.append(features)

            init = 1

        

        def update_counters(counter: dict, col: str):

            num_of_session_count = Counter(session[col])

            for k in num_of_session_count.keys():

                x = k

                if col == 'title':

                    x = title_labels[k]

                counter[x] += num_of_session_count[k]

            return counter

        

        event_feats['event_code_count'] = update_counters(event_feats['event_code_count'], "event_code")

        event_feats['event_id_count'] = update_counters(event_feats['event_id_count'], "event_id")

        event_feats['title_count'] = update_counters(event_feats['title_count'], 'title')

        event_feats['title_event_code_count'] = update_counters(event_feats['title_event_code_count'], 'title_event_code')

    

    # we pass in a test_assessment_accuracy collected from training set to update accumulated accuracy for cold starting users.

    

    if test_set:

        # if only one assessment taken

        if len(all_assessments) == 1:

            idx = all_assessments[-1]['session_title']

            if prior_accuracy:

                if idx in prior_accuracy.keys():

                    all_assessments[-1]['accumulated_accuracy'] = prior_accuracy[idx]

                else:

                    all_assessments[-1]['accumulated_accuracy'] = prior_accuracy[-1]

                    

        return all_assessments[-1]

    return all_assessments



def fix_json_colname(df):

    df.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in df.columns]

    

def get_train_test(train, test):

    compiled_train = []

    compiled_test = []

    

    for i, (ins_id, user_sample) in tqdm(enumerate(train.groupby('installation_id', sort = False)), total = train['installation_id'].nunique()):

        compiled_train.extend(get_data(user_sample,test_set = False, prior_accuracy = None))

    

    ''' prior_accuracy = {13.0: 0.24862535297870636,

                      21.0: 0.7422324419021606,

                      24.0: 0.7110419273376465,

                      27.0: 0.3873789310455322,

                      36.0: 0.7355366349220276,

                      -1: 0.564963}'''

    

    for ins_id, user_sample in tqdm(test.groupby('installation_id', sort = False), total = test['installation_id'].nunique()):

        test_data = get_data(user_sample, test_set = True, prior_accuracy = None)

        compiled_test.append(test_data)

    reduce_train = pd.DataFrame(compiled_train)

    reduce_test = pd.DataFrame(compiled_test)

    for df in [reduce_train, reduce_test]:

        for col in df.columns:

            if df[col].dtypes == np.int64 or df[col].dtypes == np.float64:

                df[col] = df[col].astype(np.float32)

    fix_json_colname(reduce_train)

    fix_json_colname(reduce_test)

    del train, test

    return reduce_train, reduce_test
reduce_train_r, reduce_test_r = get_train_test(train, test)
def remove_outlier(train, test, out_feat, q = 0.99):

    for feature in out_feat:

        if feature not in train.columns:

            return 'Feature not found in columns'

        upper = train[feature].quantile(q)

        train.loc[train[feature] > upper, [feature]] = upper

        test.loc[test[feature] >upper, [feature]] = upper
def preprocessing(reduce_train, reduce_test):

    '''

    preprocessing module

    

    '''

    for df in [reduce_train, reduce_test]:

        df['installation_session_count'] = df.groupby(['installation_id'])['Clip'].transform('count')

        # df['installation_duration_mean'] = df.groupby(['installation_id'])['duration_mean'].transform('mean')

        # df['installation_duration_std'] = df.groupby(['installation_id'])['duration_mean'].transform('std')

        # df['installation_title_nunique'] = df.groupby(['installation_id'])['session_title'].transform('nunique')

        df['sum_event_code_count'] = df[[str(x) for x in [2050, 4100, 4230, 5000, 4235, 2060, 4110, 5010, 2070, 2075, 2080, 2081, 2083, 3110, 4010, 3120, 3121, 4020, 4021, 

                                        4022, 4025, 4030, 4031, 3010, 4035, 4040, 3020, 3021, 4045, 2000, 4050, 2010, 2020, 4070, 2025, 2030, 4080, 2035, 

                                        2040, 4090, 4220, 4095]]].sum(axis = 1)

        df['installation_event_code_count_mean'] = df.groupby(['installation_id'])['sum_event_code_count'].transform('mean')



    features = reduce_train.loc[(reduce_train.sum(axis=1) != 0), (reduce_train.sum(axis=0) != 0)].columns

    features = [x for x in features if x not in [

                                                 'AssessmentTime',

                                                 'accumulated_true_attempts',

                                                 'accumulated_false_attempts',

                                                 'installation_id',

                                                 'accuracy',

                                                 'accuracy_group',

                                                  '0','1','2','3',

                                                 'binary_accuracy']]

    

    return reduce_train, reduce_test, features
reduce_train, reduce_test, features = preprocessing(reduce_train_r, reduce_test_r)
from imblearn.over_sampling import SMOTE

x = reduce_train[features].values

y = reduce_train['accuracy_group'].values

X_resampled, y_resampled = SMOTE(sampling_strategy = 'auto' ).fit_resample(x,y)

del reduce_train # save ram space
# new reduce train

reduce_train = pd.DataFrame(X_resampled, columns = features)

reduce_train['accuracy_group'] = y_resampled
reduce_train['accuracy_group'].hist()
#remove_outlier(reduce_train, reduce_test, ['ActivityTime', 'AssessmentTime', 'GameTime'], q = 0.99)
# helper functions

import sklearn

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.metrics import roc_auc_score, mean_squared_error as mse, f1_score

from sklearn.preprocessing import MinMaxScaler

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.model_selection import StratifiedKFold, KFold

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_validate

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier



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



    hist1 = np.zeros((max_rat + 1, ))

    hist2 = np.zeros((max_rat + 1, ))



    o = 0

    for k in range(a1.shape[0]):

        i, j = a1[k], a2[k]

        hist1[i] += 1

        hist2[j] += 1

        o +=  (i - j) * (i - j)

    e = 0

    for i in range(max_rat + 1):

        for j in range(max_rat + 1):

            e += hist1[i] * hist2[j] * (i - j) * (i - j)

    e = e / a1.shape[0]

    return 1 - o / e



def eval_qwk_lgb(y_true, y_pred):

    """

    Fast cappa eval function for lgb.

    """

    y_pred = y_pred.reshape(len(np.unique(y_true)), -1).argmax(axis=0)

    return 'cappa', qwk(y_true, y_pred), True



def get_cal_group(y_pred):

    coef = [1.12232214, 1.73925866, 2.22506454]

    #coef = [.1,.25,.75]

    y_pred[y_pred <= coef[0]] = 0

    y_pred[np.where(np.logical_and(y_pred > coef[0], y_pred <= coef[1]))] = 1

    y_pred[np.where(np.logical_and(y_pred > coef[1], y_pred <= coef[2]))] = 2

    y_pred[y_pred > coef[2]] = 3

    

def eval_qwk_lgb_regr(y_true, y_pred):

    """

    Fast cappa eval function for lgb.

    """

    get_cal_group(y_pred)

    # y_pred = y_pred.reshape(len(np.unique(y_true)), -1).argmax(axis=0)

    return 'cappa', qwk(y_true, y_pred), True



def plot_feature_importance(features, importances):

    d = {}

    d['features'] = features

    d['importance'] = importances

    important_features = pd.DataFrame(d)

    plt.figure(figsize = (12,15))

    important_features = important_features.groupby('features')['importance'].mean().reset_index().sort_values('importance')

    if len(features) < 80:

        n = len(features)-1

    else:

        n = 80

    sns.barplot(important_features['importance'][-n:], important_features['features'][-n:])

    return important_features



def feature_analysis(x, y):

    clf = RandomForestClassifier()

    clf.fit(x, y)

    plot_fi(features, clf.feature_importances_)

    return clf



def feature_selection(x_train, y_train, x_valid, y_valid, keep_feature = 200):

    '''

    feature selection module

    '''

    pass



class LGBWrapper_regr(object):

    """

    A wrapper for lightgbm model so that we will have a single api for various models.

    """



    def __init__(self):

        self.model = lgb.LGBMRegressor()



    def fit(self, X_train, y_train, X_valid=None, y_valid=None, X_holdout=None, y_holdout=None, params=None):

            

        if params['eval_metric']=='cappa':

            eval_metric = eval_qwk_lgb_regr

        if params['eval_metric'] == 'l2':

            eval_metric = 'l2'

        else:

            eval_metric = 'auc'

        

        eval_set = [(X_train, y_train)]

        eval_names = ['train']

        self.model = self.model.set_params(**params)

        

        if X_valid is not None:

            eval_set.append((X_valid, y_valid))

            eval_names.append('valid')



        if X_holdout is not None:

            eval_set.append((X_holdout, y_holdout))

            eval_names.append('holdout')

        

        

        if 'cat_cols' in params.keys():

            cat_cols = [col for col in params['cat_cols'] if col in X_train.columns]

            if len(cat_cols) > 0:

                categorical_columns = params['cat_cols']

            else:

                categorical_columns = 'auto'

        else:

            categorical_columns = 'auto'



        self.model.fit(X=X_train, y=y_train,

                       eval_set=eval_set, eval_names=eval_names, eval_metric=eval_metric,

                       verbose=params['verbose'], early_stopping_rounds=params['early_stopping_rounds'],

                       categorical_feature=categorical_columns)



        self.best_score_ = self.model.best_score_

        self.feature_importances_ = self.model.feature_importances_



    def predict(self, X_test):

        return self.model.predict(X_test, num_iteration=self.model.best_iteration_)



class LGBWrapper_clf(object):

    def __init__(self):

        self.model = lgb.LGBMClassifier()

    

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, X_holdout=None, y_holdout=None, params=None):

        eval_metric = eval_qwk_lgb

        eval_set = [(X_train, y_train)]

        eval_names = ['train']

        self.model = self.model.set_params(**params)

    

        if X_valid is not None:

            eval_set.append((X_valid, y_valid))

            eval_names.append('valid')



        if X_holdout is not None:

            eval_set.append((X_holdout, y_holdout))

            eval_names.append('holdout')

        if 'cat_cols' in params.keys():

            cat_cols = [col for col in params['cat_cols'] if col in X_train.columns]

            if len(cat_cols) > 0:

                categorical_columns = params['cat_cols']

            else:

                categorical_columns = 'auto'

        else:

            categorical_columns = 'auto'

            

        self.model.fit(X=X_train, y=y_train,

               eval_set=eval_set, eval_names=eval_names, eval_metric=eval_metric,

               verbose=params['verbose'], early_stopping_rounds=params['early_stopping_rounds'])

        self.best_score_ = self.model.best_score_

        self.feature_importances_ = self.model.feature_importances_

    def predict(self, X_test):

        return self.model.predict(X_test, num_iteration=self.model.best_iteration_)
#trash code

'''n_estimators = 2000,

   subsample = .75,

   subsample_freq = 1,

   learning_rate = .04,

   feature_fraction = .8,

   max_depth = 15,

   lambda_l1 = .5,

   lambda_l2 = .5,

   verbose = 1'''



'''

clf1 = lgb.LGBMClassifier()

clf2 = RandomForestClassifier()

clf3 = lgb.LGBMRegressor()

clf4 = RandomForestRegressor()

reg_cv(clf3, reduce_train[features], reduce_train['accuracy_group'], 5)'''
# basic cv module, returns feature importance list

def lgbm_cv(x, y, params, mtype = 'regr', n = 5):

    kf = KFold(n_splits = n, shuffle = True)

    c = 0

    r_list, kp_list, qwk_list = [], [], []

    for train_index, test_index in kf.split(x,y):

        c+=1

        if isinstance(x, np.ndarray):

            x_train, x_test = x[train_index, :], x[test_index,:]

            y_train, y_test = y[train_index], y[test_index]

        else:

            x_train, x_test = x.loc[train_index], x.loc[test_index]

            y_train, y_test = y.loc[train_index], y.loc[test_index]

        

        if mtype == 'clf':

            lgbm = LGBWrapper_clf()

        else:

            lgbm = LGBWrapper_regr()

            

        print('['+str(c)+']')

        print('-'*40)

        lgbm.fit(x_train, y_train, x_test, y_test, params = params)

        if mtype == 'regr':

            r_score, kp_score = lgbm.best_score_['valid']['rmse'],lgbm.best_score_['valid'][params['eval_metric']]

        else:

            r_score, kp_score = lgbm.best_score_['valid']['multi_logloss'],lgbm.best_score_['valid'][params['eval_metric']]

            

        if mtype == 'regr':    

            # fit the coefficients with training set

            y_pred = lgbm.predict(x_train)

            optR = OptimizedRounder()

            optR.fit(y_pred.reshape(-1), y_train)

            coefficients = optR.coefficients()

            # get predictions with test set and compute the qwk

            pr2 = lgbm.predict(x_test)

            opt_preds2 = optR.predict(pr2.reshape(-1,), coefficients)

            qwk_list.append(qwk(y_test, opt_preds2))

            print('qwk on test set:' + str(qwk(y_test, opt_preds2)))

            r_list.append(r_score)

            kp_list.append(kp_score)

            print('-'*40)



            print('Finished! rmse score: '

                  + str(np.array(r_list).mean())

                  + '+/-'

                  + str(round(np.array(r_list).std(), 2))

                  + 'std')



            print('Secondary score: '

                  + str(np.array(kp_list).mean())

                  + '+/-'

                  + str(round(np.array(kp_list).std(), 2))

                  + 'std')



            print('Average qwk score:' + str(np.array(qwk_list).mean()))

    important_features = plot_feature_importance(x_train.columns, lgbm.feature_importances_)

    return lgbm, important_features
n_estimators = 8000

regr_params = {'n_estimators': n_estimators,

            'boosting_type': 'gbdt',

            'objective': 'regression',

            'metric': 'rmse',

            'subsample': 0.75,

            'subsample_freq': 1,

            'learning_rate': 0.1,

            'feature_fraction': 1,

            'max_depth': 20,

            'lambda_l1': 1,  

            'lambda_l2': 1,

            'verbose': int(n_estimators/20),

            'early_stopping_rounds': 400,

            'eval_metric': 'cappa'

            }

clf_params = {

            'n_estimators':n_estimators,

            'boosting_type': 'gbdt',

            'objective': 'multiclass',

            'subsample': 0.75,

            'subsample_freq': 1,

            'learning_rate': 0.1,

            'feature_fraction': 0.9,

            'max_depth': 15,

            'lambda_l1': 1,  

            'lambda_l2': 1,

            'verbose': 100,

            'early_stopping_rounds': 300,

            'eval_metric': 'cappa'

}

#features = [f for f in features if f != 'session_title']



#lgbm, important_features = lgbm_cv(reduce_train[features], reduce_train['accuracy_group'], regr_params, 'regr', n = 10)

#lgbm, important_features = lgbm_cv(reduce_train[features], reduce_train['accuracy_group'], clf_params, 'clf', n = 10)
#features = [f for f in features if f != 'accumulated_accuracy']

#lgbm, important_features = lgbm_cv(reduce_train[features], reduce_train['accuracy_group'], regr_params, 'regr', n = 10)
from sklearn.feature_selection import VarianceThreshold, SelectFromModel

def feature_selection(reduce_train, params, pre_thresh = 20):

    X_train, X_test, y_train, y_test = train_test_split(reduce_train[features], reduce_train['accuracy_group'], test_size = .1)

    model = lgb.LGBMRegressor()

    model.set_params(**params)

    model.fit(X=X_train, y=y_train,

               eval_set=[(X_train, y_train), (X_test, y_test)], eval_names=['train', 'valid'], eval_metric='l2',

               verbose= 100, early_stopping_rounds=params['early_stopping_rounds'])

    threshold = np.unique(np.sort(model.feature_importances_))

    best_thresh = threshold[0]

    best_qwk = 0

    l = threshold.shape[0]

    if pre_thresh:

        selection = SelectFromModel(model, threshold = pre_thresh, prefit = True)

        return best_thresh, selection

        

    for thresh in threshold[int(l/4):int(l*3/4)]:

        selection = SelectFromModel(model, threshold=thresh, prefit = True)

        select_X_train = selection.transform(X_train)

        select_X_test = selection.transform(X_test)

        select_y_train = np.array(y_train)

        select_y_test = np.array(y_test)

        # train model

        kf = KFold(n_splits = 5, shuffle = True)

        qwk_list = []

        print('Thresh =' + str(thresh) +', n = ' + str(select_X_train.shape[1]))

        print('Start CV:')

        cv_round = 0

        best_thres = 0

        for train_index, test_index in kf.split(select_X_train, select_y_train):

            cv_round +=1 

            inner_X_train, inner_X_test = select_X_train[train_index][:], select_X_train[test_index][:]

            inner_y_train, inner_y_test = select_y_train[train_index], select_y_train[test_index]

            selection_model = lgb.LGBMRegressor()

            selection_model.set_params(**params)

            selection_model.fit(X=inner_X_train, y = inner_y_train,

                   eval_set=[(inner_X_train, inner_y_train), (inner_X_test, inner_y_test)], eval_names=['train', 'valid'], eval_metric='l2',

                   verbose = 500, early_stopping_rounds=params['early_stopping_rounds'])

            # optimize coefficient with y_pred

            y_pred = selection_model.predict(inner_X_train)

            optR = OptimizedRounder()

            optR.fit(y_pred.reshape(-1), inner_y_train)

            coefficients = optR.coefficients()

            # predictions made by inner_X_test

            pr2 = selection_model.predict(inner_X_test)

            opt_preds2 = optR.predict(pr2.reshape(-1,), coefficients)

            qwk_list.append(qwk(inner_y_test, opt_preds2))

            print('CV round: ' + str(cv_round))

        print('CV result: Average test qwk:' +str(np.mean(qwk_list)))

        print('-'*100)

        cur_qwk = np.mean(qwk_list)

        if cur_qwk > best_qwk:

            best_qwk = cur_qwk

            best_thresh = thresh

            best_selector = selection

    print('Best qwk: '+ str(best_qwk) + ', best thres:' + str(best_thresh))

    return best_thresh, best_selector
best_selector = None

#best_thres, best_selector = feature_selection(reduce_train, regr_params, pre_thresh = None)

#_, best_selector = feature_selection(reduce_train, regr_params, pre_thresh = 20) # prefit threshold
if best_selector:

    selected_X_train = best_selector.transform(reduce_train[features])

    selected_y_train = reduce_train['accuracy_group'].values

else:

    selected_X_train = reduce_train[features]

    selected_y_train = reduce_train['accuracy_group']
'''#clf3.fit(reduce_train[features], reduce_train['accuracy_group'])

pr1 = lgbm.predict(reduce_train[features])

optR = OptimizedRounder()

optR.fit(pr1.reshape(-1,), reduce_train['accuracy_group'])

coefficients = optR.coefficients()

opt_preds = optR.predict(pr1.reshape(-1,), coefficients)

qwk(reduce_train['accuracy_group'], )'''

''''''
# may be train 5 models and pick a best one?

def lgbm_reg_make_predictions(params, x, y, reduce_test):

    model = LGBWrapper_regr()

    x_train, x_test, y_train, y_test = train_test_split(reduce_train[features], reduce_train['accuracy_group'], test_size = .1)

    model.fit(x_train, y_train, x_test, y_test, params = params)

    pr = model.predict(x_train)

    # fit best thresholds

    optR = OptimizedRounder()

    optR.fit(pr.reshape(-1,), y_train)

    coefficients = optR.coefficients()

    # make final pred

    pr1 = model.predict(reduce_test[features])

    opt_preds = optR.predict(pr1.reshape(-1,), coefficients)

    # check pred ratio

    pr2 = model.predict(x_test)

    opt_preds2 = optR.predict(pr2.reshape(-1,), coefficients)

    print('-'*100)

    print('qwk on test set:')

    print(qwk(y_test, opt_preds2))

    return pr1, opt_preds



def lgbm_clf_make_predictions(params, x, y, reduce_test):

    model = LGBWrapper_clf()

    x_train, x_test, y_train, y_test = train_test_split(reduce_train[features], reduce_train['accuracy_group'], test_size = .1)

    model.fit(x_train, y_train, x_test, y_test, params = params)

    y_pred = model.predict(reduce_test[features])

    return y_pred

#pr1, opt_preds = lgbm_reg_make_predictions(regr_params, selected_X_train, selected_y_train, reduce_test)

pr1 = lgbm_clf_make_predictions(clf_params, selected_X_train, selected_y_train, reduce_test)
sample_submission['accuracy_group'] = pr1.astype(int)

sample_submission.to_csv('submission.csv', index=False)

sample_submission['accuracy_group'].value_counts(normalize=True)