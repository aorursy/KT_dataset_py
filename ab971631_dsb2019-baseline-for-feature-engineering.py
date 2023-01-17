import numpy as np

import pandas as pd

import os

import gc

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold

import lightgbm as lgb

from numba import jit 

#some essential libraries for data processing!

import numpy as np

import pandas as pd 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



#some for nice output descriptions

from time import time 

from tqdm import tqdm_notebook as tqdm

from collections import Counter

 

from scipy import stats #for some statistical processing



#models which we gonna use in this notebook

from xgboost import XGBClassifier

from sklearn.multiclass import OneVsRestClassifier



#kappa scoring with quadratic weights

from sklearn.metrics import cohen_kappa_score



#some pretty common validation stratigies.

from sklearn.model_selection import KFold, StratifiedKFold



#you already know it all champ!

import gc

import json

pd.set_option('display.max_columns', 1000)
# Read in the data CSV files

train = pd.read_csv('../input/data-science-bowl-2019/train.csv')

train_labels = pd.read_csv('../input/data-science-bowl-2019/train_labels.csv')

test = pd.read_csv('../input/data-science-bowl-2019/test.csv')

specs = pd.read_csv('../input/data-science-bowl-2019/specs.csv')

ss = pd.read_csv('../input/data-science-bowl-2019/sample_submission.csv')
def encode_title(train, test, train_labels):

    # encode title

    # make a list with all the unique 'titles' from the train and test set

    list_of_user_activities = list(set(train['title'].unique()).union(set(test['title'].unique())))

    # make a list with all the unique 'event_code' from the train and test set

    list_of_event_code = list(set(train['event_code'].unique()).union(set(test['event_code'].unique())))

    # make a list with all the unique worlds from the train and test set

    list_of_worlds = list(set(train['world'].unique()).union(set(test['world'].unique())))

    # create a dictionary numerating the titles

    activities_map = dict(zip(list_of_user_activities, np.arange(len(list_of_user_activities))))

    activities_labels = dict(zip(np.arange(len(list_of_user_activities)), list_of_user_activities))

    activities_world = dict(zip(list_of_worlds, np.arange(len(list_of_worlds))))

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

    return train, test, train_labels
# get usefull dict with maping encode

# train1, test1, train_labels1 = encode_title(train, test, train_labels)
# train1.head()
def extract_time_features(df):

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df['date'] = df['timestamp'].dt.date

    df['month'] = df['timestamp'].dt.month

    df['hour'] = df['timestamp'].dt.hour

    df['year'] = df['timestamp'].dt.year

    df['dayofweek'] = df['timestamp'].dt.dayofweek

    df['weekofyear'] = df['timestamp'].dt.weekofyear

    df['dayofyear'] = df['timestamp'].dt.dayofyear

    df['quarter'] = df['timestamp'].dt.quarter

    df['is_month_start'] = df['timestamp'].dt.is_month_start    

    

    return df

    

def get_object_columns(df, columns):

    df = df.groupby(['installation_id', columns])['event_id'].count().reset_index()

    df = df.pivot_table(index = 'installation_id', columns = [columns], values = 'event_id')

    df.columns = list(df.columns)

    df.fillna(0, inplace = True)

    return df



def get_numeric_columns(df, column):

    df = df.groupby('installation_id').agg({f'{column}': ['mean', 'sum', 'min', 'max', 'std', 'skew']})

    df[column].fillna(df[column].mean(), inplace = True)

    df.columns = [f'{column}_mean', f'{column}_sum', f'{column}_min', f'{column}_max', f'{column}_std', f'{column}_skew']

    return df



def get_numeric_columns_add(df, agg_column, column):

    df = df.groupby(['installation_id', agg_column]).agg({f'{column}': ['mean', 'sum', 'min', 'max', 'std', 'skew']}).reset_index()

    df = df.pivot_table(index = 'installation_id', columns = [agg_column], values = [col for col in df.columns if col not in ['installation_id', 'type']])

    df[column].fillna(df[column].mean(), inplace = True)

    df.columns = list(df.columns)

    return df
# result1.shape,test1.shape
def perform_features_engineering(train_df, test_df, train_labels_df):

    print(f'Perform features engineering')

    numerical_columns = ['game_time']

    categorical_columns = ['type', 'world']



    comp_train_df = pd.DataFrame({'installation_id': train_df['installation_id'].unique()})

    comp_train_df.set_index('installation_id', inplace = True)

    comp_test_df = pd.DataFrame({'installation_id': test_df['installation_id'].unique()})

    comp_test_df.set_index('installation_id', inplace = True)



    test_df = extract_time_features(test_df)

    train_df = extract_time_features(train_df)



    for i in numerical_columns:

        comp_train_df = comp_train_df.merge(get_numeric_columns(train_df, i), left_index = True, right_index = True)

        comp_test_df = comp_test_df.merge(get_numeric_columns(test_df, i), left_index = True, right_index = True)

    

    for i in categorical_columns:

        comp_train_df = comp_train_df.merge(get_object_columns(train_df, i), left_index = True, right_index = True)

        comp_test_df = comp_test_df.merge(get_object_columns(test_df, i), left_index = True, right_index = True)

    

#     for i in categorical_columns:

#         for j in numerical_columns:

#             comp_train_df = comp_train_df.merge(get_numeric_columns_add(train_df, i, j), left_index = True, right_index = True)

#             comp_test_df = comp_test_df.merge(get_numeric_columns_add(test_df, i, j), left_index = True, right_index = True)

    

    

    comp_train_df.reset_index(inplace = True)

    comp_test_df.reset_index(inplace = True)

    

    print('Our training set have {} rows and {} columns'.format(comp_train_df.shape[0], comp_train_df.shape[1]))



    # get the mode of the title

    labels_map = dict(train_labels_df.groupby('title')['accuracy_group'].agg(lambda x:x.value_counts().index[0]))

    # merge target

    labels = train_labels_df[['installation_id', 'title', 'accuracy_group']]

    # replace title with the mode

    labels['title'] = labels['title'].map(labels_map)

    # get title from the test set

    comp_test_df['title'] = test_df.groupby('installation_id').last()['title'].map(labels_map).reset_index(drop = True)

    # join train with labels

    comp_train_df = labels.merge(comp_train_df, on = 'installation_id', how = 'left')

    print('We have {} training rows'.format(comp_train_df.shape[0]))

    

    return comp_train_df, comp_test_df
@jit

def qwk3(a1, a2, max_rat=3):

    assert(len(a1) == len(a2))

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

def run_model(comp_train_df, comp_test_df):

    print(f'Run model')

    num_splits = 9

    kf = KFold(n_splits=num_splits)

    features = [i for i in comp_train_df.columns if i not in ['accuracy_group', 'installation_id']]

    target = 'accuracy_group'

    oof_pred = np.zeros((len(comp_train_df), 4))

    y_pred = np.zeros((len(comp_test_df), 4))

    for fold, (tr_ind, val_ind) in enumerate(kf.split(comp_train_df)):

        print(f'Fold: {fold+1}')

        x_train, x_val = comp_train_df[features].iloc[tr_ind], comp_train_df[features].iloc[val_ind]

        y_train, y_val = comp_train_df[target][tr_ind], comp_train_df[target][val_ind]

        train_set = lgb.Dataset(x_train, y_train)

        val_set = lgb.Dataset(x_val, y_val)



        params = {

            'learning_rate': 0.007,

            'metric': 'multiclass',

            'objective': 'multiclass',

            'num_classes': 4,

            'feature_fraction': 0.45,

            "bagging_fraction": 0.8,

            "bagging_seed": 22,

        }



        model = lgb.train(params, train_set, num_boost_round = 10000, early_stopping_rounds = 100, 

                          valid_sets=[train_set, val_set], verbose_eval = 100)

        oof_pred[val_ind] = model.predict(x_val)

        y_pred += model.predict(comp_test_df[features]) / num_splits

        

        val_crt_fold = qwk3(y_val, oof_pred[val_ind].argmax(axis = 1))

        print(f'Fold: {fold+1} quadratic weighted kappa score: {np.round(val_crt_fold,4)}')

        

    res = qwk3(comp_train_df['accuracy_group'], oof_pred.argmax(axis = 1))

    print(f'Quadratic weighted score: {np.round(res,4)}')

        

    return y_pred



def prepare_submission(comp_test_df, sample_submission_df, y_pred):

    comp_test_df = comp_test_df.reset_index()

    comp_test_df = comp_test_df[['installation_id']]

    comp_test_df['accuracy_group'] = y_pred.argmax(axis = 1)

    sample_submission_df.drop('accuracy_group', inplace = True, axis = 1)

    sample_submission_df = sample_submission_df.merge(comp_test_df, on = 'installation_id')

    sample_submission_df.to_csv('submission.csv', index = False)
comp_train_df, comp_test_df = perform_features_engineering(train, test, train_labels)
all_features = [x for x in comp_train_df.columns if x not in ['accuracy_group',"installation_id"]]

# cat_features = ['session_title']

X, y = comp_train_df[all_features], comp_train_df['accuracy_group']

def make_classifier():

    clf = CatBoostClassifier(

                               loss_function='MultiClass',

    #                            eval_metric="AUC",

                               task_type="CPU",

                               learning_rate=0.01,

                               iterations=5000,

                               od_type="Iter",

#                                depth=8,

                               early_stopping_rounds=500,

    #                            l2_leaf_reg=1,

    #                            border_count=96,

                               random_seed=2019

                              )

        

    return clf

oof = np.zeros(len(X))
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=2)

X, y = sm.fit_sample(X, y.ravel())
# CV

from catboost import CatBoostClassifier

from sklearn.model_selection import KFold

# preds = np.zeros(len(X_test))

# train model on all data once

clf = make_classifier()

clf.fit(X, y, verbose=500)



# del X, y


# make predictions on test set once

preds = clf.predict(comp_test_df[all_features])

# del X_test
ss['accuracy_group'] = np.round(preds).astype('int')

ss.to_csv('submission.csv', index=None)

ss.head()
ss['accuracy_group'].plot(kind='hist')

ss['accuracy_group'].value_counts(normalize=True)
# # Missing data analysis

# totalt = train.isnull().sum().sort_values(ascending=False)

# percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

# missing_data = pd.concat([totalt, percent], axis=1, keys=['Total', 'Percent'])

# print ("Missing Data at Training")

# missing_data
# train_labels.accuracy_group.value_counts().plot("bar",title='Target (accuracy_group)')

# plt.show()
# for col in test1.columns:

#     print("Column name:", col)

#     print("Unique values--->",test1[col].nunique())
# for col in result1.columns:

#     print("Column Name:", col)

#     print("Unique values--->", result1[col].nunique())




# result = pd.merge(train_, specs, how='left', on='event_id')

# result1 = pd.merge(result, train_labels, how='inner', on=['game_session', 'installation_id'])