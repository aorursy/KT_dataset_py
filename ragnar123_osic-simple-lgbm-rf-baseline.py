import os

import numpy as np

import pandas as pd

import random

import math

from tqdm import tqdm_notebook as tqdm

import lightgbm

import warnings

warnings.filterwarnings("ignore")

from sklearn import preprocessing

from sklearn.model_selection import GroupKFold

import scipy as sp

from functools import partial

import lightgbm as lgb
# define a seed

SEED = 222



## function to seed everything

def seed_everything(seed):

    random.seed(seed)

    np.random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)



# function to read and transform our training data

def read_and_transform_train():

    # read the train set

    train = pd.read_csv('/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv')

    # get the id (patient, week)

    train['Patient_Week'] = train['Patient'].astype(str) + '_' + train['Weeks'].astype(str)

    # for each user, we are going to cross each observation with each other

    # in other words we are using the given values for each user, to fill the other weeks

    # create a dataframe to store our final train set

    train_expanded = pd.DataFrame()

    # group by patient id

    patients = train.groupby('Patient')

    # iterate over each patient

    for _, user in tqdm(patients, total = len(patients)):

        # create a dataframe to store user data

        user_data = pd.DataFrame()

        # iterate over each week

        for week, week_data in user.groupby('Weeks'):

            # rename columns to usea initial values as features

            rename_cols = {

                'Weeks': 'base_Week', 

                'FVC': 'base_FVC', 

                'Percent': 'base_Percent', 

                'Age': 'base_Age'

            }

            week_data = week_data.drop(['Patient_Week'], axis = 1).rename(columns = rename_cols)

            # drop original values

            drop_cols = ['Percent', 'Age', 'Sex', 'SmokingStatus']

            user_ = user.drop(drop_cols, axis = 1).rename(columns = {'Weeks': 'predict_Week'})

            # merge all the user data with this specific week

            user_ = user_.merge(week_data, on = 'Patient')

            # calculate the week of difference between the initial values and the predicted week

            user_['diff_Week']  = user_['predict_Week'] - user_['base_Week']

            # add the data to the usar data

            user_data = pd.concat([user_data, user_], axis = 0)

        # add the user data to our final expanded dataset

        train_expanded = pd.concat([train_expanded, user_data])

    

    # filter out the same week

    train_expanded = train_expanded[train_expanded['diff_Week']!=0].reset_index(drop = True)

    return train_expanded



# function to read and transform our test data

def read_and_transform_test():

    # read the test csv

    test = pd.read_csv('/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv')

    # read the submission csv

    sub = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/sample_submission.csv")

    # rename the test column to be align with the train set

    rename_cols = {

                'Weeks': 'base_Week', 

                'FVC': 'base_FVC', 

                'Percent': 'base_Percent', 

                'Age': 'base_Age'

            }

    test.rename(columns = rename_cols, inplace = True)

    # get the patient id of the submissions

    sub['Patient'] = sub['Patient_Week'].apply(lambda x: x.split('_')[0])

    # get the prediction week

    sub['predict_Week'] = sub['Patient_Week'].apply(lambda x: x.split('_')[1]).astype(int)

    # merge the submission data with the test data (using the initial values)

    test = sub.drop(['FVC', 'Confidence'], axis = 1).merge(test, on = 'Patient', how = 'left')

    test['diff_Week'] = test['predict_Week'] - test['base_Week']

    return test



# function to preprocess data for lightgbm prediction

def preprocess_lgbm(train, test):

    for col in ['Sex', 'SmokingStatus']:

        encoder = preprocessing.LabelEncoder()

        train[col] = encoder.fit_transform(train[col])

        test[col] = encoder.transform(test[col])

    return train, test



# function to make a regresion using groupkfold because we are surely predicting unknown patients

def train_and_evaluate_lgbm(train, test, target, notarget):

    

    # define some random initial paramters

    params = {

        'boosting_type': 'rf',

        'metric': 'rmse',

        'objective': 'regression',

        'n_jobs': -1,

        'seed': SEED,

        'learning_rate': 0.1,

        'bagging_fraction': 0.8,

        'bagging_freq': 1,

    }

    

    # GroupKFold by Patient

    kf = GroupKFold(n_splits = 5)

    oof_pred = np.zeros(len(train))

    y_pred = np.zeros(len(test))

    features = [col for col in train.columns if col not in ['predict_Week', 'base_Week', 'Patient', target, 'Patient_Week', notarget, 'FVC_pred']]

    for fold, (tr_ind, val_ind) in enumerate(kf.split(train, groups = train['Patient'])):

        print(f'Training fold {fold + 1}')

        x_train, x_val = train[features].iloc[tr_ind], train[features].iloc[val_ind]

        y_train, y_val = train[target][tr_ind], train[target][val_ind]

        train_set = lgb.Dataset(x_train, y_train)

        val_set = lgb.Dataset(x_val, y_val)

        model = lgb.train(params, train_set, num_boost_round = 10000, early_stopping_rounds = 50, 

                          valid_sets = [train_set, val_set], verbose_eval = 50)

        oof_pred[val_ind] = model.predict(x_val)

        

        y_pred += model.predict(test[features]) / kf.n_splits

        

    return oof_pred, y_pred



# function to create the confidence label

def make_confidence_labels(train, oof_pred):

    # add oof predictions to the train set

    train['FVC_pred'] = oof_pred

    # define a optimization function to extract the optimal confidence

    def loss_func(weight, row):

        confidence = weight

        sigma_clipped = max(confidence, 70)

        diff = abs(row['FVC']- row['FVC_pred'])

        delta = min(diff, 1000)

        score = (-math.sqrt(2) * delta / sigma_clipped) - (np.log(math.sqrt(2) * sigma_clipped))

        return - score 

    

    # make a list to store our result

    results = []

    for ind, row in tqdm(train.iterrows(), total = len(train)):

        loss_partial = partial(loss_func, row = row)

        weight = [100]

        result = sp.optimize.minimize(loss_partial, weight, method = 'SLSQP')

        x = result['x']

        results.append(x[0])

        

    # add confidence to the train data

    train['Confidence'] = results

    train['sigma_clipped'] = train['Confidence'].apply(lambda x: max(x, 70))

    train['diff'] = abs(train['FVC'] - train['FVC_pred'])

    train['delta'] = train['diff'].apply(lambda x: min(x, 1000))

    train['score'] = (-math.sqrt(2) * train['delta'] / train['sigma_clipped']) - (np.log(math.sqrt(2) * train['sigma_clipped']))

    score = train['score'].mean()

    print(f'With our optimal confidence the laplace log likelihood is {score}')

    train.drop(['sigma_clipped', 'diff', 'delta', 'score'], axis = 1, inplace = True)

    return train



# function to calculate our out of folds laplace log likelihood

def calculate_out_of_folds(train, oof_pred):

    train['Confidence'] = oof_pred

    train['sigma_clipped'] = train['Confidence'].apply(lambda x: max(x, 70))

    train['diff'] = abs(train['FVC'] - train['FVC_pred'])

    train['delta'] = train['diff'].apply(lambda x: min(x, 1000))

    train['score'] = (-math.sqrt(2) * train['delta'] / train['sigma_clipped']) - (np.log(math.sqrt(2) * train['sigma_clipped']))

    score = train['score'].mean()

    print(f'Our out of folds laplace log likelihood is {score}')
# seed everything for deterministic results

seed_everything(SEED)

# read and transform our train data

train = read_and_transform_train()

# read and transform our tetst data

test = read_and_transform_test()

# preprocess our train and test data

train, test = preprocess_lgbm(train, test)

# train FVC and get out of folds and test predictions

oof_pred, y_pred = train_and_evaluate_lgbm(train, test, 'FVC', 'Confidence')

# save FVC predictions

test['FVC'] = y_pred

# now that we have FVC prediction we can calculate our confidence

print('-'* 50)

print('\n')

train = make_confidence_labels(train, oof_pred)

print('-'* 50)

print('\n')

# train Confidence and get out of folds and test predictions

oof_pred, y_pred = train_and_evaluate_lgbm(train, test, 'Confidence', 'FVC')

print('-'* 50)

print('\n')

# calculate our out of folds metric

calculate_out_of_folds(train, oof_pred)

print('-'* 50)

print('\n')

# add Confidence to the test set

test['Confidence'] = y_pred

# save submissions

test[['Patient_Week', 'FVC', 'Confidence']].to_csv('submission.csv', index = False)

test[['Patient_Week', 'FVC', 'Confidence']].head()