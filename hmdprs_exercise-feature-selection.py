# set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.feature_engineering.ex4 import *

print('Setup code-checing is completed!')



# load baseline data

import pandas as pd

clicks = pd.read_parquet('../input/feature-engineering-data/baseline_data.pqt')



# load features & join to baseline data

data_files = ['count_encodings.pqt',

              'catboost_encodings.pqt',

              'interactions.pqt',

              'past_6hr_events.pqt',

              'downloads.pqt',

              'time_deltas.pqt',

              'svd_encodings.pqt']

data_root = '../input/feature-engineering-data'

import os

for file in data_files:

    features = pd.read_parquet(os.path.join(data_root, file))

    clicks = clicks.join(features)



def get_data_splits(dataframe, valid_fraction=0.1):

    # sort

    dataframe = dataframe.sort_values('click_time')

    

    # split

    valid_rows = int(len(dataframe) * valid_fraction)

    train = dataframe[:-valid_rows * 2]

    valid = dataframe[-valid_rows * 2:-valid_rows]

    test = dataframe[-valid_rows:]

    

    return train, valid, test



import numpy as np

from sklearn import preprocessing, metrics

import lightgbm as lgb



def train_model(train, valid, test=None, feature_cols=None):

    # define featurs

    if feature_cols is None:

        feature_cols = train.columns.drop(['click_time', 'attributed_time', 'is_attributed'])

    

    # define train & valid datasets

    dtrain = lgb.Dataset(train[feature_cols], label=train['is_attributed'])

    dvalid = lgb.Dataset(valid[feature_cols], label=valid['is_attributed'])

    

    # fit model

    param = {'num_leaves': 64, 'objective': 'binary', 'metric': 'auc', 'seed': 7}

    num_round = 1000

    print("Training model!")

    bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid], 

                    early_stopping_rounds=20, verbose_eval=False)

    

    # make predictions

    valid_pred = bst.predict(valid[feature_cols])

    

    # evalute the model

    valid_score = metrics.roc_auc_score(valid['is_attributed'], valid_pred)

    print(f"Validation AUC score: {valid_score}")

    

    if test is not None: 

        test_pred = bst.predict(test[feature_cols])

        test_score = metrics.roc_auc_score(test['is_attributed'], test_pred)

        return bst, valid_score, test_score

    else:

        return bst, valid_score
train, valid, test = get_data_splits(clicks)

_, baseline_score, _ = train_model(train, valid, test)
# check your answer (Run this code cell to receive credit!)

q_1.solution()
from sklearn.feature_selection import SelectKBest, f_classif

feature_cols = clicks.columns.drop(['click_time', 'attributed_time', 'is_attributed'])

train, valid, test = get_data_splits(clicks)



# create the selector, keeping 40 features

selector = SelectKBest(f_classif, k=40)



# use the selector to retrieve the best features

X_new = selector.fit_transform(train[feature_cols], train['is_attributed']) 



# get back the kept features as a DataFrame with dropped columns as all 0s

selected_features = pd.DataFrame(selector.inverse_transform(X_new), 

                                 index=train.index, 

                                 columns=feature_cols)



# find the columns that were dropped

dropped_columns = selected_features.columns[selected_features.var() == 0]



# check your answer

q_2.check()
# uncomment these lines if you need some guidance

# q_2.hint()

# q_2.solution()
_ = train_model(train.drop(dropped_columns, axis=1), 

                valid.drop(dropped_columns, axis=1),

                test.drop(dropped_columns, axis=1))
# check your answer (Run this code cell to receive credit!)

q_3.solution()
from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import SelectFromModel



def select_features_l1(X, y):

    """ Return selected features using logistic regression with an L1 penalty """

    # define model

    logistic = LogisticRegression(C=0.1, penalty="l1", random_state=7).fit(X, y)

    model = SelectFromModel(logistic, prefit=True)

    

    X_new = model.transform(X)

    

    # get back the kept features as a DataFrame

    selected_features = pd.DataFrame(model.inverse_transform(X_new), 

                                     index=X.index,

                                     columns=X.columns)



    # drop columns have values of all 0s

    selected_columns = selected_features.columns[selected_features.var() != 0]

    

    return selected_columns



# check your answer

q_4.check()
# uncomment these if you're feeling stuck

# q_4.hint()

# q_4.solution()
# selected = select_features_l1(train[feature_cols], train['is_attributed'])



# dropped_columns = feature_cols.drop(selected)

# _ = train_model(train.drop(dropped_columns, axis=1), 

#                 valid.drop(dropped_columns, axis=1),

#                 test.drop(dropped_columns, axis=1))
# check your answer (Run this code cell to receive credit!)

q_5.solution()
# check your answer (Run this code cell to receive credit!)

q_6.solution()