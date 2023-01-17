# Set up code checking

!pip install -U -t /kaggle/working/ git+https://github.com/Kaggle/learntools.git

from learntools.core import binder

binder.bind(globals())

from learntools.feature_engineering.ex4 import *
import numpy as np

import pandas as pd

from sklearn import preprocessing, metrics

import lightgbm as lgb



import os



clicks = pd.read_parquet('../input/feature-engineering-data/baseline_data.pqt')

data_files = ['count_encodings.pqt',

              'catboost_encodings.pqt',

              'interactions.pqt',

              'past_6hr_events.pqt',

              'downloads.pqt',

              'time_deltas.pqt',

              'svd_encodings.pqt']

data_root = '../input/feature-engineering-data'

for file in data_files:

    features = pd.read_parquet(os.path.join(data_root, file))

    clicks = clicks.join(features)



def get_data_splits(dataframe, valid_fraction=0.1):



    dataframe = dataframe.sort_values('click_time')

    valid_rows = int(len(dataframe) * valid_fraction)

    train = dataframe[:-valid_rows * 2]

    # valid size == test size, last two sections of the data

    valid = dataframe[-valid_rows * 2:-valid_rows]

    test = dataframe[-valid_rows:]

    

    return train, valid, test



def train_model(train, valid, test=None, feature_cols=None):

    if feature_cols is None:

        feature_cols = train.columns.drop(['click_time', 'attributed_time',

                                           'is_attributed'])

    dtrain = lgb.Dataset(train[feature_cols], label=train['is_attributed'])

    dvalid = lgb.Dataset(valid[feature_cols], label=valid['is_attributed'])

    

    param = {'num_leaves': 64, 'objective': 'binary', 

             'metric': 'auc', 'seed': 7}

    num_round = 1000

    print("Training model!")

    bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid], 

                    early_stopping_rounds=20, verbose_eval=False)

    

    valid_pred = bst.predict(valid[feature_cols])

    valid_score = metrics.roc_auc_score(valid['is_attributed'], valid_pred)

    print(f"Validation AUC score: {valid_score}")

    

    if test is not None: 

        test_pred = bst.predict(test[feature_cols])

        test_score = metrics.roc_auc_score(test['is_attributed'], test_pred)

        return bst, valid_score, test_score

    else:

        return bst, valid_score
train, valid, test = get_data_splits(clicks)

_, baseline_score = train_model(train, valid)
# Check your answer (Run this code cell to receive credit!)

q_1.solution()
from sklearn.feature_selection import SelectKBest, f_classif

feature_cols = clicks.columns.drop(['click_time', 'attributed_time', 'is_attributed'])

train, valid, test = get_data_splits(clicks)



# Create the selector, keeping 40 features

selector = SelectKBest(f_classif, k = 40)



# Use the selector to retrieve the best features

X_new = selector.fit_transform(train[feature_cols], train['is_attributed']) 



# Get back the kept features as a DataFrame with dropped columns as all 0s

selected_features = pd.DataFrame(selector.inverse_transform(X_new), index = train.index, columns= feature_cols)



# Find the columns that were dropped

dropped_columns = selected_features.columns[selected_features.var()==0]



# Check your answer

q_2.check()
# Uncomment these lines if you need some guidance

# q_2.hint()

q_2.solution()
_ = train_model(train.drop(dropped_columns, axis=1), 

                valid.drop(dropped_columns, axis=1))
# Check your answer (Run this code cell to receive credit!)

q_3.solution()
from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import SelectFromModel



def select_features_l1(X, y):

    """Return selected features using logistic regression with an L1 penalty."""

    logistic = LogisticRegression(C = 0.1, penalty = 'l1', random_state = 7, solver='liblinear').fit(X,y)

    model = SelectFromModel(logistic, prefit = True)

    X_new = model.transform(X)

    # Get back the kept features as a DataFrame with dropped columns as all 0s

    selected_features = pd.DataFrame(model.inverse_transform(X_new), index = X.index, columns = X.columns)

    # Dropped columns have values of all 0s, keep other columns

    cols_to_keep = selected_features.columns[selected_features.var() != 0]

    

    return cols_to_keep



# Check your answer

q_4.check()
# Uncomment these if you're feeling stuck

#q_4.hint()

q_4.solution()
n_samples = 10000

X, y = train[feature_cols][:n_samples], train['is_attributed'][:n_samples]

selected = select_features_l1(X, y)



dropped_columns = feature_cols.drop(selected)

_ = train_model(train.drop(dropped_columns, axis=1), 

                valid.drop(dropped_columns, axis=1))
# Check your answer (Run this code cell to receive credit!)

q_5.solution()
# Check your answer (Run this code cell to receive credit!)

q_6.solution()