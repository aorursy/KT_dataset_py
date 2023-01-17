import numpy as np

import pandas as pd

from sklearn import preprocessing, metrics

import lightgbm as lgb



# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.feature_engineering.ex3 import *



# Create features from   timestamps

click_data = pd.read_csv('../input/feature-engineering-data/train_sample.csv', 

                         parse_dates=['click_time'])

click_times = click_data['click_time']

clicks = click_data.assign(day=click_times.dt.day.astype('uint8'),

                           hour=click_times.dt.hour.astype('uint8'),

                           minute=click_times.dt.minute.astype('uint8'),

                           second=click_times.dt.second.astype('uint8'))



# Label encoding for categorical features

cat_features = ['ip', 'app', 'device', 'os', 'channel']

for feature in cat_features:

    label_encoder = preprocessing.LabelEncoder()

    clicks[feature] = label_encoder.fit_transform(clicks[feature])

    

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

    print("Training model. Hold on a minute to see the validation score")

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



print("Baseline model score")

train, valid, test = get_data_splits(clicks)

_ = train_model(train, valid, test)
import itertools



cat_features = ['ip', 'app', 'device', 'os', 'channel']

interactions = pd.DataFrame(index=clicks.index)



# Iterate through each pair of features, combine them into interaction features

for col1, col2 in itertools.combinations(cat_features, 2):

    new_col_name = '_'.join([col1, col2])

    # Convert to strings and combine

    new_values = clicks[col1].map(str) + "_" + clicks[col2].map(str)



    encoder = preprocessing.LabelEncoder()

    interactions[new_col_name] = encoder.fit_transform(new_values)



# Check your answer

q_1.check()
# Uncomment if you need some guidance

q_1.hint()

q_1.solution()
clicks = clicks.join(interactions)

print("Score with interactions")

train, valid, test = get_data_splits(clicks)

_ = train_model(train, valid)
def count_past_events(series , time_window='6H'):

    series = pd.Series(series.index, index=series)

    past_event = series.rolling(time_window).count() - 1

    return past_event



# Check your answer

q_2.check()
# Uncomment if you need some guidance

q_2.hint()

q_2.solution()
# Loading in from saved Parquet file

past_events = pd.read_parquet('../input/feature-engineering-data/past_6hr_events.pqt')

clicks['ip_past_6hr_counts'] = past_events



train, valid, test = get_data_splits(clicks)

_ = train_model(train, valid, test)
# Check your answer (Run this code cell to receive credit!)

q_3.solution()
def time_diff(series):

    """ Returns a series with the time since the last timestamp in seconds """

    ____

    return series.diff().dt.total_seconds()

# Uncomment if you need some guidance

q_4.hint()

q_4.solution()
# Check your answer

q_4.check()
# Loading in from saved Parquet file

past_events = pd.read_parquet('../input/feature-engineering-data/time_deltas.pqt')

clicks['past_events_6hr'] = past_events



train, valid, test = get_data_splits(clicks.join(past_events))

_ = train_model(train, valid, test)
def previous_attributions(series):

    """ Returns a series with the """

    all_in = series.expanding(min_periods=2).sum()- series

    return all_in



# Check your answer

q_5.check()
# Uncomment if you need some guidance

q_5.hint()

q_5.solution()
# Loading in from saved Parquet file

past_events = pd.read_parquet('../input/feature-engineering-data/downloads.pqt')

clicks['ip_past_6hr_counts'] = past_events
train, valid, test = get_data_splits(clicks)

_ = train_model(train, valid, test)
# Check your answer (Run this code cell to receive credit!)

q_6.solution()