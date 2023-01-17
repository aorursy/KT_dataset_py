# set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.feature_engineering.ex3 import *



# create features from timestamps

import pandas as pd

click_data = pd.read_csv(

    '../input/feature-engineering-data/train_sample.csv', parse_dates=['click_time']

)

click_times = click_data['click_time']

clicks = click_data.assign(

    day=click_times.dt.day.astype('uint8'),

    hour=click_times.dt.hour.astype('uint8'),

    minute=click_times.dt.minute.astype('uint8'),

    second=click_times.dt.second.astype('uint8')

)



# label encoding for categorical features

cat_features = ['ip', 'app', 'device', 'os', 'channel']

from sklearn.preprocessing import LabelEncoder

for feature in cat_features:

    label_encoder = LabelEncoder()

    clicks[feature] = label_encoder.fit_transform(clicks[feature])

    

def get_data_splits(dataframe, valid_fraction=0.1):

    # sort data

    dataframe = dataframe.sort_values('click_time')

    

    # split data

    valid_rows = int(len(dataframe) * valid_fraction)

    train = dataframe[:-valid_rows * 2]

    valid = dataframe[-valid_rows * 2:-valid_rows]

    test = dataframe[-valid_rows:]

    

    return train, valid, test



def train_model(train, valid, test=None, feature_cols=None):

    # choose features

    if feature_cols is None:

        feature_cols = train.columns.drop(

            ['click_time', 'attributed_time', 'is_attributed']

        )

    

    # define train & valid dataset

    import lightgbm as lgb

    dtrain = lgb.Dataset(train[feature_cols], label=train['is_attributed'])

    dvalid = lgb.Dataset(valid[feature_cols], label=valid['is_attributed'])

    

    # fit model

    param = {'num_leaves': 64, 'objective': 'binary', 'metric': 'auc', 'seed': 7}

    num_round = 1000

    print("Training model. Hold on a minute to see the validation score")

    bst = lgb.train(

        param, dtrain, num_round, valid_sets=[dvalid], early_stopping_rounds=20, verbose_eval=False

    )

    

    # make predictions

    valid_pred = bst.predict(valid[feature_cols])

    

    # evaluate the model

    from sklearn.metrics import roc_auc_score

    valid_score = roc_auc_score(valid['is_attributed'], valid_pred)

    print(f"Validation AUC score: {valid_score}")

    

    # test

    if test is not None:

        test_pred = bst.predict(test[feature_cols])

        test_score = roc_auc_score(test['is_attributed'], test_pred)

        return bst, valid_score, test_score

    else:

        return bst, valid_score



print("Baseline model score")

train, valid, test = get_data_splits(clicks)

_ = train_model(train, valid, test)
cat_features = ['ip', 'app', 'device', 'os', 'channel']

interactions = pd.DataFrame(index=clicks.index)



# Iterate through each pair of features, combine them into interaction features

from itertools import combinations

for col1, col2 in combinations(cat_features, 2):

    new_col_name = '_'.join([col1, col2])

    

    # convert to strings and combine

    new_values = clicks[col1].map(str) + "_" + clicks[col2].map(str)

    

    # encode

    encoder = LabelEncoder()

    interactions[new_col_name] = encoder.fit_transform(new_values)



# check your answer

q_1.check()
# uncomment if you need some guidance

# q_1.hint()

# q_1.solution()
interactions.head()
clicks = clicks.join(interactions)

print("Score with interactions")

train, valid, test = get_data_splits(clicks)

_ = train_model(train, valid)
# we removed ip in past exercise. are ip-combined-features have benefits now?

clicks_wo_ip = clicks.drop(['ip_app', 'ip_device', 'ip_os', 'ip_channel'], axis=1)

print("Score with droped-`ip`-interactions")

train, valid, test = get_data_splits(clicks_wo_ip)

_ = train_model(train, valid)
def count_past_events(series, time_window='6H'):

    series = pd.Series(series.index, index=series).sort_index()

    # subtract 1 so the current event isn't counted

    past_events = series.rolling(time_window).count() - 1

    return past_events



# check your answer

q_2.check()
# uncomment if you need some guidance

# q_2.hint()

# q_2.solution()
# loading in from saved Parquet file

past_events = pd.read_parquet('../input/feature-engineering-data/past_6hr_events.pqt')

clicks['ip_past_6hr_counts'] = past_events



train, valid, test = get_data_splits(clicks)

_ = train_model(train, valid, test)
# check your answer (Run this code cell to receive credit!)

q_3.solution()
def time_diff(series):

    return series.diff().dt.total_seconds()



# check your answer

q_4.check()
# uncomment if you need some guidance

# q_4.hint()

# q_4.solution()
# loading in from saved Parquet file

past_events = pd.read_parquet('../input/feature-engineering-data/time_deltas.pqt')

clicks['past_events_6hr'] = past_events



train, valid, test = get_data_splits(clicks.join(past_events))

_ = train_model(train, valid, test)
def previous_attributions(series):

    return series.expanding(min_periods=2).sum() - series



# Check your answer

q_5.check()
# uncomment if you need some guidance

# q_5.hint()

# q_5.solution()
# loading in from saved Parquet file

past_events = pd.read_parquet('../input/feature-engineering-data/downloads.pqt')

clicks['ip_past_6hr_counts'] = past_events



train, valid, test = get_data_splits(clicks)

_ = train_model(train, valid, test)
# check your answer (Run this code cell to receive credit!)

q_6.solution()