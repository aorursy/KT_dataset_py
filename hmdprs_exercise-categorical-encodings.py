# set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.feature_engineering.ex2 import *

print('Setup is completed!')



import numpy as np

from sklearn import preprocessing, metrics

import lightgbm as lgb



import pandas as pd

clicks = pd.read_parquet('../input/feature-engineering-data/baseline_data.pqt')
def get_data_splits(dataframe, valid_fraction=0.1):

    # sort DataFrame, timeseries data consideration

    dataframe = dataframe.sort_values('click_time')

    

    # split data, size(valid) = size(test), last two sections of the data

    valid_rows = int(len(dataframe) * valid_fraction)

    train = dataframe[:-valid_rows * 2]

    valid = dataframe[-valid_rows * 2:-valid_rows]

    test = dataframe[-valid_rows:]

    

    return train, valid, test



def train_model(train, valid, test=None, feature_cols=None):

    # define features

    if feature_cols is None:

        feature_cols = train.columns.drop(

            ['click_time', 'attributed_time', 'is_attributed']

        )

    

    # define train and valid datasets

    dtrain = lgb.Dataset(train[feature_cols], label=train['is_attributed'])

    dvalid = lgb.Dataset(valid[feature_cols], label=valid['is_attributed'])

    

    # fit model

    param = {

        'num_leaves': 64, 'objective': 'binary', 'metric': 'auc', 'seed': 7

    }

    num_round = 1000

    print("Training model!")

    bst = lgb.train(

        param, dtrain, num_round, valid_sets=[dvalid], early_stopping_rounds=20, verbose_eval=False

    )

    

    # make predictions

    valid_pred = bst.predict(valid[feature_cols])

    

    # evaluate the model

    valid_score = metrics.roc_auc_score(valid['is_attributed'], valid_pred)

    print(f"Validation AUC score: {valid_score}")

    

    if test is not None:

        test_pred = bst.predict(test[feature_cols])

        test_score = metrics.roc_auc_score(test['is_attributed'], test_pred)

        return bst, valid_score, test_score

    else:

        return bst, valid_score
print("Baseline model")

train, valid, test = get_data_splits(clicks)

_ = train_model(train, valid)
# check your answer (Run this code cell to receive credit!)

q_1.solution()
cat_features = ['ip', 'app', 'device', 'os', 'channel']

train, valid, test = get_data_splits(clicks)



# create the count encoder

from category_encoders import CountEncoder

count_enc = CountEncoder(cols=cat_features)



# learn encoding from the training set

count_enc.fit(train[cat_features])



# apply encoding to the train and validation sets as new columns

# make sure to add `_count` as a suffix to the new columns

train_encoded = train.join(count_enc.transform(train[cat_features]).add_suffix("_count"))

valid_encoded = valid.join(count_enc.transform(valid[cat_features]).add_suffix("_count"))



# check your answer

q_2.check()
# uncomment if you need some guidance

# q_2.hint()

# q_2.solution()
# train the model on the encoded datasets (takes around 30 seconds to complete)

_ = train_model(train_encoded, valid_encoded)
# check your answer (Run this code cell to receive credit!)

q_3.solution()
cat_features = ['ip', 'app', 'device', 'os', 'channel']

train, valid, test = get_data_splits(clicks)



# create the target encoder. you can find this easily by using tab completion.

from category_encoders import TargetEncoder

target_enc = TargetEncoder(cols=cat_features)



# learn encoding from the training set. use the 'is_attributed' column as the target

target_enc.fit(train[cat_features], train['is_attributed'])



# apply encoding to the train and validation sets as new columns

# make sure to add `_target` as a suffix to the new columns

train_encoded = train.join(target_enc.transform(train[cat_features]).add_suffix("_target"))

valid_encoded = valid.join(target_enc.transform(valid[cat_features]).add_suffix("_target"))



# check your answer

q_4.check()
# uncomment these if you need some guidance

# q_4.hint()

# q_4.solution()
_ = train_model(train_encoded, valid_encoded)
# check your answer (Run this code cell to receive credit!)

q_5.solution()
cat_features = ['app', 'device', 'os', 'channel']

train, valid, test = get_data_splits(clicks)



# create the CatBoost encoder

from category_encoders import CatBoostEncoder

cb_enc = CatBoostEncoder(cols=cat_features)



# learn encoding from the training set. use the 'is_attributed' column as the target

cb_enc.fit(train[cat_features], train['is_attributed'])



# apply encoding to the train and validation sets as new columns

# make sure to add `_cb` as a suffix to the new columns

train_encoded = train.join(cb_enc.transform(train[cat_features]).add_suffix("_cb"))

valid_encoded = valid.join(cb_enc.transform(valid[cat_features]).add_suffix("_cb"))



# check your answer

q_6.check()
# uncomment these if you need some guidance

# q_6.hint()

# q_6.solution()
_ = train_model(train, valid)
encoded = cb_enc.transform(clicks[cat_features])

for col in encoded:

    clicks.insert(len(clicks.columns), col + '_cb', encoded[col])

clicks.head()