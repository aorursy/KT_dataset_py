import numpy as np

import pandas as pd

from sklearn import preprocessing, metrics

import lightgbm as lgb



# Set up code checking

# This can take a few seconds, thanks for your patience

from learntools.core import binder

binder.bind(globals())

from learntools.feature_engineering.ex2 import *



clicks = pd.read_parquet('../input/feature-engineering-data/baseline_data.pqt')
def get_data_splits(dataframe, valid_fraction=0.1):

    """ Splits a dataframe into train, validation, and test sets. First, orders by 

        the column 'click_time'. Set the size of the validation and test sets with

        the valid_fraction keyword argument.

    """



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
print("Baseline model")

train, valid, test = get_data_splits(clicks)

_ = train_model(train, valid)
# Check your answer (Run this code cell to receive credit!)

q_1.solution()
import category_encoders as ce



cat_features = ['ip', 'app', 'device', 'os', 'channel']

train, valid, test = get_data_splits(clicks)



# Create the count encoder

count_enc = ce.CountEncoder(cols=cat_features)



# Learn encoding from the training set

count_encoded = count_enc.fit(train[cat_features])



# Apply encoding to the train and validation sets as new columns

# Make sure to add `_count` as a suffix to the new columns

train_encoded = train.join(count_encoded.transform(train[cat_features]).add_suffix('_count'))

valid_encoded = valid.join(count_encoded.transform(valid[cat_features]).add_suffix('_count'))



# Check your answer

q_2.check()
# Uncomment if you need some guidance

#q_2.hint()

#q_2.solution()
# Train the model on the encoded datasets

# This can take around 30 seconds to complete

_ = train_model(train_encoded, valid_encoded)
# Check your answer (Run this code cell to receive credit!)

q_3.solution()
cat_features = ['ip', 'app', 'device', 'os', 'channel']

train, valid, test = get_data_splits(clicks)



# Create the target encoder. You can find this easily by using tab completion.

# Start typing ce. the press Tab to bring up a list of classes and functions.

target_enc = ce.TargetEncoder(cols=cat_features)



# Learn encoding from the training set. Use the 'is_attributed' column as the target.

target_enc.fit(train[cat_features], train['is_attributed'])



# Apply encoding to the train and validation sets as new columns

# Make sure to add `_target` as a suffix to the new columns

train_encoded = train.join(target_enc.transform(train[cat_features]).add_suffix('_target'))

valid_encoded = valid.join(target_enc.transform(valid[cat_features]).add_suffix('_target'))



# Check your answer

q_4.check()
# Uncomment these if you need some guidance

#q_4.hint()

#q_4.solution()
_ = train_model(train_encoded, valid_encoded)
# Check your answer (Run this code cell to receive credit!)

q_5.solution()
train, valid, test = get_data_splits(clicks)

cat_features = ['app', 'device', 'os', 'channel']



# Create the CatBoost encoder

cb_enc = ce.CatBoostEncoder(cols=cat_features)



# Learn encoding from the training set

cb_enc.fit(train[cat_features], train['is_attributed'])



# Apply encoding to the train and validation sets as new columns

# Make sure to add `_cb` as a suffix to the new columns

train_encoded = train.join(cb_enc.transform(train[cat_features]).add_suffix('_cb'))

valid_encoded = valid.join(cb_enc.transform(valid[cat_features]).add_suffix('_cb'))



# Check your answer

q_6.check()
# Uncomment these if you need some guidance

#q_6.hint()

#q_6.solution()
_ = train_model(train, valid)
encoded = cb_enc.transform(clicks[cat_features])

for col in encoded:

    clicks.insert(len(clicks.columns), col + '_cb', encoded[col])