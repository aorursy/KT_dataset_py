# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))
import pandas as pd

import numpy as np

from sklearn import preprocessing, metrics

import lightgbm as lgb
# Functions to Test Models



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
# dataset0.head(5)
dataset0 = pd.read_csv('../input/feature-engineering-data/train_sample.csv',

                         parse_dates=['click_time'])
##### 1) Features from timestamps



dataset0_index = 'click_time'



### dataset1 is a dataframe columns with additional timestamp features day, hour, minute, and second

### we use to represent time series elaboration

# Add new columns for timestamp features day, hour, minute, and second

dataset1 = dataset0.copy()

dataset1['day']    = dataset1[dataset0_index].dt.day.astype('uint8')

dataset1['hour']   = dataset1[dataset0_index].dt.hour.astype('uint8')

dataset1['minute'] = dataset1[dataset0_index].dt.minute.astype('uint8')

dataset1['second'] = dataset1[dataset0_index].dt.second.astype('uint8')
from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder

import category_encoders as ce



# Function: multi_encoders_function(train, valid, categorical_features, target_column, method print_transform_request = False)

def multi_encoders_function(train, valid, categorical_features, target_column, method = 'label', print_transform_request = False):

    print("These are the categorical features used: ", categorical_features)

    print_before_encoder(train, valid, print_transform_request)

    if method == 'label':

        train_enc, valid_enc = my_label_encoder(train, valid, categorical_features)

    elif method == 'count':

        train_enc, valid_enc = my_count_encoder(train, valid, categorical_features)

    elif method == 'target':

        train_enc, valid_enc = my_target_encoder(train, valid, categorical_features, target_column)

    elif method == 'catboost':

        train_enc, valid_enc =my_catboost_encoder(train, valid, categorical_features, target_column)

    print_after_encoder(train_enc, valid_enc, print_transform_request)

    return train_enc, valid_enc



# Function to Print Original Dataframe

def print_before_encoder(train, valid, print_transform_request):

    if print_transform_request == True:

        print("")

        print("These are the columns from original training dataframe: ")

        print(train.iloc[0:4])

        print("")



# Function to Print Transformed Dataframe

def print_after_encoder(train_encoded, valid_encoded, print_transform_request):

    if print_transform_request == True:

        print("")

        print("These are the columns on the transformed training dataframe: ")

        print(train_encoded.iloc[0:4])  

        print("")



# Label Encoder Function

def my_label_encoder(train, valid, categorical_features):

    print("my_label_encoder function is called.")

    #

    # Create a LabelEncoder instance

    label_encoder = preprocessing.LabelEncoder()

    #

    # Create new columns in dataframe using preprocessing.LabelEncoder()

    for feature in categorical_features:

        encoded_feature_train = label_encoder.fit_transform(train[feature])    # apply label_encoder

        train[feature + '_labels'] = encoded_feature_train                     # save the result to a differently named column, with '_labels'

        encoded_feature_valid = label_encoder.fit_transform(valid[feature])    # apply label_encoder

        valid[feature + '_labels'] = encoded_feature_valid                     # save the result to a differently named column, with '_labels'

    #    

    return train, valid



# Count Encoder Function

def my_count_encoder(train, valid, categorical_features):

    ######################### Count Encoder###############################

    # Count encoding replaces each categorical value with the number of times it appears 

    # in the original dataset. 

    # For example, if the value "GB" occured 10 times in the country feature, then each "GB" 

    # would be replaced with the number 10.

    ######################################################################

    print("my_count_encoder function is called.")

    # Create a Count Encoder instance

    count_enc = ce.CountEncoder(cols=categorical_features)

    #

    # Learn encoding from the training set

    count_enc.fit(train[cat_features])

    #

    # Apply encoding to the train and validation sets as new columns

    # Make sure to add `_count` as a suffix to the new columns

    train_encoded = train.join(count_enc.transform(train[cat_features]).add_suffix('_count'))    

    valid_encoded = valid.join(count_enc.transform(valid[cat_features]).add_suffix('_count'))

    #      

    return train_encoded, valid_encoded



def my_target_encoder(train, valid, categorical_features, target_column):

    ############################ Target Encoding ##################################

    # Target encoding replaces a categorical value with the average value of the target for that value of the feature. 

    # For example, given the country value "CA", you'd calculate the average outcome for all the rows with country == 'CA', 

    # around 0.28. This is often blended with the target probability over the entire dataset to reduce the 

    # variance of values with few occurences.

    ################################################################################ 

    print("my_target_encoder function is called.")

    #

    # Create the target encoder. 

    target_enc = ce.TargetEncoder(cols=cat_features)

    #

    # Learn encoding from the training set. Use the the target column as target_column.

    target_enc.fit(train[cat_features], train[target_column])

    #

    # Apply encoding to the train and validation sets as new columns

    # Make sure to add `_target` as a suffix to the new columns

    train_encoded = train.join(target_enc.transform(train[cat_features]).add_suffix('_target'))

    valid_encoded = valid.join(target_enc.transform(valid[cat_features]).add_suffix('_target'))

    #

    return train_encoded, valid_encoded



def my_catboost_encoder(train, valid, categorical_features, target_column):

    ########################### CatBoost Encoding #####################################

    # CatBoost encoding. This is similar to target encoding in that it's based on the target probablity 

    # for a given value. However with CatBoost, for each row, the target probability is calculated only from the 

    # rows before it.

    ###################################################################################

    print("my_catboost_encoder function is called.")

    #

    # Have to tell it which features are categorical when they aren't strings

    cb_enc = ce.CatBoostEncoder(cols=cat_features, random_state=7)

    #

    # Learn encoding from the training set

    cb_enc.fit(train[cat_features], train[target_column])

    #

    train_encoded = train.join(cb_enc.transform(train[cat_features]).add_suffix('_cb'))

    valid_encoded = valid.join(cb_enc.transform(valid[cat_features]).add_suffix('_cb'))

    #

    return train_encoded, valid_encoded

# Test multi_encoders_function  label, count, target, catboost

cat_features = ['ip', 'app', 'device', 'os', 'channel']

target_column = 'is_attributed'

method_list = ['label', 'count', 'target', 'catboost']



# warning: train_model function return three variables if used with train, valid, test as argument

# for the moment, only use train, valid for the argument

# because we only want to check the encoding method

dictionary_encoder_result = {}

for method_i in method_list:

    train, valid, test = get_data_splits(dataset1)

    train, valid = multi_encoders_function(train, valid, cat_features, target_column, method_i , print_transform_request = False)

    bst, valid_score = train_model(train, valid)

    dictionary_encoder_result[method_i] = valid_score

    print(dictionary_encoder_result)

    print("")

    