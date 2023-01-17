!pip install -U -t /kaggle/working/ git+https://github.com/Kaggle/learntools.git@dan-fe-review-2
import sys

sys.path.append('/kaggle/working')
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
# q_1.solution()
import category_encoders as ce



cat_features = ['ip', 'app', 'device', 'os', 'channel']

train, valid, test = get_data_splits(clicks)



# Create the count encoder

count_enc = ____



# Learn encoding from the training set

____



# Apply encoding to the train and validation sets as new columns

# Make sure to add `_count` as a suffix to the new columns

train_encoded = ____

valid_encoded = ____



q_2.check()
# Uncomment if you need some guidance

#q_2.hint()

#q_2.solution()
#%%RM_IF(PROD)%%

cat_features = ['ip', 'app', 'device', 'os', 'channel']

train, valid, test = get_data_splits(clicks)



# Create the count encoder

count_enc = ce.CountEncoder(cols=cat_features)



# Learn encoding from the training set

count_enc.fit(train[cat_features])



# Apply encoding to the train and validation sets

train_encoded = train.join(count_enc.transform(train[cat_features]).add_suffix('_count'))

valid_encoded = valid.join(count_enc.transform(valid[cat_features]).add_suffix('_count'))



q_2.assert_check_passed()
# Train the model on the encoded datasets

# This can take around 30 seconds to complete

_ = train_model(train_encoded, valid_encoded)
# q_3.solution()
cat_features = ['ip', 'app', 'device', 'os', 'channel']

train, valid, test = get_data_splits(clicks)



# Create the target encoder. You can find this easily by using tab completion.

# Start typing ce. the press Tab to bring up a list of classes and functions.

target_enc = ____



# Learn encoding from the training set. Use the 'is_attributed' column as the target.

____



# Apply encoding to the train and validation sets as new columns

# Make sure to add `_target` as a suffix to the new columns

train_encoded = ____

valid_encoded = ____



q_4.check()
# Uncomment these if you need some guidance

#q_4.hint()

#q_4.solution()
#%%RM_IF(PROD)%%

cat_features = ['ip', 'app', 'device', 'os', 'channel']

target_enc = ce.TargetEncoder(cols=cat_features)



train, valid, test = get_data_splits(clicks)

target_enc.fit(train[cat_features], train['is_attributed'])



train_encoded = train.join(target_enc.transform(train[cat_features]).add_suffix('_target'))

valid_encoded = valid.join(target_enc.transform(valid[cat_features]).add_suffix('_target'))



q_4.check()
_ = train_model(train_encoded, valid_encoded)
# q_5.solution()
train, valid, test = get_data_splits(clicks)



# Create the CatBoost encoder

cb_enc = ____



# Learn encoding from the training set

____



# Apply encoding to the train and validation sets as new columns

# Make sure to add `_cb` as a suffix to the new columns

train_encoded = ____

valid_encoded = ____

q_6.check()
# Uncomment these if you need some guidance

#q_6.hint()

#q_6.solution()
#%%RM_IF(PROD)%%

cat_features = ['app', 'device', 'os', 'channel']

train, valid, _ = get_data_splits(clicks)



cb_enc = ce.CatBoostEncoder(cols=cat_features, random_state=7)



# Learn encodings on the train set

cb_enc.fit(train[cat_features], train['is_attributed'])



# Apply encodings to each set

train_encoded = train.join(cb_enc.transform(train[cat_features]).add_suffix('_cb'))

valid_encoded = valid.join(cb_enc.transform(valid[cat_features]).add_suffix('_cb'))



q_6.assert_check_passed()
_ = train_model(train, valid)
encoded = cb_enc.transform(clicks[cat_features])

for col in encoded:

    clicks.insert(len(clicks.columns), col + '_cb', encoded[col])
import itertools

from sklearn.decomposition import TruncatedSVD
train, valid, test = get_data_splits(clicks)

cat_features = ['app', 'device', 'os', 'channel']



# Create the SVD transformer with 5 components, set random_state to 7

svd = ____



# Learn SVD feature vectors and store in svd_components as DataFrames

# Make sure you're only using the train set!

svd_components = {}

for col1, col2 in itertools.permutations(cat_features, 2):

    # Create the count matrix

    ____

    

    # Fit the SVD with the count matrix

    ____

    

    # Store the components in the dictionary. 

    svd_components['_'.join([col1, col2])] = ____



q_7.check()
# Uncomment these if you need some guidance

#q_7.hint()

#q_7.solution()
#%%RM_IF(PROD)%%

train, valid, test = get_data_splits(clicks)



# Learn SVD feature vectors

cat_features = ['app', 'device', 'os', 'channel']

svd_components = {}

svd = TruncatedSVD(n_components=5, random_state=7)

# Loop through each pair of categorical features

for col1, col2 in itertools.permutations(cat_features, 2):

    # For a pair, create a sparse matrix with cooccurence counts

    pair_counts = train.groupby([col1, col2])['is_attributed'].count()

    pair_matrix = pair_counts.unstack(fill_value=0)

    

    # Fit the SVD and store the components

    svd_components['_'.join([col1, col2])] = pd.DataFrame(svd.fit_transform(pair_matrix))



q_7.check()
svd_encodings = pd.DataFrame(index=clicks.index)



for feature in svd_components:

    ## Use svd_components to encode the categorical features and join with svd_encodings

    ____



q_8.check()
# Uncomment these if you need some guidance

#q_8.hint()

#q_8.solution()
#%%RM_IF(PROD)%%

svd_encodings = pd.DataFrame(index=clicks.index)

for feature in svd_components:

    # Get the feature column the SVD components are encoding

    col = feature.split('_')[0]



    ## Use SVD components to encode the categorical features

    feature_components = svd_components[feature]

    comp_cols = feature_components.reindex(clicks[col]).set_index(clicks.index)

    

    # Doing this so we know what these features are

    comp_cols = comp_cols.add_prefix(feature + '_svd_')

    

    svd_encodings = svd_encodings.join(comp_cols)



# Fill null values with the mean

svd_encodings = svd_encodings.fillna(svd_encodings.mean())



q_8.assert_check_passed()
train, valid, test = get_data_splits(clicks.join(svd_encodings))

_ = train_model(train, valid)