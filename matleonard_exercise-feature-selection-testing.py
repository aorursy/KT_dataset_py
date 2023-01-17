!pip install -U -t /kaggle/working/ git+https://github.com/Kaggle/learntools.git@dan-fe-review-2
import sys

sys.path.append('/kaggle/working')
import numpy as np

import pandas as pd

from sklearn import preprocessing, metrics

import lightgbm as lgb



# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.feature_engineering.ex4 import *



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

_, baseline_score, _ = train_model(train, valid, test)
#q_1.solution()
from sklearn.feature_selection import SelectKBest, f_classif
feature_cols = clicks.columns.drop(['click_time', 'attributed_time', 'is_attributed'])

train, valid, test = get_data_splits(clicks)



# Create the selector, keeping 40 features

selector = ____



# Use the selector to retrieve the best features

X_new = ____ 



# Get back the kept features as a DataFrame with dropped columns as all 0s

selected_features = ____



# Find the columns that were dropped

dropped_columns = ____
# Uncomment these lines if you need some guidance

q_2.hint()

q_2.solution()
#%%RM_IF(PROD)%%

feature_cols = clicks.columns.drop(['click_time', 'attributed_time', 'is_attributed'])

train, valid, test = get_data_splits(clicks)



# Do feature extraction on the training data only!

selector = SelectKBest(f_classif, k=40)

X_new = selector.fit_transform(train[feature_cols], train['is_attributed'])



# Get back the features we've kept, zero out all other features

selected_features = pd.DataFrame(selector.inverse_transform(X_new), 

                                 index=train.index, 

                                 columns=feature_cols)



# Dropped columns have values of all 0s, so var is 0, drop them

dropped_columns = selected_features.columns[selected_features.var() == 0]



q_2.check()
_ = train_model(train.drop(dropped_columns, axis=1), 

                valid.drop(dropped_columns, axis=1),

                test.drop(dropped_columns, axis=1))
#q_3.solution()
from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import SelectFromModel



def select_features_l1(X, y):

    """ Return selected features using logistic regression with an L1 penalty """

    ____



    return ____
# Uncomment these if you're feeling stuck

#q_4.hint()

#q_4.solution()
# Run this cell to check your work

q_4.check()
#%%RM_IF(PROD)%%



def select_features_l1(X, y):

    logistic = LogisticRegression(C=0.1, penalty="l1", random_state=7).fit(X, y)

    model = SelectFromModel(logistic, prefit=True)



    X_new = model.transform(X)

    

    # Get back the kept features as a DataFrame with dropped columns as all 0s

    selected_features = pd.DataFrame(model.inverse_transform(X_new), 

                                     index=X.index,

                                     columns=X.columns)

    

    # Dropped columns have values of all 0s, keep other columns 

    cols_to_keep = selected_features.columns[selected_features.var() != 0]

    

    return cols_to_keep



feature_cols = clicks.columns.drop(['click_time', 'attributed_time', 'is_attributed'])

train, valid, test = get_data_splits(clicks)



X, y = train[feature_cols][:10000], train['is_attributed'][:10000]



selected = select_features_l1(X, y)



q_4.check()
dropped_columns = feature_cols.drop(selected)

_ = train_model(train.drop(dropped_columns, axis=1), 

                valid.drop(dropped_columns, axis=1),

                test.drop(dropped_columns, axis=1))
#q_5.solution()
from sklearn.decomposition import PCA



train, valid, test = get_data_splits(clicks)

feature_cols = train.columns[-63:]



# Create the PCA transformer with 20 components

pca = ____



# Fit PCA to the feature columns

____
# Uncomment these if you're feeling stuck

#q_6.hint()

#q_6.solution()
# Run this cell to check your work

q_6.check()
#%%RM_IF(PROD)%%

from sklearn.decomposition import PCA



train, valid, test = get_data_splits(clicks)

feature_cols = train.columns[-63:]



pca = PCA(n_components=20, random_state=7)

pca.fit(train[feature_cols], train['is_attributed'])

q_6.check()
def encode_pcs(df, pca, feature_cols):

    """ Returns a new dataframe with the feature columns of a dataframe (defined with the

        feature_cols argument) encoded using a trained PCA transformer

        

        Arguments

        ---------

        df: DataFrame

        pca: Trained PCA transformer

        feature_cols: the feature columns of df that will be encoded

        

        Returns

        -------

        DataFrame with PCA encoded features

    """

    ____

    return ____
# Uncomment these if you're feeling stuck

# q_7.hint()

# q_7.solution()
#%%RM_IF(PROD)%%

def encode_pcs(df, pca, feature_cols):

    encodings = pd.DataFrame(pca.transform(df[feature_cols]),

                             index=df.index).add_prefix('pca_')

    encoded_df = df.drop(feature_cols, axis=1).join(encodings)

    return encoded_df



q_7.check()
_ = train_model(encode_pcs(train, pca, feature_cols),

                encode_pcs(valid, pca, feature_cols), 

                encode_pcs(test, pca, feature_cols))
from boruta import BorutaPy
def fit_boruta(df, feature_cols, target):

    """ Returns a new dataframe with the feature columns of a dataframe (defined with the

        feature_cols argument) encoded using Boruta

        

        Arguments

        ---------

        df: input DataFrame

        feature_cols: the feature columns of df that will be encoded

        target: the target column in df

        

        Returns

        -------

        List (or list-like) of the selected features

    """

    # Set random state to 7

    ____

    return ____
# Uncomment these if you're feeling stuck

#q_8.hint()

#q_8.solution()
# Run this to check your work

q_8.check()
#%%RM_IF(PROD)%%

from sklearn.ensemble import RandomForestClassifier



def fit_boruta(df, feature_cols, target):

    

    X = df[feature_cols].values

    y = df[target].values

    

    # define random forest classifier, with utilising all cores and

    # sampling in proportion to y labels

    rf = RandomForestClassifier(class_weight='balanced', max_depth=5, 

                                n_jobs=-1, random_state=7)



    # define Boruta feature selection method

    feat_selector = BorutaPy(rf, n_estimators='auto', random_state=7)



    # Fit the Boruta selector

    feat_selector.fit(X, y)



    # Get the selected columns

    selected_columns = feature_cols[feat_selector.support_]

    return selected_columns



q_8.check()
## Still need to fit boruta on the entire dataset and save the results.

## I'll load in the smaller dataset here
feature_cols = clicks.columns.drop(['click_time', 'attributed_time', 'is_attributed'])

train, valid, test = get_data_splits(clicks)



selected = fit_boruta(train[:5000], feature_cols, 'is_attributed')

rejected_columns = feature_cols.drop(selected)



_ = train_model(train.drop(rejected_columns, axis=1),

                valid.drop(rejected_columns, axis=1),

                test.drop(rejected_columns, axis=1))