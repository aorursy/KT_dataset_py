# Imports

# General imports

import time

import math



from pathlib import Path



# FFT

from scipy import fftpack



# Standard DS imports

import numpy as np

import pandas as pd

import scipy as sp

import matplotlib.pyplot as plt



from scipy.stats import mode



# ML Related Imports

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import cross_val_score

import lightgbm as lgb





# Metrics

from sklearn.metrics import mean_squared_error, accuracy_score



# Constanta

HOME_DIR = Path('../input')

TRAIN_VAL_SPLIT = 0.2

RANDOM_SEED = 42

F_S = 400    # Sampling frequency (need to investigate this more . . . )
# Read in all data

X_train_df = pd.read_csv(HOME_DIR/'X_train.csv')

y_train_df = pd.read_csv(HOME_DIR/'y_train.csv')

X_test_df = pd.read_csv(HOME_DIR/'X_test.csv')

sample_submission_df = pd.read_csv(HOME_DIR/'sample_submission.csv')
print(f'X_train shape: {X_train_df.shape}\ny_train shape: {y_train_df.shape}\n'

      f'X_test shape: {X_test_df.shape}\nsample submission shape: {sample_submission_df.shape}\n')
X_train_df.columns
y_train_df.columns
X_test_df.columns
X_train_df.head(10)
y_train_df.head()
y_train_df.surface.unique()
def group_by_series_id(df):



    columns_to_use = [ #'orientation_X', 'orientation_Y', 'orientation_Z', 'orientation_W',

                      'angular_velocity_X', 'angular_velocity_Y', 'angular_velocity_Z',

                      'linear_acceleration_X', 'linear_acceleration_Y', 'linear_acceleration_Z']

    # columns_to_use = ['orientation_X', 'orientation_Y']



    def get_col_values(grouped_df):  

        tmp_df = pd.DataFrame()

        for col in columns_to_use:

            # Abbreviate 'orientation_X' to 'oX', etc.

            col_abbr = ''.join([col[0], col[-1]])

            

            tmp_df[col_abbr] = grouped_df[col].values

        return tmp_df

        

    return df.groupby('series_id').apply(get_col_values)
def group_by_series_id2(df):

    # This version is faster. I think because its not being forced into tmp_df over and over again



    columns_to_use = [ 'orientation_X', 'orientation_Y', 'orientation_Z', 'orientation_W',

                      'angular_velocity_X', 'angular_velocity_Y', 'angular_velocity_Z',

                      'linear_acceleration_X', 'linear_acceleration_Y', 'linear_acceleration_Z']

    # columns_to_use = ['orientation_X', 'orientation_Y']



    def get_col_values(grouped_df):

        # Make a dict of col_abbr:full_column_name. This is used both to make col names short and create a new DF

        # with the values of the columns

        tmp_dict = {}

        for col in columns_to_use:

            # Abbreviate 'orientation_X' to 'oX', etc.

            col_abbr = ''.join([col[0], col[-1]])

            tmp_dict[col_abbr] = col

        # Now use a dict comprehension and return only the values in each of the columns

        return pd.DataFrame({col_abbr:grouped_df[col].values for (col_abbr, col) in tmp_dict.items()})

        

    return df.groupby('series_id').apply(get_col_values)
%%time

# Get the data into a DF grouped by series_id

X_train_grouped = group_by_series_id2(X_train_df)

X_test_grouped = group_by_series_id2(X_test_df)
# Some sanity checking

X_train_df[X_train_df.series_id == 247].sum()
X_train_grouped.loc[247].sum()
X_train_grouped.head()
def convert_td_to_fd(td):

    ''' Return frequency domain data from time domain data'''

    fd = fftpack.fft(td)

    return np.abs(fd)
def add_fd_columns(df):

    ''' This will add freq domain columns (appending_f) to the td columns of df '''

    # Apply FFT to each column of the df and add the '_f' suffix to resulting dataframe columns to indicate freq domain

    df_f = df.apply(convert_td_to_fd).add_suffix('_fd')

    # Now merge the two df and return it

    return pd.merge(df, df_f, how='outer', left_index=True, right_index=True)
%%time

# Now add the freq domain data to both train and test dfs (suffix _wfd = with freq domain)

X_train_wfd = X_train_grouped.groupby('series_id').apply(add_fd_columns)

X_test_wfd = X_test_grouped.groupby('series_id').apply(add_fd_columns)
X_train_wfd.head()
# Extract values from the FD columns of X_train_wfd and Y_train_wfd to prep for training

x_train_fd_cols = [x for x in X_train_wfd if x.endswith('_fd') ]

X_train_values=X_train_wfd[x_train_fd_cols].groupby('series_id').apply(lambda x: x.values.T)

x_test_fd_cols = [x for x in X_test_wfd if x.endswith('_fd') ]

X_test_values=X_test_wfd[x_train_fd_cols].groupby('series_id').apply(lambda x: x.values.T)
# Convert the y_train catogories (concrete, etc) to numbers

y_labels, y_categoricals = pd.factorize(y_train_df.surface)
# Split the training set to train and valid sets. Doing spimple splitting for now. More sophisticated splits will be done

# in later experiments

train_X, val_X, train_y, val_y = train_test_split(X_train_values, y_labels, test_size=TRAIN_VAL_SPLIT, random_state=RANDOM_SEED)
train_X.shape, train_y.shape, val_X.shape, val_y.shape
# Now the train and val X need to be "stacked" to make them a 2 dim NP array so it can be fed into the models

train_X_np = np.vstack(train_X)

val_X_np = np.vstack(val_X)

# Similarly with the test set

test_X_np = np.vstack(X_test_values)
# After the train and val are stacket, the y labels must be repeated by len(x_train_fd_cols) times.

train_y_rep = np.repeat(train_y, len(x_train_fd_cols))

val_y_rep = np.repeat(val_y, len(x_train_fd_cols))
train_X_np.shape, train_y_rep.shape, val_X_np.shape, val_y_rep.shape, test_X_np.shape
# LightGBM Parameters

params = {'application': 'multiclass',

          'num_class': 9,

          'boosting': 'gbdt',

          # 'metric': 'rmse',

          'num_leaves': 600,    # Orig: 90

          'max_depth': 100,      # Orig: 9

          'min_data_in_leaf': 80,

          'learning_rate': 0.25, # Orig: 0.01. 0.25 gives best result so far

          # 'n_estimators': 100,

          'bagging_fraction': 0.85,

          'feature_fraction': 0.8,

          'min_split_gain': 0.02,    # Orig: 0.02

          'min_child_samples': 150,   # Orig: 150

          'min_child_weight': 0.002,    # Orig: 0.02

          'lambda_l2': 0.0475,             # Orig: 0.0475

          # 'lambda_l1': 0.1,

          'verbosity': -1,

          # 'n_jobs': 8,

          'data_random_seed': RANDOM_SEED}



# Additional parameters:

early_stop = 1000

verbose_eval = 100

n_splits = 5
%%time

lgb_clf = lgb.LGBMClassifier(**params)

lgb_clf.fit(train_X_np, train_y_rep)
# Predict on the validation set

pred_val_y = lgb_clf.predict(val_X_np)
# Predict multiclass accuracy

accuracy_score(val_y_rep, pred_val_y)
# Retrain on the full set

train_X_full_np = np.vstack(X_train_values)

train_y_full_rep = np.repeat(y_labels, len(x_train_fd_cols))
train_X_full_np.shape, train_y_full_rep.shape
%%time

lgb_clf_full = lgb.LGBMClassifier(**params)

lgb_clf_full.fit(train_X_full_np, train_y_full_rep)
# Predict for test set

pred_test_y = lgb_clf_full.predict(test_X_np)

test_X_np.shape, pred_test_y.shape
# Reshape the array such that predictions for one series is in one row

pred_test_y_reshape = pred_test_y.reshape((X_test_values.shape[0], len(x_test_fd_cols)))
pred_test_y_reshape.shape
# For now, simply take the mode of each row

submission_preds = sp.stats.mode(pred_test_y_reshape, axis=1)[0].reshape(-1)

submission_preds_counts = sp.stats.mode(pred_test_y_reshape, axis=1)[1]
# Get the equivalent of "value_counts" (or table() in R) for the prediction counts.

# This will give us a feel for the number of predictions that were same for each feature

unique, counts = np.unique(submission_preds_counts, return_counts=True)

np.asarray((unique, counts)).T
sample_submission_df.surface = list(map(lambda x: y_categoricals[int(x)], submission_preds))
# Do a quick sanity check

sample_submission_df.head()
# Do a quick sanity check

sample_submission_df.shape
# Now write it out

sample_submission_df.to_csv('submission_fft_lgbm.csv', index=False)