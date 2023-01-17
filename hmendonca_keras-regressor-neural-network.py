import random
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.base import clone
from keras.models import Sequential, load_model
from keras.layers import Dense, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping, TensorBoard
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score, check_cv, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from subprocess import check_output
from IPython.display import display # Allows the use of display() for DataFrames
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

random_seed = 42
random.seed = random_seed
np.random.seed = random_seed
import tensorflow as tf
tf.set_random_seed(random_seed)
# Read train files
train_df = pd.read_csv('../input/train.csv')
cols = ['f190486d6', '58e2e02e6', 'eeb9cd3aa', '9fd594eec', '6eef030c1',
        '15ace8c9f', 'fb0f5dbfe', '58e056e12', '20aa07010', '024c577b9',
        'd6bb78916', 'b43a7cfd5', '58232a6fb', '1702b5bf0', '324921c7b',
        '62e59a501', '2ec5b290f', '241f0f867', 'fb49e4212', '66ace2992',
        'f74e8f13d', '5c6487af1', '963a49cdc', '26fc93eb7', '1931ccfdd',
        '703885424', '70feb1494', '491b9ee45', '23310aa6f', 'e176a204a',
        '6619d81fc', '1db387535', 'fc99f9426', '91f701ba2', '0572565c2',
        '190db8488', 'adb64ff71', 'c47340d97', 'c5a231d81', '0ff32eb98']
def moving_avg(df, prefix, win_size):
    print('Creating rolling average on {} columns'.format(win_size))
    ts = df.T.rolling(win_size*2, min_periods=1, center=True).mean().iloc[ win_size//2 : : win_size ].T # rolling average
    ts.columns = [prefix+str(n) for n in np.arange(ts.columns.size)]
    ## also calculate moving deltas
    dts = pd.DataFrame(ts.iloc[:,:-1].values - ts.iloc[:,1:].values)
    dts.columns = ['d_'+prefix+str(n) for n in np.arange(dts.columns.size)]
    return ts.join(dts)

def moving_max(df, prefix, win_size):
    print('Creating rolling max on {} columns'.format(win_size))
    ts = df.T.rolling(win_size*2, min_periods=1, center=True).max().iloc[ win_size//2 : : win_size ].T # rolling max
    ts.columns = [prefix+str(n) for n in np.arange(ts.columns.size)]
    ## also calculate moving deltas
    dts = pd.DataFrame(ts.iloc[:,:-1].values - ts.iloc[:,1:].values)
    dts.columns = ['d_'+prefix+str(n) for n in np.arange(dts.columns.size)]
    return ts.join(dts)

def agg_features(df, suffix=''):
    X = pd.DataFrame()    
    X[suffix+"median_n0"] = df[df > 0].median(axis=1)
    X[suffix+"mean_n0"] = df[df > 0].mean(axis=1)
    X[suffix+"mean"] = df.mean(axis=1)
    X[suffix+"std"] = df.std(axis=1)
    X[suffix+"std_n0"] = df[df > 0].std(axis=1)
    X[suffix+"max"] = df.max(axis=1)
    X[suffix+"min_n0"] = df[df > 0].min(axis=1)
    X[suffix+"non0"] = (df > 0).sum(axis=1)
    return X

def prepare_X(df):
    # Drop ID, target and columns with less than 20% non-zeros
    col_mask = (train_df != 0).sum() > train_df.shape[0]*0.2
    X = df.loc[:,col_mask] ## top non0 raw columns
    
    X = pd.concat([X, agg_features(df)], axis=1) ## general aggregations
    X = pd.concat([X, agg_features(df.loc[:,col_mask], suffix='msk_')], axis=1) ## masked aggregations
    X = pd.concat([X, agg_features(df[cols], suffix='ord_')], axis=1) ## ordered aggregations
    
    # yearly, quartally, monthly, fortnightly, weekly, daily mean
#     X = pd.concat([X, moving_avg(df[cols], 'y_', 365)], axis=1)
#     X = pd.concat([X, moving_avg(df[cols], 'q_', 91)], axis=1)
    X = pd.concat([X, moving_avg(df[cols], 'm_', 30)], axis=1)
    X = pd.concat([X, moving_avg(df[cols], 'f_', 15)], axis=1)
    X = pd.concat([X, moving_avg(df[cols], 'd_', 7)], axis=1)
    #  ... and max
#     X = pd.concat([X, moving_max(df[cols], 'ym_', 365)], axis=1)
#     X = pd.concat([X, moving_max(df[cols], 'qm_', 91)], axis=1)
    X = pd.concat([X, moving_max(df[cols], 'mm_', 30)], axis=1)
    X = pd.concat([X, moving_max(df[cols], 'fm_', 15)], axis=1)
    X = pd.concat([X, moving_max(df[cols], 'dm_', 7)], axis=1)
    
    return X.fillna(0)
y_train = np.log1p(train_df["target"].values)
X_train = prepare_X(np.log1p(train_df.drop(["ID", "target"], axis=1)))
print(X_train.shape, X_train.columns.tolist())
X_train.head()
def get_model(size1=16, size2=32, size3=0):
    model = Sequential()
    model.add(Dense(size1, input_dim=X_train.shape[1], activation='relu'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.2))
    if size2 > 2:
        model.add(Dense(size2, activation='relu'))
        model.add(BatchNormalization())
        #model.add(Dropout(0.1))
    if size3 > 2:
        model.add(Dense(size3, activation='relu'))
        model.add(BatchNormalization())
    model.add(Dense(1, activation='relu'))
    model.compile(loss='mse', optimizer='adam')
    return model

def fit(X, y, folds=5):
    cv = KFold(folds, random_state=random_seed)
    estimators = []

    for train, val in cv.split(X):
        estimators.append(
            clone(model).fit(X[train], y[train], callbacks=[EarlyStopping(verbose=2, patience=10)], epochs=999, validation_data=[X[val], y[val]])
        )
    return estimators

def mean_score():
    best_scores = []
    for e in estimators:
        best_scores.append(
            e.history['val_loss'][-1] ## last score...
        )
    return np.mean(best_scores)

def predict(X):
    y_pred = []
    for e in estimators:
        y_pred.append(e.model.predict(X))
    return np.mean(y_pred, axis=0)
model = KerasRegressor(build_fn=get_model, size1=20, size2=0, size3=0, batch_size=16, verbose=2)
estimators = fit(X_train.values, y_train, folds=10)
print('RMSE:', np.sqrt(mean_score()))
test_df = pd.read_csv('../input/test.csv')
X_test = prepare_X(np.log1p(test_df.drop(['ID'], axis=1)))
print(X_test.shape)
sub = pd.read_csv('../input/sample_submission.csv')
sub["target"] = np.expm1(predict(X_test.values))
print(sub.head())
sub.to_csv('sub_keras_{:.2f}.csv'.format(np.sqrt(mean_score())), index=False)