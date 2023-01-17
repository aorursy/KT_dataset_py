MAX_ROUNDS = 5000

OPTIMIZE_ROUNDS = True
import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from multiprocessing import *

import gc

import warnings

warnings.filterwarnings("ignore")

import xgboost as xgb

from numba import jit
### Gini



def ginic(actual, pred):

    actual = np.asarray(actual) 

    n = len(actual)

    a_s = actual[np.argsort(pred)]

    a_c = a_s.cumsum()

    giniSum = a_c.sum() / a_s.sum() - (n + 1) / 2.0

    return giniSum / n

 

def gini_normalized(a, p):

    if p.ndim == 2:

        p = p[:,1] 

    return ginic(a, p) / ginic(a, a)

    



def gini_xgb(preds, dtrain):

    labels = dtrain.get_label()

    gini_score = gini_normalized(labels, preds)

    return 'gini', gini_score



# from CPMP's kernel https://www.kaggle.com/cpmpml/extremely-fast-gini-computation

@jit

def eval_gini(y_true, y_prob):

    y_true = np.asarray(y_true)

    y_true = y_true[np.argsort(y_prob)]

    ntrue = 0

    gini = 0

    delta = 0

    n = len(y_true)

    for i in range(n-1, -1, -1):

        y_i = y_true[i]

        ntrue += y_i

        gini += y_i * delta

        delta += 1 - y_i

    gini = 1 - 2 * gini / (ntrue * (n - ntrue))

    return gini
def transform_df(df):

    df = pd.DataFrame(df)

    dcol = [c for c in df.columns if c not in ['id','target']]

    df['ps_car_13_x_ps_reg_03'] = df['ps_car_13'] * df['ps_reg_03']

    df['negative_one_vals'] = np.sum((df[dcol]==-1).values, axis=1)

    for c in dcol:

        if '_bin' not in c: #standard arithmetic

            df[c+str('_median_range')] = (df[c].values > d_median[c]).astype(np.int)

            df[c+str('_mean_range')] = (df[c].values > d_mean[c]).astype(np.int)



    for c in one_hot:

        if len(one_hot[c])>2 and len(one_hot[c]) < 7:

            for val in one_hot[c]:

                df[c+'_oh_' + str(val)] = (df[c].values == val).astype(np.int)

    return df



def multi_transform(df):

    print('Init Shape: ', df.shape)

    p = Pool(cpu_count())

    df = p.map(transform_df, np.array_split(df, cpu_count()))

    df = pd.concat(df, axis=0, ignore_index=True).reset_index(drop=True)

    p.close(); p.join()

    print('After Shape: ', df.shape)

    return df
#### Load Data

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
### 

y = train['target'].values

testid= test['id'].values

trainid = train['id'].values





train.drop(['id','target'],axis=1,inplace=True)

test.drop(['id'],axis=1,inplace=True)



### Drop calc

unwanted = train.columns[train.columns.str.startswith('ps_calc_')]

train = train.drop(unwanted, axis=1)  

test = test.drop(unwanted, axis=1)
### Great Recovery from Pascal's materpiece



def recon(reg):

    integer = int(np.round((40*reg)**2)) 

    for a in range(32):

        if (integer - a) % 31 == 0:

            A = a

    M = (integer - A)//31

    return A, M

train['ps_reg_A'] = train['ps_reg_03'].apply(lambda x: recon(x)[0])

train['ps_reg_M'] = train['ps_reg_03'].apply(lambda x: recon(x)[1])

train['ps_reg_A'].replace(19,-1, inplace=True)

train['ps_reg_M'].replace(51,-1, inplace=True)

test['ps_reg_A'] = test['ps_reg_03'].apply(lambda x: recon(x)[0])

test['ps_reg_M'] = test['ps_reg_03'].apply(lambda x: recon(x)[1])

test['ps_reg_A'].replace(19,-1, inplace=True)

test['ps_reg_M'].replace(51,-1, inplace=True)
# Set up folds

K = 4

kf = KFold(n_splits = K, random_state = 1, shuffle = True)

y_valid_pred = pd.DataFrame(0*y)

y_test_pred = 0

X = pd.DataFrame(train)

ydf = pd.DataFrame(y)
# Set up classifier

params = {

    'eta': 0.025, 

    'max_depth': 4, 

    'subsample': 0.9, 

    'colsample_bytree': 0.7, 

    'colsample_bylevel':0.7,

    'min_child_weight':100,

    'alpha':4,

    'objective': 'binary:logistic', 

    'eval_metric': 'auc', 

    'seed': 99, 

    'silent': True

}
# Run CV



for i, (train_index, test_index) in enumerate(kf.split(train)):

    

    # Create data for this fold

    y_train, y_valid = ydf.iloc[train_index].copy(), ydf.iloc[test_index].copy()

    X_train, X_valid = X.iloc[train_index,:].copy(), X.iloc[test_index,:].copy()

    X_test = test.copy()

    print( "\nFold ", i)



    # Transform data for this fold

    one_hot = {c: list(X_train[c].unique()) for c in X_train.columns}

    X_train = X_train.replace(-1, np.NaN)  # Get rid of -1 while computing summary stats

    d_median = X_train.median(axis=0)

    d_mean = X_train.mean(axis=0)

    X_train = X_train.fillna(-1)  # Restore -1 for missing values



    X_train = multi_transform(X_train)

    X_valid = multi_transform(X_valid)

    X_test = multi_transform(X_test)



    # Run model for this fold

    if OPTIMIZE_ROUNDS:

        watchlist = [(xgb.DMatrix(X_train, y_train), 'train'), 

                     (xgb.DMatrix(X_valid, y_valid), 'valid')]

        model = xgb.train( params, xgb.DMatrix(X_train, y_train), MAX_ROUNDS,  

                           watchlist, feval=gini_xgb, maximize=True, 

                           verbose_eval=100, early_stopping_rounds=70)

        pred = model.predict(xgb.DMatrix(X_valid), ntree_limit=model.best_ntree_limit)

        test_pred = model.predict(xgb.DMatrix(X_test), ntree_limit=model.best_ntree_limit)

    else:

        model = xgb.train( params, xgb.DMatrix(X_train, y_train), MAX_ROUNDS,  

                           feval=gini_xgb, maximize=True, verbose_eval=100)

        pred = model.predict( xgb.DMatrix(X_valid) )

        test_pred = model.predict( xgb.DMatrix(X_test) )



    # Save validation predictions for this fold

    print( "  Gini = ", eval_gini(y_valid, pred) )

    y_valid_pred.iloc[test_index] = pred.reshape( y_valid_pred.iloc[test_index].shape )

    

    # Accumulate test set predictions

    y_test_pred += test_pred

    

y_test_pred /= K  # Average test set predictions



print( "\nGini for full training set:" )

eval_gini(y, y_valid_pred[0].values)
# Save validation predictions for stacking/ensembling

val = pd.DataFrame()

val['id'] = trainid

val['target'] = y_valid_pred[0].values

val.to_csv('forza_pascal_oof.csv', float_format='%.6f', index=False)
# Create submission file

sub = pd.DataFrame()

sub['id'] = testid

sub['target'] = y_test_pred

sub.to_csv('forza_pascal_test.csv', float_format='%.6f', index=False)