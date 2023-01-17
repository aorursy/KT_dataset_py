import math

import pandas as pd

from itertools import islice

import numpy as np

import lightgbm as lgb

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold

import seaborn as sns

import logging

from tqdm import tqdm

import category_encoders

from sklearn.model_selection import TimeSeriesSplit

from sklearn.preprocessing import LabelEncoder, scale, MinMaxScaler, Normalizer, QuantileTransformer, PowerTransformer, StandardScaler

from scipy.stats import boxcox

import math

from sklearn.preprocessing import KBinsDiscretizer

from catboost import Pool, CatBoostClassifier



pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)
test_path = '../input/test.csv'



df_test = pd.read_csv(test_path)

df_test.drop(['ID_code'], axis=1, inplace=True)

df_test = df_test.values



unique_samples = []

unique_count = np.zeros_like(df_test)

for feature in tqdm(range(df_test.shape[1])):

    _, index_, count_ = np.unique(df_test[:, feature], return_counts=True, return_index=True)

    unique_count[index_[count_ == 1], feature] += 1



real_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) > 0)[:, 0]

synthetic_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) == 0)[:, 0]



del df_test
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



test_real = test.iloc[real_samples_indexes]



features = train.drop(['ID_code', 'target'], axis = 1).columns.tolist()



data = train.append(test_real)
num_round = 1000000

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof = np.zeros(len(train))

predictions = np.zeros(len(test))



model = CatBoostClassifier(loss_function="Logloss",

                           eval_metric="AUC",

                           task_type="GPU",

                           learning_rate=0.01,

                           iterations=70000,

                           l2_leaf_reg=50,

                           random_seed=42,

                           od_type="Iter",

                           depth=5,

                           early_stopping_rounds=15000,

                           border_count=64

                          )



for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, train.target.values)):

    print("Fold {}".format(fold_))

    X_train, y_train = train.iloc[trn_idx][features], train.iloc[trn_idx]['target']

    X_valid, y_valid = train.iloc[val_idx][features], train.iloc[val_idx]['target']

    

    for col in tqdm(features):

        gr = data[col].value_counts()

        gr_bin = data.groupby(col)[col].count()>1

        

        X_train[col + '_un'] = X_train[col].map(gr).astype('category').cat.codes

        X_valid[col + '_un'] = X_valid[col].map(gr).astype('category').cat.codes

        test[col + '_un'] = test[col].map(gr).astype('category').cat.codes

        

        X_train[col + '_un_bin'] = X_train[col].map(gr_bin).astype('category').cat.codes

        X_valid[col + '_un_bin'] = X_valid[col].map(gr_bin).astype('category').cat.codes

        test[col + '_un_bin'] = test[col].map(gr_bin).astype('category').cat.codes

        

        X_train[col + '_raw_mul'] = X_train[col] * X_train[col + '_un_bin']

        X_valid[col + '_raw_mul'] = X_valid[col] * X_valid[col + '_un_bin']

        test[col + '_raw_mul'] = test[col] * test[col + '_un_bin']

        

        X_train[col + '_raw_mul_2'] = X_train[col] * X_train[col + '_un']

        X_valid[col + '_raw_mul_2'] = X_valid[col] * X_valid[col + '_un']

        test[col + '_raw_mul_2'] = test[col] * test[col + '_un']

        

        X_train[col + '_raw_mul_3'] = X_train[col + '_un_bin'] * X_train[col + '_un']

        X_valid[col + '_raw_mul_3'] = X_valid[col + '_un_bin'] * X_valid[col + '_un']

        test[col + '_raw_mul_3'] = test[col + '_un_bin'] * test[col + '_un']





    _train = Pool(X_train, label=y_train)

    _valid = Pool(X_valid, label=y_valid)

    clf = model.fit(_train,

                    eval_set=_valid,

                    use_best_model=True,

                    verbose=5000)

    pred = clf.predict_proba(X_valid)[:,1]

    oof[val_idx] = pred

    print( "  auc = ", roc_auc_score(y_valid, pred) )

    predictions += clf.predict_proba(test.drop('ID_code', axis=1))[:,1] / folds.n_splits

print("CV score: {:<8.5f}".format(roc_auc_score(train.target, oof)))
sub = pd.DataFrame({"ID_code": test.ID_code.values})

sub["target"] = predictions

sub.to_csv("Range_bins_sub_3.csv", index=False)
sub.head()