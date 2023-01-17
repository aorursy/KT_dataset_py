import pandas as pd

import numpy as np

np.random.seed(1133)

import itertools

import xgboost as xgb

from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
usable_columns = list(set(df_train.columns) - set(['ID', 'y']))



y_train = df_train['y'].values

id_test = df_test['ID'].values



x_train = df_train[usable_columns]

x_test = df_test[usable_columns]



for column in usable_columns:

    cardinality = len(np.unique(x_train[column]))

    if cardinality == 1:

        x_train.drop(column, axis=1) # Column with only one value is useless so we drop it

        x_test.drop(column, axis=1)

    if cardinality > 2: # Column is categorical

        mapper = lambda x: sum([ord(digit) for digit in x])

        x_train[column] = x_train[column].apply(mapper)

        x_test[column] = x_test[column].apply(mapper)
#code is from an old competition(santander)

def remove_feat_identicals(data_frame):

    # Find feature vectors having the same values in the same order and

    # remove all but one of those redundant features.

    print("")

    print("Delete these identical features...")

    n_features_originally = data_frame.shape[1]

    # Find the names of identical features by going through all the

    # combinations of features (each pair is compared only once).

    feat_names_delete = []

    for feat_1, feat_2 in itertools.combinations(

            iterable=data_frame.columns, r=2):

        if np.array_equal(data_frame[feat_1], data_frame[feat_2]):

            feat_names_delete.append(feat_2)

    feat_names_delete = np.unique(feat_names_delete)

    # Delete the identical features

    #data_frame = data_frame.drop(labels=feat_names_delete, axis=1)

    n_features_deleted = len(feat_names_delete)

    print("  - Delete %s / %s features (~= %.1f %%)" % (

        n_features_deleted, n_features_originally,

        100.0 * (np.float(n_features_deleted) / n_features_originally)))

    return feat_names_delete
feature_to_delete = remove_feat_identicals(x_train)
#delete identical features 





x_train.drop(feature_to_delete, axis=1, inplace=True)

x_test.drop(feature_to_delete, axis=1, inplace=True)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)



d_train = xgb.DMatrix(x_train, label=y_train)

d_valid = xgb.DMatrix(x_valid, label=y_valid)

d_test = xgb.DMatrix(x_test)



params = {}

params['objective'] = 'reg:linear'

params['eta'] = 0.02

params['max_depth'] = 4

params["objective"] = "reg:linear"

params["min_child_weight"] = 1

params["subsample"] = 0.9

params["colsample_bytree"] = 0.8

params["silent"] = 1

params["seed"] = 1


def xgb_r2_score(preds, dtrain):

    labels = dtrain.get_label()

    return 'r2', r2_score(labels, preds)
watchlist = [(d_train, 'train'), (d_valid, 'valid')]



clf = xgb.train(params, d_train, 1000, watchlist, early_stopping_rounds=50, feval=xgb_r2_score, maximize=True, verbose_eval=10)
#do prediction

p_test = clf.predict(d_test)



sub = pd.DataFrame()

sub['ID'] = id_test

sub['y'] = p_test

sub.to_csv('xgb_ord.csv', index=False)