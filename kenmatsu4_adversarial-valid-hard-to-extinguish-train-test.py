# Basic library

import pandas as pd

import pandas.io.sql as psql

import numpy as np

import numpy.random as rd

import gc

import multiprocessing as mpa

import os

import sys

import pickle

from collections import defaultdict

from glob import glob

import math

from datetime import datetime as dt

from pathlib import Path

import scipy.stats as st

import re



# Visualization

import matplotlib

from matplotlib import font_manager

import matplotlib.pyplot as plt

import matplotlib.cm as cm

from matplotlib import rc



from matplotlib import animation as ani

from IPython.display import Image



plt.rcParams["patch.force_edgecolor"] = True

#rc('text', usetex=True)

from IPython.display import display # Allows the use of display() for DataFrames

import seaborn as sns

sns.set(style="whitegrid", palette="muted", color_codes=True)

sns.set_style("whitegrid", {'grid.linestyle': '--'})

red = sns.xkcd_rgb["light red"]

green = sns.xkcd_rgb["medium green"]

blue = sns.xkcd_rgb["denim blue"]



pd.set_option("display.max_colwidth", 100)

pd.set_option("display.max_rows", None)

pd.set_option("display.max_columns", None)

pd.options.display.float_format = '{:,.5f}'.format



%matplotlib inline

#%config InlineBackend.figure_format='retina'



from lightgbm import LGBMClassifier

from sklearn.model_selection import StratifiedKFold, KFold

from sklearn.metrics import roc_auc_score

from scipy.stats import rankdata

from datetime import  datetime as dt

def current_time():

    return dt.strftime(dt.now(),'%Y-%m-%d %H:%M:%S')
print(os.listdir("../input"))
HOME_PATH = Path("../")

INPUT_PATH = Path(HOME_PATH/"input")

SAVE_PATH = Path(HOME_PATH/f"processed")

SAVE_PATH.mkdir(parents=True, exist_ok=True)
train = pd.read_csv(INPUT_PATH/"train.csv", index_col=0)

test = pd.read_csv(INPUT_PATH/"test.csv", index_col=0)

del train["target"]

train["is_train"] = 1

test["is_train"] = 0

df_all = pd.concat([train, test], axis=0)

del train, test



target = df_all.is_train.values

del df_all["is_train"]
df_all.shape
DEBUG = False



DATA_VERSION = "v001"

TRIAL_NO = "001"

FOLD_NUM = 5

STRATIFIED = True



HOME_PATH = Path("../")

INPUT_PATH = Path(HOME_PATH/"input")

SAVE_PATH = Path(HOME_PATH/f"processed")

SAVE_PATH.mkdir(parents=True, exist_ok=True)

def fit_predict(data: pd.DataFrame, target: np.array, colnames: list, categorical_features: list,

                params: dict, fold_seed=71):



    categorical_features = [c for c in categorical_features if c in data.columns]



    oof_preds = np.zeros(data.shape[0])

    clf_list = []

    auc_list = []

    feature_importance_df = pd.DataFrame()

    print(f"fold_seed: {fold_seed}")

    

    if STRATIFIED:

        folds = StratifiedKFold(n_splits=FOLD_NUM, shuffle=True, random_state=fold_seed)

        fold_iter = folds.split(data, target)

    else:

        folds = KFold(n_splits=FOLD_NUM, shuffle=False, random_state=fold_seed)

        fold_iter = folds.split(data)



    for n_fold, (trn_idx, val_idx) in enumerate(fold_iter):



        print(current_time(), "start cv {}".format(n_fold + 1))

        # Train lightgbm

        X_train = data.iloc[trn_idx]

        y_train = target[trn_idx]

        X_valid = data.iloc[val_idx]

        y_valid = target[val_idx]

        print(X_train.shape, y_train.shape)

        print(X_valid.shape, y_valid.shape)



        clf = LGBMClassifier(**params)



        if n_fold == 0:

            print(clf)

            clf_print = str(clf)



        clf.fit(X_train, y_train,

                eval_set=[(X_train, y_train,), (X_valid, y_valid)],

                # eval_metric=['logloss', 'auc'],

                verbose=50,

                early_stopping_rounds=300,

                categorical_feature=categorical_features)



        pred_valid = clf.predict_proba(X_valid, num_iteration=clf.best_iteration_)[:, 1]

        ranked_pred = rankdata(pred_valid)

        oof_preds[val_idx] = (ranked_pred - np.min(ranked_pred)) / (np.max(ranked_pred) - np.min(ranked_pred))



        fold_importance_df = pd.DataFrame()

        fold_importance_df["feature"] = colnames

        fold_importance_df["importance_split"] = clf.booster_.feature_importance(importance_type='split')

        fold_importance_df["importance_gain"] = clf.booster_.feature_importance(importance_type='gain')

        fold_importance_df["fold"] = n_fold + 1

        fold_importance_df.sort_values("importance_gain", ascending=False, inplace=True)

        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)



        try:

            auc_score = roc_auc_score(y_valid, oof_preds[val_idx])

        except Exception as e:

            print(e)

            if DEBUG:

                auc_score = np.zeros(len(val_idx))

            else:

                raise (e)



        print(f'Fold {n_fold+1} AUC : {auc_score:.6f}')



        auc_list.append(auc_score)

        clf_list.append(clf)

        del clf, X_train, y_train, X_valid, y_valid

        gc.collect()



    try:

        full_auc = roc_auc_score(target, oof_preds)

    except Exception as e:

        print(e)

        if DEBUG:

            full_auc = np.zeros(len(val_idx))

        else:

            raise (e)



    result_string = f"{DATA_VERSION}_{TRIAL_NO} \n"

    result_string += clf_print

    result_string += "\n"

    for i in range(len(auc_list)):

        result_string += f"Fold  {i+1} AUC : {auc_list[i]:.6f}\n"

    result_string += '---------\n'

    result_string += f'Full AUC score {full_auc:.6f} ({np.std(auc_list):.6f})\n'

    print(result_string)

    # send_message(result_string)



    with open(f'result_lgbm_{DATA_VERSION}_{TRIAL_NO}.txt', 'w') as f:

        f.write(result_string)



    return oof_preds, feature_importance_df, clf_list, full_auc



def importance_summary(feature_importance_df, only_gain=False):

    if only_gain:

        df_imp = pd.DataFrame(feature_importance_df.set_index(["feature", "fold"])["importance_gain"].unstack().mean(axis=1))

        df_imp.columns = ["gain"]

        return df_imp

    else:

        feature_importance_df["importance_split_normed"] = (feature_importance_df["importance_split"]-feature_importance_df["importance_split"].mean()) / feature_importance_df["importance_split"].std()

        feature_importance_df["importance_gain_normed"] = (feature_importance_df["importance_gain"]-feature_importance_df["importance_gain"].mean()) / feature_importance_df["importance_gain"].std()

        feature_importance_df["score"] = feature_importance_df["importance_split_normed"] + feature_importance_df["importance_gain_normed"]

        df_imp = pd.DataFrame(feature_importance_df.set_index(["feature", "fold"])["score"].unstack().mean(axis=1))

        df_imp.columns = ["score"]

        return df_imp

    

# original from: https://www.kaggle.com/ogrellier/good-fun-with-ligthgbm

def display_importances(feature_importance_df_):

    # Plot feature importances

    cols = feature_importance_df_[["feature", "importance_gain"]].groupby("feature").mean().sort_values(

        by="importance_gain", ascending=False)[:50].index

    

    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]

    

    plt.figure(figsize=(8,10))

    sns.barplot(x="importance_gain", y="feature", 

                data=best_features.sort_values(by="importance_gain", ascending=False))

    plt.title('LightGBM Features (avg over folds)')

    plt.tight_layout()

    plt.savefig('lgbm_importances.png')

lgb_params = {

    'objective': 'binary',

    'boosting_type': 'gbdt',

    'metric': 'auc',  # ['logloss', 'auc'],

    'learning_rate': 0.5, 

    'n_estimators': 10000,



    'max_depth': -1,

    'num_leaves': 24,

    'colsample_bytree': 0.8,

#     'min_child_samples': 33,

#     'min_child_weight': 11.1583,

#     'min_split_gain': 0.1,

#     'reg_alpha': 1.2456,

#     'reg_lambda': 0.1950,

    'subsample_freq': 1,  # 2,

    'subsample': 0.7,

#     'max_bin': 238,



    'seed': 71,

    'verbose': -1,

    'bagging_seed': 72,

    'feature_fraction_seed': 73,

    'drop_seed': 74,

    'n_jobs': -1

}
oof_preds, feature_importance_df, clf_list, full_auc = fit_predict(df_all, target, colnames=df_all.columns, categorical_features=[], params=lgb_params, fold_seed=123)
df_imp = importance_summary(feature_importance_df, only_gain=True).sort_values("gain", ascending=False)
df_imp.head(30)
display_importances(feature_importance_df)