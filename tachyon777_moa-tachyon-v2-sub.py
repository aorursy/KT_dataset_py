DEBUG = False
import os

import numpy as np

import pandas as pd

from sklearn import preprocessing

from sklearn.metrics import log_loss

from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb

from tqdm import tqdm_notebook as tqdm

import time

import pickle



import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv("../input/lish-moa/train_features.csv")

all_train_len = len(train)

test = pd.read_csv("../input/lish-moa/test_features.csv")

train_targets_scored = pd.read_csv("../input/lish-moa/train_targets_scored.csv")

train_targets_nonscored = pd.read_csv("../input/lish-moa/train_targets_nonscored.csv")

sub = pd.read_csv("../input/lish-moa/sample_submission.csv")



models_dir = "../input/moa-tachyon-v2-models"



if DEBUG:

    train = train[:1000]

    train_targets_scored = train_targets_scored[:1000]

    test = test[:1000]

    sub = sub[:1000]
def label_encoding(train: pd.DataFrame, test: pd.DataFrame, encode_cols):

    n_train = len(train)

    train = pd.concat([train, test], sort=False).reset_index(drop=True)

    for f in encode_cols:

        try:

            lbl = preprocessing.LabelEncoder()

            train[f] = lbl.fit_transform(list(train[f].values))

        except:

            print(f)

    test = train[n_train:].reset_index(drop=True)

    train = train[:n_train]

    return train, test
#features

drop_features = ["sig_id","cp_type"]
res = train["cp_type"]=="trt_cp"

train = train[res].reset_index(drop=True)

train_targets_scored = train_targets_scored[res].reset_index(drop=True)

train, test = label_encoding(train, test, ['cp_dose','cp_time'])



categorical_features = ['cp_dose','cp_time']
X_test = test.drop(drop_features, axis=1)
def prediction(models,target_col):

    y_preds = []

    for model in models:

        y_pred = model.predict(X_test,

                               num_iteration=model.best_iteration)

    y_preds.append(y_pred)

    return sum(y_preds) / len(y_preds)
for target_col in train_targets_scored.columns[:3] if DEBUG else train_targets_scored.columns:

    if target_col == "sig_id":continue

    with open(os.path.join(models_dir,target_col+".pickle"),mode="rb") as fp:

        models = pickle.load(fp)

    print("Now predicting : "+target_col)

    preds = prediction(models,target_col)

    sub[target_col] = preds
for idx,i in enumerate(test["cp_type"]):

    if i == "ctl_vehicle":

        sub.loc[idx,1:] = 0 #全て0に
for idx,i in enumerate(train_targets_scored.columns):

    if i == "sig_id":continue

    if sum(train_targets_scored[i]) < 5:

        sub[i] = 0

    
sub.to_csv('submission.csv', index=False)
sub.head()