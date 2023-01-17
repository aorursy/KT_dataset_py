DEBUG = False
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
def run_lgbm(target_col: str):

    

    X_train = train.drop(drop_features, axis=1)

    y_train = train_targets_scored[target_col]

    X_test = test.drop(drop_features, axis=1)

    y_preds = []

    models = []

    oof_train = np.zeros((len(X_train),))



    for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train,y_train)):

        X_tr = X_train.loc[train_index, :]

        X_val = X_train.loc[valid_index, :]

        y_tr = y_train[train_index]

        y_val = y_train[valid_index]



        lgb_train = lgb.Dataset(X_tr,

                                y_tr,

                                categorical_feature=categorical_features)



        lgb_eval = lgb.Dataset(X_val,

                               y_val,

                               reference=lgb_train,

                               categorical_feature=categorical_features)



        model = lgb.train(params,

                          lgb_train,

                          valid_sets=[lgb_train, lgb_eval],

                          verbose_eval=False,

                          num_boost_round=1000,

                          early_stopping_rounds=10)





        oof_train[valid_index] = model.predict(X_val,

                                               num_iteration=model.best_iteration)

        y_pred = model.predict(X_test,

                               num_iteration=model.best_iteration)



        y_preds.append(y_pred)

        models.append(model)

    

    with open(target_col + ".pickle", mode='wb') as fp:

        pickle.dump(models , fp)



    return oof_train, sum(y_preds) / len(y_preds)
cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)



params = {

    'num_leaves': 24,

    'max_depth': 5,

    'objective': 'binary',

    'learning_rate': 0.01,

    #'device': 'gpu',

    #'gpu_platform_id': 0,

    #'gpu_device_id': 0

}



categorical_cols = ['cp_type', 'cp_dose']

oof = train_targets_scored.copy()
for target_col in train_targets_scored.columns[:3] if DEBUG else train_targets_scored.columns:

    t1=time.time()

    if target_col == "sig_id":continue

    print("Training",target_col)

    _oof, _preds = run_lgbm(target_col)

    oof[target_col] = _oof

    sub[target_col] = _preds

    t2=time.time()

    print(t2-t1)
for idx,i in enumerate(test["cp_type"]):

    if i == "ctl_vehicle":

        sub.loc[idx,1:] = 0 #全て0に
scores = []

for target_col in train_targets_scored.columns[:3] if DEBUG else train_targets_scored.columns:

    if target_col != "sig_id":

        scores.append(log_loss(train_targets_scored[target_col], oof[target_col]))

print(np.sum(scores))
sub.head()
sub.to_csv('submission.csv', index=False)