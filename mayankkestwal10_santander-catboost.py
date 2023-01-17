# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
df_train.head()
train_df = df_train.copy()

test_df = df_test.copy()

train_df.drop(columns=["ID_code", "target"], inplace=True)

test_df.drop(columns=["ID_code"], inplace=True)

target = df_train.target
def augment_train(df_train, y_train):   

    t0 = df_train[y_train == 0].copy()

    t1 = df_train[y_train == 1].copy()

    i = 0

    N = 3

    for I in range(2):  # augment data into 2x

        for col in df_train.columns:

            i = i + 1000

            np.random.seed(i)

            np.random.shuffle(t0[col].values)

            np.random.shuffle(t1[col].values)

        df_train = pd.concat([df_train, t0.copy()])

        df_train = pd.concat([df_train, t1.copy()])

        y_train = pd.concat([y_train, pd.Series([0] * t0.shape[0]), pd.Series([1] * t1.shape[0])])

    return df_train, y_train
from sklearn.model_selection import StratifiedKFold, KFold

from sklearn.metrics import roc_auc_score

from catboost import Pool, CatBoostClassifier



model = CatBoostClassifier(subsample=0.36, #rawdata 0.5  ×2 0.45 ×3 0.36

                            custom_loss='Logloss',

                           random_strength = 0,

                           max_depth=3,

                           eval_metric="AUC",

                           learning_rate=0.01,

                           iterations=60000,

                           #class_weights=[1,2],

                           bootstrap_type='Bernoulli',

                           #rsm=0.045,

                           l2_leaf_reg=0.3,

                           task_type="GPU",

                           random_seed=432013,

                           od_type="Iter",

                           border_count=128

                           #has_time= True 

                          )
def run_cat(model,  trt, tst, tar,n_splits=11, plot=False):   

    kf = KFold(n_splits=n_splits, random_state=432013, shuffle=True)

    oof = np.zeros(len(trt))

    feature_importance_df = pd.DataFrame()

    y_valid_pred = 0 * tar

    y_test_pred = 0

    for n_fold, (train_index, valid_index) in enumerate(kf.split(trt, tar)):

        y_train, y_valid = tar.iloc[train_index], tar.iloc[valid_index]

        X_train, X_valid = trt.iloc[train_index,:], trt.iloc[valid_index,:]

        X_train, y_train = augment_train(X_train, y_train)

        _train = Pool(X_train, label=y_train)

        _valid = Pool(X_valid, label=y_valid)

        print( "Fold ", n_fold)

        fit_model = model.fit(_train,

                              verbose_eval=1000, 

                              early_stopping_rounds=1000,

                              eval_set=[_valid],

                              use_best_model=True,

                              plot=False,

                                            

                             )

        pred = fit_model.predict_proba(X_valid)[:,1]

        oof[valid_index] = pred

        print( "auc = ", roc_auc_score(y_valid, pred) )

        y_valid_pred.iloc[valid_index] = pred

        y_test_pred += fit_model.predict_proba(tst)[:,1]

    y_test_pred /= n_splits

    print("average auc:", roc_auc_score(tar, oof))

    return y_test_pred, oof
y_test_pred, oof = run_cat(model,train_df, test_df, target)
submission = pd.read_csv("../input/sample_submission.csv")

submission['target'] = y_test_pred

pd.Series(oof).to_csv("Cat_oof.csv", index = False)

submission.to_csv('submission.csv', index=False)