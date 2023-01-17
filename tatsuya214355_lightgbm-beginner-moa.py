# ライブラリインポート

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.preprocessing import LabelEncoder



import lightgbm as lgb

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.metrics import log_loss



import warnings

warnings.simplefilter('ignore')
# ファイルへのアクセス設定

ROOT = '../input/lish-moa/'



# ファイルのインポート

sample_sub = pd.read_csv(ROOT + 'sample_submission.csv' ,index_col='sig_id')

train_fe = pd.read_csv(ROOT + 'train_features.csv')

train_ta = pd.read_csv(ROOT + 'train_targets_nonscored.csv')

train_ta_score = pd.read_csv(ROOT + 'train_targets_scored.csv')

val_fe = pd.read_csv(ROOT + 'test_features.csv')
sample_sub
# 目的変数のリストを作成。予測時にはこれらリストの全てを予測していく

target_list = list(train_ta_score.columns)

target_list.remove('sig_id')

target_list
fe = pd.concat([train_fe, val_fe])
# objectのcp_type, cp_doseをlabelencodingを行う



le = LabelEncoder()

le = le.fit(fe['cp_type'])

fe['cp_type_label'] = le.transform(fe['cp_type']).astype('int8')

fe = fe.drop(['cp_type'], axis=1)



le = LabelEncoder()

le = le.fit(fe['cp_dose'])

fe['cp_does_label'] = le.transform(fe['cp_dose']).astype('int8')

fe = fe.drop(['cp_dose'], axis=1)
train_fe = fe.iloc[:23814]

val_fe = fe.iloc[23814:]
cat_cols = ['cp_does_label', 'cp_type_label']

feature_cols = list(train_fe.columns)

feature_cols.remove('sig_id')
train_fe[cat_cols] = train_fe[cat_cols].astype('category')

val_fe[cat_cols] = val_fe[cat_cols].astype('category')
lgb_params = {

                    'boosting_type': 'gbdt',

                    'objective': 'binary',

                    'metric': 'binary_logloss',

                    'subsample': 0.5,

                    'subsample_freq': 1,

                    'learning_rate': 0.01,

                    'num_leaves': 2**11-1,

                    'min_data_in_leaf': 2**12-1,

                    'feature_fraction': 0.5,

                    'n_estimators': 200,

                    'verbose': -1,

                    'seed':42,

                    'reg_alpha' : 0.3,

                    'reg_lambda' : 0.3

                } 
skf = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)
loss_list = []

for target in target_list:

    X = train_fe[feature_cols]

    y = train_ta_score[target]

    preds = np.zeros(val_fe.shape[0])

    for train_idx, test_idx in skf.split(X, y):

        tr_x, tr_y = X.iloc[train_idx], y.iloc[train_idx]

        te_x, te_y = X.iloc[test_idx], y.iloc[test_idx]

        

        train_data = lgb.Dataset(tr_x, tr_y)

        test_data = lgb.Dataset(te_x, te_y)

        estimator = lgb.train(lgb_params,

                             train_data,

                             valid_sets = test_data,

                             categorical_feature=cat_cols,

                             verbose_eval=-1

                             )

        test_pre = estimator.predict(te_x, num_iterations = estimator.best_iteration)

        try:

            loss = log_loss(te_y, test_pre)

        except:

            print('loss error')

        preds += estimator.predict(val_fe[feature_cols]) / skf.n_splits

        

        print('---------------------')

        print(target)

        print(loss)

        loss_list.append(loss)

        print(sum(loss_list)/len(loss_list))

    

    sample_sub[target]=preds

    #pd.DataFrame(preds).to_pickle(target+'_pred.pkl')

    

sample_sub.to_csv('submission.csv')
print('testデータのloss平均:')

print(sum(loss_list)/len(loss_list))