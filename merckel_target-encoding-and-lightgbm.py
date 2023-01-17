import pandas as pd

import category_encoders as ce

import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold
train = pd.read_csv('../input/cat-in-the-dat/train.csv')

test = pd.read_csv('../input/cat-in-the-dat/test.csv')



train.sort_index(inplace=True)

train_y = train['target']

test_id = test['id']

train.drop(['target', 'id'], axis=1, inplace=True)

test.drop('id', axis=1, inplace=True)
cat_feat_to_encode = train.columns.tolist()

smoothing=50.0



oof = pd.DataFrame([])

for tr_idx, oof_idx in StratifiedKFold(

    n_splits=5, random_state=1, shuffle=True).split(

        train, train_y):

    ce_target_encoder = ce.TargetEncoder(cols = cat_feat_to_encode, smoothing=smoothing)

    ce_target_encoder.fit(train.iloc[tr_idx, :], train_y.iloc[tr_idx])

    oof = oof.append(ce_target_encoder.transform(train.iloc[oof_idx, :]), ignore_index=False)



ce_target_encoder = ce.TargetEncoder(cols = cat_feat_to_encode, smoothing=smoothing)

ce_target_encoder.fit(train, train_y)

train = oof.sort_index() 

test = ce_target_encoder.transform(test)
clf = lgb.train(

    params={

        'max_depth': 2, 

        'num_leaves': 150,

        'reg_alpha': 0.6, 

        'reg_lambda': 0.6,

        'objective': 'binary',

        "boosting_type": "gbdt",

        "metric": 'auc',

        "verbosity": -1,

        'random_state': 1},

    train_set=lgb.Dataset(train, label=train_y),

    num_boost_round=700)
from datetime import datetime

pd.DataFrame({'id': test_id, 'target': clf.predict(test)}).to_csv(

    'sub_' + str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + '.csv', 

    index=False)