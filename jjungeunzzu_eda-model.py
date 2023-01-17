import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import operator

#import gc

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import describe

%matplotlib inline



from sklearn.model_selection import KFold

from sklearn.preprocessing import LabelEncoder



from sklearn.metrics import mean_squared_error

import lightgbm as lgb

import xgboost as xgb
print(os.listdir("../input"))
train_df = pd.read_csv('../input/train.csv')

train_df.head()
train_df.shape
train_df.info()
train_df.isnull().values.sum(axis=0)
train_df_describe = train_df.describe()

train_df_describe
test_df = pd.read_csv('../input/test.csv')

test_df.head()
test_df.shape
test_df.info()
test_df.isnull().values.sum(axis=0)
test_df_describe = test_df.describe()

test_df_describe
plt.figure(figsize=(12, 5))

plt.hist(train_df.Target.values, bins=4)

plt.title('Histogram target counts')

plt.xlabel('Count')

plt.ylabel('Target')

plt.show()
np.unique(train_df.Target.values)
pd.value_counts(train_df.Target)
columns_to_use = train_df.columns[1:-1]
columns_to_use
y = train_df['Target'].values-1
train_test_df = pd.concat([train_df[columns_to_use], test_df[columns_to_use]], axis=0)

cols = [f_ for f_ in train_test_df.columns if train_test_df[f_].dtype == 'object']



for col in cols:

    le = LabelEncoder()

    le.fit(train_test_df[col].astype(str))

    train_df[col] = le.transform(train_df[col].astype(str))

    test_df[col] = le.transform(test_df[col].astype(str))

del le
train = lgb.Dataset(train_df[columns_to_use].astype('float'),y ,feature_name = "auto")
params = {

    'boosting_type': 'gbdt',

    'objective': 'multiclass',

    'metric': 'multi_logloss',

    'max_depth': 5,

    'num_leaves': 31,

    'learning_rate': 0.05,

    'feature_fraction': 0.9,

    'bagging_fraction': 0.6,

    'bagging_freq': 5,

    'verbose': -1,

    'num_threads': 6,

    'lambda_l2': 1.0,

    'min_gain_to_split': 0,

    'num_class': len(np.unique(y)),

}
clf = lgb.train(params,

        train,

        num_boost_round = 500,

        verbose_eval=True)
preds1 = clf.predict(test_df[columns_to_use])

xgb_params = {

        'learning_rate': 0.1,

        'n_estimators': 1000,

        'max_depth': 5,

        'min_child_weight': 1,

        'gamma': 0,

        'subsample': 0.9,

        'colsample_bytree': 0.84,

        'objective': 'multi:softprob',

        'scale_pos_weight': 1,

        'eval_metric': 'merror',

        'silent': 1,

        'verbose': False,

        'num_class': 4,

        'seed': 44}

    

d_train = xgb.DMatrix(train_df[columns_to_use].values.astype('float'), y)

d_test = xgb.DMatrix(test_df[columns_to_use].values.astype('float'))

    

model = xgb.train(xgb_params, d_train, num_boost_round = 500, verbose_eval=100)

                        

xgb_pred = model.predict(d_test)
xgb_pred.shape
preds = 0.5*preds1 + 0.5*xgb_pred



preds = np.argmax(preds, axis = 1) +1

preds
sample_submission = pd.read_csv("../input/sample_submission.csv")

sample_submission.head()
sample_submission['Target'] = preds

sample_submission.to_csv('simple_lgbm_xgb_1.csv', index=False)

sample_submission.head()
np.mean(preds)