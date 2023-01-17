# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import lightgbm as lgb
from tqdm import tqdm

pd.set_option('precision', 5)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
tr = pd.read_csv('../input/train.csv',parse_dates=["activation_date"],infer_datetime_format=True)
te = pd.read_csv('../input/test.csv',parse_dates=["activation_date"],infer_datetime_format=True)
print('train data shape is :', tr.shape)
print('test data shape is :', te.shape)
tr.head()
tr.describe()
tr.describe(include=["O"])
data = pd.concat([tr, te], axis=0)
print("merged shape:",data.shape)
print("unique item_seq_number:", len(set(data.item_seq_number)))
print("unique image_top_1:", len(set(data.image_top_1)))
# data['day_of_month'] = data.activation_date.apply(lambda x: x.day)
# data['day_of_week'] = data.activation_date.apply(lambda x: x.weekday())

# tr['day_of_month'] = tr.activation_date.apply(lambda x: x.day)
# tr['day_of_week'] = tr.activation_date.apply(lambda x: x.weekday())
agg_cols = ['region', 'city', 'parent_category_name', 'category_name',
            "param_1","param_2",'image_top_1', 'user_type','item_seq_number']
max_cols = ["category_name", 'region',"city", "param_1","param_2"
            ,'item_seq_number'
#             'image_top_1'
           ]

for c in tqdm(agg_cols):
    gp = tr.groupby(c)['deal_probability']
    mean = gp.mean()
    std  = gp.std()
    maximum = gp.max()
    data[c + '_deal_proba_avg'] = data[c].map(mean)
    data[c + '_deal_proba_std'] = data[c].map(std)
    if c in max_cols:
        data[c + '_deal_proba_max_one'] = data[c].map(maximum) # added , may easily overfit!
        data[c + '_deal_proba_max_one'] = data[c + '_deal_proba_max_one']>0.6

for c in tqdm(agg_cols):
    gp = tr.groupby(c)['price']
    mean = gp.mean()
    data[c + '_price_avg'] = data[c].map(mean)
data.shape
print("tr (orig)",tr.shape , "\n te (orig)",te.shape)
tr = data.loc[~data.deal_probability.isnull()]
te = data.loc[data.deal_probability.isnull()].drop("deal_probability",axis=1)
print("tr (new)",tr.shape , "\n te (new)",te.shape)
tr.info()
tr.to_csv("avitoAdDemand-train-augV1.csv.gz",index=False,compression="gzip")
te.to_csv("avitoAdDemand-test-augV1.csv.gz",index=False,compression="gzip")
# new_data = data.drop(['user_id','description','image',
#                       'item_id','param_1','param_2','param_3','title'], axis=1)
# import gc
# del data
# del tr
# del te
# gc.collect()

# from sklearn.model_selection import train_test_split

# X = new_data.loc[new_data.activation_date<=pd.to_datetime('2017-04-07')]
# X_te = new_data.loc[new_data.activation_date>=pd.to_datetime('2017-04-08')]

# y = X['deal_probability']
# X = X.drop(['deal_probability','activation_date'],axis=1)
# X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=2018)
# X_te = X_te.drop(['deal_probability','activation_date'],axis=1)

# print(X_tr.shape, X_va.shape, X_te.shape)


# del X
# del y
# gc.collect()
# # Create the LightGBM data containers
# tr_data = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cate_cols)
# va_data = lgb.Dataset(X_va, label=y_va, categorical_feature=cate_cols, reference=tr_data)
# del X_tr
# del X_va
# del y_tr
# del y_va
# gc.collect()

# # Train the model
# parameters = {
#     'task': 'train',
#     'boosting_type': 'gbdt',
#     'objective': 'regression',
#     'metric': 'rmse',
#     'num_leaves': 31,
#     'learning_rate': 0.05,
#     'feature_fraction': 0.9,
#     'bagging_fraction': 0.8,
#     'bagging_freq': 5,
#     'verbose': 50
# }


# model = lgb.train(parameters,
#                   tr_data,
#                   valid_sets=va_data,
#                   num_boost_round=2000,
#                   early_stopping_rounds=120,
#                   verbose_eval=50)
# y_pred = model.predict(X_te)
# sub = pd.read_csv('../input/sample_submission.csv')
# sub['deal_probability'] = y_pred
# sub['deal_probability'].clip(0.0, 1.0, inplace=True)
# sub.to_csv('lgb_with_mean_encode.csv', index=False)
# sub.head()
# lgb.plot_importance(model, importance_type='gain', figsize=(10,20))
