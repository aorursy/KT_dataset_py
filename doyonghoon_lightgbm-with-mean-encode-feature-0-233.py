# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('precision', 5)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
tr = pd.read_csv('../input/train.csv')
te = pd.read_csv('../input/test.csv')
print('train data shape is :', tr.shape)
print('test data shape is :', te.shape)
data = pd.concat([tr, te], axis=0)
data.shape
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from tqdm import tqdm
data.activation_date = pd.to_datetime(data.activation_date)
tr.activation_date = pd.to_datetime(tr.activation_date)

data['day_of_month'] = data.activation_date.apply(lambda x: x.day)
data['day_of_week'] = data.activation_date.apply(lambda x: x.weekday())

tr['day_of_month'] = tr.activation_date.apply(lambda x: x.day)
tr['day_of_week'] = tr.activation_date.apply(lambda x: x.weekday())
data['char_len_title'] = data.title.apply(lambda x: len(str(x)))
data['char_len_desc'] = data.description.apply(lambda x: len(str(x)))
agg_cols = ['region', 'city', 'parent_category_name', 'category_name',
            'image_top_1', 'user_type','item_seq_number','day_of_month','day_of_week'];
for c in tqdm(agg_cols):
    gp = tr.groupby(c)['deal_probability']
    mean = gp.mean()
    std  = gp.std()
    data[c + '_deal_probability_avg'] = data[c].map(mean)
    data[c + '_deal_probability_std'] = data[c].map(std)

for c in tqdm(agg_cols):
    gp = tr.groupby(c)['price']
    mean = gp.mean()
    data[c + '_price_avg'] = data[c].map(mean)
cate_cols = ['city',  'category_name', 'user_type',]
for c in cate_cols:
    data[c] = LabelEncoder().fit_transform(data[c].values)
new_data = data.drop(['user_id','description','image','parent_category_name','region',
                      'item_id','param_1','param_2','param_3','title'], axis=1)
import gc
del data
del tr
del te
gc.collect()
from sklearn.model_selection import train_test_split
X = new_data.loc[new_data.activation_date<=pd.to_datetime('2017-04-07')]
X_te = new_data.loc[new_data.activation_date>=pd.to_datetime('2017-04-08')]

y = X['deal_probability']
X = X.drop(['deal_probability','activation_date'],axis=1)
X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=2018)
X_te = X_te.drop(['deal_probability','activation_date'],axis=1)

print(X_tr.shape, X_va.shape, X_te.shape)


del X
del y
gc.collect()
# Create the LightGBM data containers
tr_data = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cate_cols)
va_data = lgb.Dataset(X_va, label=y_va, categorical_feature=cate_cols, reference=tr_data)
del X_tr
del X_va
del y_tr
del y_va
gc.collect()

# Train the model
parameters = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 50
}


model = lgb.train(parameters,
                  tr_data,
                  valid_sets=va_data,
                  num_boost_round=300,
                  early_stopping_rounds=120,
                  verbose_eval=50)
y_pred = model.predict(X_te)
sub = pd.read_csv('../input/sample_submission.csv')
sub['deal_probability'] = y_pred
sub['deal_probability'].clip(0.0, 1.0, inplace=True)
sub.to_csv('lgb_with_mean_encode.csv', index=False)
sub.head()
lgb.plot_importance(model, importance_type='gain', figsize=(10,20))