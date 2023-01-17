import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

import warnings
warnings.filterwarnings('ignore')

# Any results you write to the current directory are saved as output.
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import lightgbm as lgb

import gc
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
category_cols = ['region', 'city', 'parent_category_name', 'category_name',\
                 'param_1', 'param_2', 'param_3', 'user_type']
numerical_cols = ['price', 'item_seq_number']
train_data = train.loc[:, train.columns.isin(category_cols + numerical_cols)]
test_data = test.loc[:, test.columns.isin(category_cols + numerical_cols)]
train_y = train['deal_probability']
train_data.isnull().sum()
test_data.isnull().sum()
train_data['param_1'].fillna(value='missing', inplace=True)
train_data['param_2'].fillna(value='missing', inplace=True)
train_data['param_3'].fillna(value='missing', inplace=True)
test_data['param_1'].fillna(value='missing', inplace=True)
test_data['param_2'].fillna(value='missing', inplace=True)
test_data['param_3'].fillna(value='missing', inplace=True)
data_df = pd.concat([train_data, test_data])

for col in category_cols:
    train_data_index = train_data.shape[0]
    le_col_data = LabelEncoder().fit_transform(data_df[col])
    train_data[col+'_le'] = le_col_data[:train_data_index]
    test_data[col+'_le'] = le_col_data[train_data_index:]
del data_df
gc.collect()
train_data.drop(category_cols, axis=1, inplace=True)
test_data.drop(category_cols, axis=1, inplace=True)
train_data['price'].fillna(0, inplace=True)
test_data['price'].fillna(0, inplace=True)
train_data['price'] = np.log1p(train_data['price'])
test_data['price'] = np.log1p(test_data['price'])
train_data['item_seq_number'] = np.log1p(train_data['item_seq_number'])
test_data['item_seq_number'] = np.log1p(test_data['item_seq_number'])
X_train, X_valid, y_train, y_valid = train_test_split(train_data, train_y, test_size=0.3)
print(X_train.shape)
print(X_valid.shape)
print(y_train.shape)
print(y_valid.shape)
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_iterations': 1000,
    'learning_rate': 0.07,
    'num_leaves': 50,
    'feature_fraction': 0.6,
    'lambda_l1': 0.05,
    'min_gain_to_split': 0.009,
    'bagging_fraction': 0.9
}

lgb_categorical_features = ['region_le', 'city_le', 'parent_category_name_le', 
                            'category_name_le', 'param_1_le', 'param_2_le', 
                            'param_3_le', 'user_type_le']

lgb_train_x = lgb.Dataset(X_train, label = y_train,  categorical_feature=lgb_categorical_features)
lgb_val_x = lgb.Dataset(X_valid, label=y_valid, categorical_feature=lgb_categorical_features)
eval_result = {}
lgb_model = lgb.train(params, lgb_train_x, valid_sets=[lgb_val_x], verbose_eval=50, evals_result=eval_result)
prediction = lgb_model.predict(test_data, num_iteration = lgb_model.best_iteration)
prediction[prediction < 0] = 0
prediction[prediction > 1] = 1
submission = pd.DataFrame({'item_id': test['item_id'], 'deal_probability': prediction})
submission[['item_id', 'deal_probability']].to_csv('simple_baseline_lgbm_gridtune.csv', index=False)
