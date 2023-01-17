import pandas as pd
import datetime
from sklearn.preprocessing import MinMaxScaler, Imputer

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

train['first_active_month'] = pd.to_datetime(train['first_active_month'])
test['first_active_month'] = pd.to_datetime(test['first_active_month'])
train['elapsed_time'] = (datetime.date(2018, 2, 1) - train['first_active_month'].dt.date).dt.days
test['elapsed_time'] = (datetime.date(2018, 2, 1) - test['first_active_month'].dt.date).dt.days

target = train['target']
train = train.drop(['target', 'first_active_month'], axis=1)
test = test.drop('first_active_month', axis=1)

imputer = Imputer(strategy='median')
scaler = MinMaxScaler(feature_range=(0, 1))
card_ids = test['card_id'].values
train = train.drop('card_id', axis= 1)
test = test.drop('card_id', axis = 1)
# Fit on the training data
imputer.fit(train)

# Transform both training and testing data
train = imputer.transform(train)
test = imputer.transform(test)

# Repeat with the scaler
scaler.fit(train)
train = scaler.transform(train)
test = scaler.transform(test)

import lightgbm as lgb
from sklearn.model_selection import train_test_split


train = pd.DataFrame(train)
test = pd.DataFrame(test)

train.info()

train.head()

test.head()



train[[0, 1, 2]] = train[[0, 1, 2]].astype('category')
test[[0, 1, 2]] = test[[0, 1, 2]].astype('category')

train.info()

X_tr, X_val, y_tr, y_val = train_test_split(train, target)



lgb_train = lgb.Dataset(X_tr, y_tr)
lgb_val = lgb.Dataset(X_val, y_val)

params ={
        'task': 'train',
        'boosting': 'goss',
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.01,
        'subsample': 0.9855232997390695,
        'max_depth': 7,
        'top_rate': 0.9064148448434349,
        'num_leaves': 63,
        'min_child_weight': 41.9612869171337,
        'other_rate': 0.0721768246018207,
        'reg_alpha': 9.677537745007898,
        'colsample_bytree': 0.5665320670155495,
        'min_split_gain': 9.820197773625843,
        'reg_lambda': 8.2532317400459,
        'min_data_in_leaf': 21,
        'verbose': 0,
        'seed':int(2**1),
        'bagging_seed':int(2**1),
        'drop_seed':int(2**1)
        }

lgbm_model = lgb.train(params, train_set = lgb_train, valid_sets = lgb_val)



predictions = lgbm_model.predict(test)

# Writing output to file
subm = pd.DataFrame()
subm['card_id'] = card_ids
subm['target'] = predictions
subm.to_csv('subLGBM.csv', index = False)
subm.head()