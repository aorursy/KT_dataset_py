import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import xgboost as xgb

%matplotlib inline
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

df_macro = pd.read_csv('../input/macro.csv')
train_master = pd.merge(df_train, df_macro, on=['timestamp', 'timestamp'])

test_master = pd.merge(df_test, df_macro, on=['timestamp', 'timestamp'])

train_master = train_master.drop('sub_area', axis=1)

test_master = test_master.drop('sub_area', axis=1)
train_master.ix[train_master['product_type']=='Investment', 'product_type']=1

train_master.ix[train_master['product_type']!='Investment', 'product_type']=0

test_master.ix[test_master['product_type']=='Investment', 'product_type']=1

test_master.ix[test_master['product_type']!='Investment', 'product_type']=0



train_master.ix[train_master['child_on_acc_pre_school']=='#!', 'child_on_acc_pre_school']=np.nan

train_master.ix[train_master['child_on_acc_pre_school']!='#!', 'child_on_acc_pre_school']=np.nan

test_master.ix[test_master['child_on_acc_pre_school']=='#!', 'child_on_acc_pre_school']=np.nan

test_master.ix[test_master['child_on_acc_pre_school']!='#!', 'child_on_acc_pre_school']=np.nan



d_list = ['culture_objects_top_25', 'thermal_power_plant_raion', 'incineration_raion', 'oil_chemistry_raion',

          'radiation_raion', 'railroad_terminal_raion', 'big_market_raion', 'nuclear_reactor_raion',

          'detention_facility_raion', 'water_1line', 'big_road1_1line', 'railroad_1line']

for d in d_list:

    train_master.ix[train_master[d]=='yes', d]=1

    train_master.ix[train_master[d]=='no', d]=0

    test_master.ix[test_master[d]=='yes', d]=1

    test_master.ix[test_master[d]=='no', d]=0
plt.figure(figsize=(8,6))

plt.scatter(train_master['full_sq'], train_master['price_doc'])

plt.xlabel('sqm', fontsize=12)

plt.ylabel('price', fontsize=12)

plt.show()
df = train_master[train_master['full_sq'] < 5000]
plt.figure(figsize=(8,6))

plt.scatter(df['full_sq'], df['price_doc'])

plt.xlabel('sqm', fontsize=12)

plt.ylabel('price', fontsize=12)

plt.show()
corrmat = df.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, square=True)
n = 0

delete_list = []

for i, row in corrmat.iterrows():

    to_delete = False

    for col in row[0:n]:

        if col > 0.5 or col <-0.5:

            delete_list.append(i)

            break

    n += 1

delete_list.append('id')

delete_list.append('timestamp')

delete_list.append('ecology')

delete_list.append('child_on_acc_pre_school')

delete_list.append('modern_education_share')

delete_list.append('old_education_build_share')

delete_list.remove('price_doc')

dff = df.drop(delete_list, axis=1)

print(dff.shape)

var_list = []

for n in dff:

    var_list.append(n)

print(delete_list)
corrmat = dff.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, square=True)
y_train = dff['price_doc']

y_train = np.log1p(y_train.values)
try:

    var_list.remove('price_doc')

except:

    pass
x_train = dff[var_list]

x_train = x_train.values

x_test = test_master[var_list]

x_test = x_test.values
dtrain = xgb.DMatrix(x_train, y_train, feature_names=var_list)

dtest = xgb.DMatrix(x_test, feature_names=var_list)
xgb_params = {

    'eta': 0.05,

    'max_depth': 5,

    'subsample': 1.0,

    'colsample_bytree': 0.7,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'silent': 1,

}



tune_model = xgb.train(xgb_params, dtrain, num_boost_round=1000)

num_boost_round = tune_model.best_iteration
xgb_params['silent'] = 0

model = xgb.train(xgb_params, dtrain, num_boost_round=num_boost_round)
fig, ax = plt.subplots(1, 1, figsize=(8, 16))

xgb.plot_importance(tune_model, max_num_features=50, height=0.5, ax=ax)
y_hat = model.predict(dtest)

y_pred = np.exp(y_hat) - 1

df_submission = pd.DataFrame({'id': test_master['id'], 'price_doc': y_pred})

#df_submission.ix[df_submission['price_doc']<0, 'price_doc']=0

df_submission.to_csv('submission.csv', index=False)