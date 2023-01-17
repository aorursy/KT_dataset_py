import numpy as np

import pandas as pd

import xgboost as xgb

import matplotlib.pyplot as plt
df_train = pd.read_csv("../input/vt-nh-real-estate/train.csv")

df_test = pd.read_csv("../input/vt-nh-real-estate/test.csv")



df_train.columns
ax = df_train['price_closed'].hist(bins=50)
# ylog will be log(1+y), as suggested by https://github.com/dmlc/xgboost/issues/446#issuecomment-135555130

ylog_train_all = np.log1p(df_train['price_closed'].values)

id_test = df_test['id']

print(len(ylog_train_all))

df_train.drop(['id', 'garage_type', 'total_stories', 'surveyed', 'seasonal', 'water_body_type', 'water_frontage_length', 'short_sale', 'rooms_total', 'garage', 'flood_zone', 'easements', 'current_use', 'covenants', 'common_land_acres', 'basement_access_type', 'basement', 'price_closed'], axis=1, inplace=True)

df_test.drop(['id', 'garage_type', 'total_stories', 'surveyed', 'seasonal', 'water_body_type', 'water_frontage_length', 'short_sale', 'rooms_total', 'garage', 'flood_zone', 'easements', 'current_use', 'covenants', 'common_land_acres', 'basement_access_type', 'basement'], axis=1, inplace=True)



# Build df_all = (df_train+df_test).join(df_macro)

num_train = len(df_train)

print(num_train)

df_all = pd.concat([df_train, df_test])

print(df_all.shape)
# Deal with categorical values

df_numeric = df_all.select_dtypes(exclude=['object'])

df_obj = df_all.select_dtypes(include=['object']).copy()



for c in df_obj:

    df_obj[c] = pd.factorize(df_obj[c])[0]



df_values = pd.concat([df_numeric, df_obj], axis=1)
# Convert to numpy values

X_all = df_values.values

print(X_all.shape)



# Create a validation set, with last 20% of data

num_val = 23



X_train_all = X_all[:num_train]

X_train = X_all[:num_train]

X_val = X_all[num_train-num_val:num_train]

ylog_train = ylog_train_all

ylog_val = ylog_train_all[-num_val:]



X_test = X_all[num_train:]



df_columns = df_values.columns



print('X_train_all shape is', X_train_all.shape)

print('X_train shape is', X_train.shape)

print('y_train shape is', ylog_train.shape)

print('X_val shape is', X_val.shape)

print('y_val shape is', ylog_val.shape)

print('X_test shape is', X_test.shape)
dtrain_all = xgb.DMatrix(X_train_all, ylog_train_all, feature_names=df_columns)

dtrain = xgb.DMatrix(X_train, ylog_train, feature_names=df_columns)

dval = xgb.DMatrix(X_val, ylog_val, feature_names=df_columns)

dtest = xgb.DMatrix(X_test, feature_names=df_columns)
xgb_params = {

    'eta': 0.05,

    'max_depth': 5,

    'subsample': .5,

    'colsample_bytree': 0.7,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'silent': 1

}



# Uncomment to tune XGB `num_boost_rounds`

partial_model = xgb.train(xgb_params, dtrain, num_boost_round=1000, evals=[(dval, 'val')],

                       early_stopping_rounds=20, verbose_eval=20)



num_boost_round = partial_model.best_iteration
fig, ax = plt.subplots(1, 1, figsize=(8, 16))

xgb.plot_importance(partial_model, max_num_features=50, height=0.5, ax=ax)
num_boost_round = partial_model.best_iteration
model = xgb.train(dict(xgb_params, silent=0), dtrain_all, num_boost_round=num_boost_round)
fig, ax = plt.subplots(1, 1, figsize=(8, 16))

xgb.plot_importance(model, max_num_features=50, height=0.5, ax=ax)
ylog_pred = model.predict(dtest)

y_pred = np.exp(ylog_pred) - 1



df_sub = pd.DataFrame({'id': id_test, 'price_closed': y_pred})



df_sub.to_csv('sub.csv', index=False)