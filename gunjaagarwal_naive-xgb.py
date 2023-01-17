import numpy as np

import pandas as pd

import xgboost as xgb

import matplotlib.pyplot as plt
df_train = pd.read_csv("../input/train.csv", parse_dates=['timestamp'])

df_test = pd.read_csv("../input/test.csv", parse_dates=['timestamp'])

df_macro = pd.read_csv("../input/macro.csv", parse_dates=['timestamp'])
df_train.sample(3)



sample = df_train.sample(frac=0.5)



price_per_sq = sample.price_doc / sample['full_sq']

price_per_sq = price_per_sq[ np.isinf(price_per_sq) == False ].mean()

# create the price_by_sq parameters by using the train data, cross validation will do the same with inner train and inner validation

df_train['price_by_sq'] =df_train['full_sq'] * price_per_sq

df_test['price_by_sq'] = df_test['full_sq'] * price_per_sq

y_train = df_train['price_doc'].values

id_test = df_test['id']



df_train.drop(['id', 'price_doc'], axis=1, inplace=True)

df_test.drop(['id'], axis=1, inplace=True)



# Build df_all = (df_train+df_test).join(df_macro)

num_train = len(df_train)

df_all = pd.concat([df_train, df_test])

df_all = df_all.join(df_macro, on='timestamp', rsuffix='_macro')

print(df_all.shape)

multiplier = 0.960



#clean data

bad_index = df_all[df_all.life_sq > df_all.full_sq].index

df_all.ix[bad_index, "life_sq"] = np.NaN

equal_index = [601,1896,2791]

bad_index = df_all[df_all.life_sq < 5].index

df_all.ix[bad_index, "life_sq"] = np.NaN

bad_index = df_all[df_all.full_sq < 5].index

df_all.ix[bad_index, "full_sq"] = np.NaN

kitch_is_build_year = [13117]

df_all.ix[kitch_is_build_year, "build_year"] = df_all.ix[kitch_is_build_year, "kitch_sq"]

bad_index = df_all[df_all.kitch_sq >= df_all.life_sq].index

df_all.ix[bad_index, "kitch_sq"] = np.NaN

bad_index =df_all[(df_all.kitch_sq == 0).values + (df_all.kitch_sq == 1).values].index

df_all.ix[bad_index, "kitch_sq"] = np.NaN

bad_index = df_all[(df_all.full_sq > 210) & (df_all.life_sq / df_all.full_sq < 0.3)].index

df_all.ix[bad_index, "full_sq"] = np.NaN

bad_index = df_all[df_all.life_sq > 300].index

df_all.ix[bad_index, ["life_sq", "full_sq"]] = np.NaN

df_all.product_type.value_counts(normalize= True)

bad_index = df_all[df_all.build_year < 1500].index

df_all.ix[bad_index, "build_year"] = np.NaN

bad_index = df_all[df_all.num_room == 0].index 

df_all.ix[bad_index, "num_room"] = np.NaN

bad_index = [10076, 11621, 17764, 19390, 24007, 26713, 29172,3174, 7313]

df_all.ix[bad_index, "num_room"] = np.NaN

#bad_index = [3174, 7313]

bad_index =df_all[(df_all.floor == 0).values * (df_all.max_floor == 0).values].index

df_all.ix[bad_index, ["max_floor", "floor"]] = np.NaN

bad_index = df_all[df_all.floor == 0].index

df_all.ix[bad_index, "floor"] = np.NaN

bad_index = df_all[df_all.max_floor == 0].index

df_all.ix[bad_index, "max_floor"] = np.NaN

bad_index = df_all[df_all.floor > df_all.max_floor].index

df_all.ix[bad_index, "max_floor"] = np.NaN

df_all.floor.describe(percentiles= [0.9999])

bad_index = [23584]

df_all.ix[bad_index, "floor"] = np.NaN

df_all.material.value_counts()

df_all.state.value_counts()

bad_index = df_all[df_all.state == 33].index

df_all.ix[bad_index, "state"] = np.NaN

# Add month-year

month_year = (df_all.timestamp.dt.month + df_all.timestamp.dt.year * 100)

month_year_cnt_map = month_year.value_counts().to_dict()

df_all['month_year_cnt'] = month_year.map(month_year_cnt_map)



# Add week-year count

week_year = (df_all.timestamp.dt.weekofyear + df_all.timestamp.dt.year * 100)

week_year_cnt_map = week_year.value_counts().to_dict()

df_all['week_year_cnt'] = week_year.map(week_year_cnt_map)



# Add month and day-of-week

df_all['month'] = df_all.timestamp.dt.month

df_all['dow'] = df_all.timestamp.dt.dayofweek



# Other feature engineering

df_all['rel_floor'] = df_all['floor'] / df_all['max_floor'].astype(float)

df_all['rel_kitch_sq'] = df_all['kitch_sq'] / df_all['full_sq'].astype(float)



# Add apartment id (as suggested in https://www.kaggle.com/c/sberbank-russian-housing-market/discussion/33269)

# and replace it with its count and count by month

df_all['apartment_name'] = pd.factorize(df_all.sub_area + df_all['metro_km_avto'].astype(str))[0]

df_all['apartment_name_month_year'] = pd.factorize(df_all['apartment_name'].astype(str) + month_year.astype(str))[0]



df_all['apartment_name_cnt'] = df_all['apartment_name'].map(df_all['apartment_name'].value_counts())

df_all['apartment_name_month_year_cnt'] = df_all['apartment_name_month_year'].map(df_all['apartment_name_month_year'].value_counts())



# Remove timestamp column (may overfit the model in train)

df_all.drop(['timestamp', 'timestamp_macro'], axis=1, inplace=True)
factorize = lambda t: pd.factorize(t[1])[0]



df_obj = df_all.select_dtypes(include=['object'])



X_all = np.c_[

    df_all.select_dtypes(exclude=['object']).values,

    np.array(list(map(factorize, df_obj.iteritems()))).T

]

print(X_all.shape)



X_train = X_all[:num_train]

X_test = X_all[num_train:]
# Deal with categorical values

df_numeric = df_all.select_dtypes(exclude=['object'])

df_obj = df_all.select_dtypes(include=['object']).copy()



for c in df_obj:

    df_obj[c] = pd.factorize(df_obj[c])[0]



df_values = pd.concat([df_numeric, df_obj], axis=1)
# Convert to numpy values

X_all = df_values.values

print(X_all.shape)



X_train = X_all[:num_train]

X_test = X_all[num_train:]



df_columns = df_values.columns
xgb_params = {

    'eta': 0.05,

    'max_depth': 5,

    'subsample': 0.7,

    'colsample_bytree': 0.7,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'silent': 1

}



dtrain = xgb.DMatrix(X_train, y_train, feature_names=df_columns)

dtest = xgb.DMatrix(X_test, feature_names=df_columns)
# Uncomment to tune XGB `num_boost_rounds`



#cv_result = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,

#    verbose_eval=10, show_stdv=False)

#cv_result[['train-rmse-mean', 'test-rmse-mean']].plot()

#num_boost_rounds = len(cv_result)

#print("num_boost_rounds:", num_boost_rounds)



num_boost_round = 489
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_round)
y_pred = model.predict(dtest)

df_sub = pd.DataFrame({'id': id_test, 'price_doc': y_pred})

df_sub.to_csv('sub_modified.csv', index=False)