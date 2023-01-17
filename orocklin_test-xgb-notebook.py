import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import model_selection, preprocessing

import xgboost as xgb

color = sns.color_palette()



%matplotlib inline



pd.options.mode.chained_assignment = None  # default='warn'

pd.set_option('display.max_columns', 500)
train_df = pd.read_csv("../input/train.csv", parse_dates=['timestamp'])

test_df = pd.read_csv("../input/test.csv", parse_dates=['timestamp'])

macro_df = pd.read_csv("../input/macro.csv", parse_dates=['timestamp'])



train_df = train_df.merge(macro_df, on='timestamp')

test_df = test_df.merge(macro_df, on='timestamp')
train_df.head()
test_df.head()
# Add month-year

month_year = (train_df.timestamp.dt.month + train_df.timestamp.dt.year * 100)

month_year_cnt_map = month_year.value_counts().to_dict()

train_df['month_year_cnt'] = month_year.map(month_year_cnt_map)

month_year = (test_df.timestamp.dt.month + test_df.timestamp.dt.year * 100)

month_year_cnt_map = month_year.value_counts().to_dict()

test_df['month_year_cnt'] = month_year.map(month_year_cnt_map)





# Add week-year count

week_year = (train_df.timestamp.dt.weekofyear + train_df.timestamp.dt.year * 100)

week_year_cnt_map = week_year.value_counts().to_dict()

train_df['week_year_cnt'] = week_year.map(week_year_cnt_map)

week_year = (test_df.timestamp.dt.weekofyear + test_df.timestamp.dt.year * 100)

week_year_cnt_map = week_year.value_counts().to_dict()

test_df['week_year_cnt'] = week_year.map(week_year_cnt_map)





# Add month and day-of-week

train_df['month'] = train_df.timestamp.dt.month

train_df['dow'] = train_df.timestamp.dt.dayofweek

test_df['month'] = test_df.timestamp.dt.month

test_df['dow'] = test_df.timestamp.dt.dayofweek





train_df['rel_floor'] = train_df['floor'] / train_df['max_floor'].astype(float)

train_df['rel_kitch_sq'] = train_df['kitch_sq'] / train_df['life_sq'].astype(float)

train_df['rel_life_full'] = train_df['life_sq'] / train_df['full_sq'].astype(float)



train_df.drop(["child_on_acc_pre_school"], axis=1, inplace=True)



test_df['rel_floor'] = test_df['floor'] / test_df['max_floor'].astype(float)

test_df['rel_kitch_sq'] = test_df['kitch_sq'] / test_df['life_sq'].astype(float)

test_df['rel_life_full'] = test_df['life_sq'] / test_df['full_sq'].astype(float)



test_df.drop(["child_on_acc_pre_school"], axis=1, inplace=True)
for f in train_df.columns:

    if train_df[f].dtype=='object':

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(train_df[f].values)) 

        train_df[f] = lbl.transform(list(train_df[f].values))

        

train_y = train_df.price_doc.values

train_X = train_df.drop(["id", "timestamp", "price_doc"], axis=1)



xgb_params = {

    'eta': 0.05,

    'max_depth': 8,

    'subsample': 0.7,

    'colsample_bytree': 0.7,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'silent': 1

}

dtrain = xgb.DMatrix(train_X, train_y, feature_names=train_X.columns.values)

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=400)



# plot the important features #

fig, ax = plt.subplots(figsize=(12,25))

xgb.plot_importance(model, max_num_features=30, height=0.8, ax=ax)

plt.show()
for f in test_df.columns:

    if test_df[f].dtype=='object':

        print (f)

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(test_df[f].values))

        print (test_df[f].values)

        test_df[f] = lbl.transform(list(test_df[f].values))



test_df.columns

id_test = test_df["id"]

test_X = test_df.drop(["id","timestamp"], axis=1)
dtest = xgb.DMatrix(test_X, feature_names=test_X.columns.values)



y_pred = model.predict(dtest)



res_df = pd.DataFrame({'id': id_test, 'price_doc': y_pred})



res_df.to_csv('result.csv', index=False)