# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

from sklearn import model_selection, preprocessing, ensemble



import matplotlib.pyplot as plt

% matplotlib inline

import seaborn as sns

sns.set(style="whitegrid", color_codes=True)

sns.set(font_scale=1)



import plotly.plotly as py

import plotly.graph_objs as go

from plotly import tools





# Any results you write to the current directory are saved as output.
import os

import sys

import operator

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



import xgboost as xgb

import random

from sklearn import model_selection, preprocessing, ensemble

from sklearn.metrics import log_loss

from pandas import DataFrame

import seaborn as sns

color = sns.color_palette()

sns.set(style="whitegrid", color_codes=True)

sns.set(font_scale=1)
df_train=pd.read_csv('../input/train.csv',parse_dates=['timestamp'])

df_test=pd.read_csv('../input/test.csv',parse_dates=['timestamp'])

df_train["price_doc"].max()


df_train['price_doc'].ix[df_train['price_doc']>33333333] = 33333333

y_train=df_train.pop('price_doc').values

id_test = df_test['id']



num_train = len(df_train)

train_df = pd.concat([df_train, df_test])

train_df['month'] = train_df.timestamp.dt.month

train_df['year'] = train_df.timestamp.dt.year

train_df['Age_building']=2018-train_df['build_year']

ulimit = np.percentile(train_df.Age_building.values, 99)

train_df['Age_building'].ix[train_df['Age_building']>ulimit] = ulimit

train_df['my_comb']=(train_df['year']-2000)*12+train_df['month']

train_df['mth_id']=160

train_df['mth_id'][train_df['my_comb'] > 160] = train_df['my_comb']

# Add month-year

month_year = (train_df.timestamp.dt.month + train_df.timestamp.dt.year * 100)

month_year_cnt_map = month_year.value_counts().to_dict()

train_df['month_year_cnt'] = month_year.map(month_year_cnt_map)



# Add week-year count

week_year = (train_df.timestamp.dt.weekofyear + train_df.timestamp.dt.year * 100)

week_year_cnt_map = week_year.value_counts().to_dict()

train_df['week_year_cnt'] = week_year.map(week_year_cnt_map)





# Other feature engineering

train_df['rel_floor'] = train_df['floor'] / train_df['max_floor'].astype(float)

train_df['rel_kitch_sq'] = train_df['kitch_sq'] / train_df['full_sq'].astype(float)
train_df.pop('timestamp').values
train_df1 = pd.DataFrame()

train_df1["full_sq"]=train_df["full_sq"]

train_df1["floor"]=train_df["floor"]

train_df1["life_sq"]=train_df["life_sq"]

train_df1["build_year"]=train_df["build_year"]

train_df1["max_floor"]=train_df["max_floor"]

train_df1["mth_id"] = train_df["mth_id"]

train_df1["kitch_sq"] = train_df["kitch_sq"]

train_df1["num_room"] = train_df["num_room"]

train_df1["material"] = train_df["material"]

train_df1["Age_building"] = train_df["Age_building"]

train_df1
# Deal with categorical values

train_df_numeric = train_df.select_dtypes(exclude=['object'])

train_df_obj =train_df.select_dtypes(include=['object']).copy()



for c in train_df_obj:

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(train_df_obj[c].values)) 

        train_df_obj[c] = lbl.transform(list(train_df_obj[c].values))



train_df_values = pd.concat([train_df_numeric, train_df_obj], axis=1)



X_all = train_df_values.values

print(X_all.shape)



x_train = X_all[:num_train]

x_test = X_all[num_train:]



train_df_columns=train_df_values.columns


xgb_params = {

    'eta': 0.05,

    'max_depth': 8,

    'subsample': 0.7,

    'colsample_bytree': 0.7,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'silent': 1

}

dtrain = xgb.DMatrix(x_train, y_train, feature_names=train_df_columns)

dtest = xgb.DMatrix(x_test, feature_names=train_df_columns)

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100)



# plot the important features #

fig, ax = plt.subplots(figsize=(12,18))

xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)

plt.show()
cv_result = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,

    verbose_eval=True, show_stdv=False)

cv_result[['train-rmse-mean', 'test-rmse-mean']].plot()

num_boost_rounds = len(cv_result)
num_boost_rounds
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)
y_pred = model.predict(dtest)



df_sub = pd.DataFrame({'id': id_test, 'price_doc': y_pred})



df_sub.to_csv('sub.csv', index=False)