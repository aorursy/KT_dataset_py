import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import datetime

import warnings

warnings.filterwarnings('ignore')

import lightgbm as lgb

import gc
building = pd.read_csv('../input/ashrae-energy-prediction/building_metadata.csv')

weather_train = pd.read_csv('../input/ashrae-energy-prediction/weather_train.csv')

train = pd.read_csv('../input/ashrae-energy-prediction/train.csv')
print('Size of building data', building.shape)

print('Size of weather_train data', weather_train.shape)

print('Size of train data', train.shape)
building.head()
weather_train.head()
train.head()
labels = building['primary_use'].unique()

count = sns.countplot(data = building,x = 'primary_use')

count.set(title = 'Number of Buildings primary use wise')

count.set_xticklabels(labels,rotation = 90)

count
merge1 = building.merge(train, left_on = 'building_id', right_on = 'building_id')

X = merge1.merge(weather_train, left_on = ['site_id','timestamp'], right_on = ['site_id','timestamp'])

y = np.log1p(X.meter_reading)

X.drop("timestamp", axis = 1, inplace = True)

X.drop("site_id", axis = 1, inplace = True)

X.drop("building_id", axis = 1, inplace = True)

X.drop("meter_reading", axis = 1, inplace = True)

X.drop("primary_use", axis = 1, inplace = True)

X_half_1 = X[:int(X.shape[0] / 2)]

X_half_2 = X[int(X.shape[0] / 2):]

y_half_1 = y[:int(X.shape[0] / 2)]

y_half_2 = y[int(X.shape[0] / 2):]

d_half_1 = lgb.Dataset(X_half_1, label = y_half_1,  free_raw_data = False)

d_half_2 = lgb.Dataset(X_half_2, label = y_half_2,  free_raw_data = False)

watchlist_1 = [d_half_1, d_half_2]

watchlist_2 = [d_half_2, d_half_1]

params = {

    "objective": "regression",

    "boosting": "gbdt",

    "num_leaves": 40,

    "learning_rate": 0.05,

    "feature_fraction": 0.85,

    "reg_lambda": 2,

    "metric": "rmse"

}

print("Building model with first half and validating on second half:")

model_half_1 = lgb.train(params, train_set = d_half_1, num_boost_round = 1000, valid_sets = watchlist_1, verbose_eval = 200, early_stopping_rounds = 200)

print("Building model with second half and validating on first half:")

model_half_2 = lgb.train(params, train_set = d_half_2, num_boost_round = 1000, valid_sets = watchlist_2, verbose_eval = 200, early_stopping_rounds = 200)
df_fimp_1 = pd.DataFrame()

df_fimp_1["feature"] = X.columns.values

df_fimp_1["importance"] = model_half_1.feature_importance()

df_fimp_1["half"] = 1

df_fimp_2 = pd.DataFrame()

df_fimp_2["feature"] = X.columns.values

df_fimp_2["importance"] = model_half_2.feature_importance()

df_fimp_2["half"] = 2

df_fimp = pd.concat([df_fimp_1, df_fimp_2], axis=0)

plt.figure(figsize=(14, 7))

sns.barplot(x = "importance", y = "feature", data = df_fimp.sort_values(by = "importance", ascending = False))

plt.title("LightGBM Feature Importance")

plt.tight_layout()
train['timestamp'] = pd.to_datetime(train.timestamp)

train = train.set_index(['timestamp'])

f,a = plt.subplots(1,4,figsize = (20,30))

for meter in np.arange(4):

    df = train[train.meter==meter].copy().reset_index()

    df['timestamp'] = pd.to_timedelta(df.timestamp).dt.total_seconds() / 3600

    df['timestamp'] = df.timestamp.astype(int)

    df.timestamp -= df.timestamp.min()

    missmap = np.empty((1449, df.timestamp.max() + 1))

    missmap.fill(np.nan)

    for l in df.values:

        if l[2] != meter:continue

        missmap[int(l[1]), int(l[0])] = 0 if l[3] == 0 else 1

    a[meter].set_title(f'meter {meter:d}')

    sns.heatmap(missmap, cmap = 'Paired', ax = a[meter], cbar = False)
time_format = "%Y-%m-%d %H:%M:%S"

start_date = datetime.datetime.strptime(weather_train['timestamp'].min(),time_format)

end_date = datetime.datetime.strptime(weather_train['timestamp'].max(),time_format)

total_hours = int(((end_date - start_date).total_seconds() + 3600) / 3600)

hours_list = [(end_date - datetime.timedelta(hours = x)).strftime(time_format) for x in range(total_hours)]

missing_hours = []

for site_id in range(16):

    site_hours = np.array(weather_train[weather_train['site_id'] == site_id]['timestamp'])

    new_rows = pd.DataFrame(np.setdiff1d(hours_list,site_hours),columns = ['timestamp'])

    new_rows['site_id'] = site_id

    weather_train = pd.concat([weather_train,new_rows])

weather_train = weather_train.reset_index(drop = True)    
def missing_statistics(df):    

    statistics = pd.DataFrame(df.isnull().sum()).reset_index()

    statistics.columns=['COLUMN NAME',"MISSING VALUES"]

    statistics['TOTAL ROWS'] = df.shape[0]

    statistics['% MISSING'] = round((statistics['MISSING VALUES']/statistics['TOTAL ROWS']) * 100,2)

    return statistics

missing_statistics_weather_train = missing_statistics(weather_train)

missing_statistics_weather_train.to_excel("missing_statistics_weather_train.xls")
missing_statistics_building = missing_statistics(building)

missing_statistics_building.to_excel("missing_statistics_building.xls")
missing_statistics(building)