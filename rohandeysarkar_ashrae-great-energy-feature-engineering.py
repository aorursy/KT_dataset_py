import gc

import os

import numpy as np 

import pandas as pd  

import warnings

warnings.filterwarnings('ignore')

import gc



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import matplotlib.patches as patches



from plotly import tools, subplots

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.express as px

pd.set_option('max_columns', 100)

py.init_notebook_mode(connected=True)

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go



from statsmodels.tsa.seasonal import seasonal_decompose

import statsmodels.api as sm



from sklearn.preprocessing import LabelEncoder

from sklearn import metrics

from sklearn.metrics import mean_squared_error

import lightgbm as lgb

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

import os, random, math, psutil, pickle

from tqdm import tqdm

import lightgbm as lgb

from sklearn.model_selection import KFold

import matplotlib.pyplot as plt

import seaborn as sns

from datetime import datetime
root = '../input/ashrae-energy-prediction/'



train_df = pd.read_csv(root + 'train.csv')

train_df["timestamp"] = pd.to_datetime(train_df["timestamp"], format='%Y-%m-%d %H:%M:%S')



weather_train_df = pd.read_csv(root + 'weather_train.csv')

test_df = pd.read_csv(root + 'test.csv')

weather_test_df = pd.read_csv(root + 'weather_test.csv')

building_meta_df = pd.read_csv(root + 'building_metadata.csv')

sample_submission = pd.read_csv(root + 'sample_submission.csv')
train_df.head()
weather_train_df.head()
building_meta_df
## Function to reduce the DF size

def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
## Reducing memory

train_df = reduce_mem_usage(train_df)

test_df = reduce_mem_usage(test_df)



weather_train_df = reduce_mem_usage(weather_train_df)

weather_test_df = reduce_mem_usage(weather_test_df)

building_meta_df = reduce_mem_usage(building_meta_df)



train_df['meter'] = pd.Categorical(train_df['meter']).rename_categories({0: 'electricity', 1: 'chilledwater', 2: 'steam', 3: 'hotwater'})



daily_train = train_df.copy()

daily_train['date'] = daily_train['timestamp'].dt.date

daily_train = daily_train.groupby(['date', 'building_id', 'meter']).sum()



daily_train_agg = daily_train.groupby(['date', 'meter']).agg(['sum', 'mean', 'idxmax', 'max'])

daily_train_agg = daily_train_agg.reset_index()



level_0 = daily_train_agg.columns.droplevel(0)

level_1 = daily_train_agg.columns.droplevel(1)

level_0 = ['' if x == '' else '-' + x for x in level_0]

daily_train_agg.columns = level_1 + level_0

daily_train_agg.rename_axis(None, axis=1)



daily_train_agg['building_id_max'] = [x[1] for x in daily_train_agg['meter_reading-idxmax']]

# daily_train_agg.head()

# daily_train.head()

# max consuming buildings

daily_train_electricity = daily_train_agg[daily_train_agg['meter'] == 'electricity'].copy()

daily_train_electricity['building_id_max'] = pd.Categorical(daily_train_electricity['building_id_max']) # convert this col to categorical



daily_train_chilledwater = daily_train_agg[daily_train_agg['meter'] == 'chilledwater'].copy()

daily_train_chilledwater['building_id_max'] = pd.Categorical(daily_train_chilledwater['building_id_max'])



daily_train_steam = daily_train_agg[daily_train_agg['meter']=='steam'].copy()

daily_train_steam['building_id_max'] = pd.Categorical(daily_train_steam['building_id_max'])



daily_train_hotwater = daily_train_agg[daily_train_agg['meter']=='hotwater'].copy()

daily_train_hotwater['building_id_max'] = pd.Categorical(daily_train_hotwater['building_id_max'])
# Feature engineering

train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])

test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])

weather_train_df['timestamp'] = pd.to_datetime(weather_train_df['timestamp'])

weather_test_df['timestamp'] = pd.to_datetime(weather_test_df['timestamp'])

    

building_meta_df['primary_use'] = building_meta_df['primary_use'].astype('category')



temp_df = train_df[['building_id']]

temp_df = temp_df.merge(building_meta_df, on=['building_id'])



del temp_df['building_id']

train_df = pd.concat([train_df, temp_df], axis=1)



temp_df = test_df[['building_id']]

temp_df = temp_df.merge(building_meta_df, on=['building_id'], how='left')



del temp_df['building_id']

test_df = pd.concat([test_df, temp_df], axis=1)



del temp_df, building_meta_df



temp_df = train_df[['site_id', 'timestamp']]

temp_df = temp_df.merge(weather_train_df, on=['site_id', 'timestamp'], how='left')



del temp_df['site_id'], temp_df['timestamp']

train_df = pd.concat([train_df, temp_df], axis=1)



temp_df = test_df[['site_id','timestamp']]

temp_df = temp_df.merge(weather_test_df, on=['site_id','timestamp'], how='left')



del temp_df['site_id'], temp_df['timestamp']

test_df = pd.concat([test_df, temp_df], axis=1)



del temp_df, weather_train_df, weather_test_df

le = LabelEncoder()

train_df['primary_use'] = le.fit_transform(train_df['primary_use']).astype(np.int8)

test_df['primary_use'] = le.fit_transform(test_df['primary_use']).astype(np.int8)
# missing data handling

train_df['age'] = train_df['year_built'].max() - train_df['year_built'] + 1

test_df['age'] = test_df['year_built'].max() - test_df['year_built'] + 1



train_df['floor_count'] = train_df['floor_count'].fillna(-999).astype(np.int16)

test_df['floor_count'] = test_df['floor_count'].fillna(-999).astype(np.int16)



train_df['year_built'] = train_df['year_built'].fillna(-999).astype(np.int16)

test_df['year_built'] = test_df['year_built'].fillna(-999).astype(np.int16)



train_df['age'] = train_df['age'].fillna(-999).astype(np.int16)

test_df['age'] = test_df['age'].fillna(-999).astype(np.int16)



train_df['cloud_coverage'] = train_df['cloud_coverage'].fillna(-999).astype(np.int16)

test_df['cloud_coverage'] = test_df['cloud_coverage'].fillna(-999).astype(np.int16) 
# some datetime features

train_df['month_datetime'] = train_df['timestamp'].dt.month.astype(np.int8)

train_df['weekofyear_datetime'] = train_df['timestamp'].dt.weekofyear.astype(np.int8)

train_df['dayofyear_datetime'] = train_df['timestamp'].dt.dayofyear.astype(np.int16)

    

train_df['hour_datetime'] = train_df['timestamp'].dt.hour.astype(np.int8)  

train_df['day_week'] = train_df['timestamp'].dt.dayofweek.astype(np.int8)

train_df['day_month_datetime'] = train_df['timestamp'].dt.day.astype(np.int8)

train_df['week_month_datetime'] = train_df['timestamp'].dt.day/7

train_df['week_month_datetime'] = train_df['week_month_datetime'].apply(lambda x: math.ceil(x)).astype(np.int8)

    

train_df['year_built'] = train_df['year_built']-1900

train_df['square_feet'] = np.log(train_df['square_feet'])

    

test_df['month_datetime'] = test_df['timestamp'].dt.month.astype(np.int8)

test_df['weekofyear_datetime'] = test_df['timestamp'].dt.weekofyear.astype(np.int8)

test_df['dayofyear_datetime'] = test_df['timestamp'].dt.dayofyear.astype(np.int16)

    

test_df['hour_datetime'] = test_df['timestamp'].dt.hour.astype(np.int8)

test_df['day_week'] = test_df['timestamp'].dt.dayofweek.astype(np.int8)

test_df['day_month_datetime'] = test_df['timestamp'].dt.day.astype(np.int8)

test_df['week_month_datetime'] = test_df['timestamp'].dt.day/7

test_df['week_month_datetime'] = test_df['week_month_datetime'].apply(lambda x: math.ceil(x)).astype(np.int8)

    

test_df['year_built'] = test_df['year_built']-1900

test_df['square_feet'] = np.log(test_df['square_feet'])
# drop cols

drop_cols = ["precip_depth_1_hr", "sea_level_pressure", "wind_direction", "wind_speed","timestamp"]

target = np.log1p(train_df["meter_reading"]) 



# del train["meter_reading"]

train_df = train_df.drop(drop_cols, axis=1)
# lightbgm

params = {

    'boosting_type': 'gbdt',

    'objective': 'regression',

    'metric': {'rmse'},

    'subsample': 0.2,

    'learning_rate': 0.9,

    'feature_fraction': 0.9,

    'bagging_fraction': 0.9,

    'alpha': 0.1,

    'lambda': 0.1

}



folds = 3

seed = 41 #666

kf = KFold(n_splits=folds, shuffle=True, random_state=seed)



models = []

for train_index, val_index in kf.split(train_df):

    train_X = train_df.iloc[train_index]

    val_X = train_df.iloc[val_index]

    train_y = target.iloc[train_index]

    val_y = target.iloc[val_index]

    lgb_train = lgb.Dataset(train_X, train_y)

    lgb_eval = lgb.Dataset(val_X, val_y)

    gbm = lgb.train(params,

                    lgb_train,

                    num_boost_round=10, #300,

                    valid_sets=(lgb_train, lgb_eval),

                    early_stopping_rounds= 10,#100,

                    verbose_eval=10) #100)

    models.append(gbm)

# del train, train_X, val_X, lgb_train, lgb_eval, train_y, val_y, target

gc.collect()
feature_imp = pd.DataFrame(sorted(zip(gbm.feature_importance(), gbm.feature_name()),reverse = True), columns=['Value','Feature'])

plt.figure(figsize=(10, 5))

sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))

plt.title('LightGBM FEATURES')

plt.tight_layout()

plt.show()
i = 0

res = []

step_size = 50000

for j in tqdm(range(int(np.ceil(test.shape[0] / 50000)))):

    res.append(np.expm1(sum([model.predict(test.iloc[i:i + step_size]) for model in models]) / folds))

    i += step_size
res = np.concatenate(res)

sample_submission["meter_reading"] = res

sample_submission.loc[sample_submission['meter_reading'] < 0, 'meter_reading'] = 0

sample_submission.to_csv('sub_' + str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + '.csv', index=False)

sample_submission.head(10)