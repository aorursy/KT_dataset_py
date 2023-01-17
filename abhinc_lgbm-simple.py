import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import lightgbm as lgb



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import GroupKFold



sns.set_style('darkgrid')
%matplotlib inline
DATA_DIR = '../input/ashrae-energy-prediction'

print(os.listdir(DATA_DIR))
# Function to reduce the DF size

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



def load_df(fname):

    df = pd.read_csv(os.path.join(DATA_DIR, fname))

    if 'timestamp' in df.columns:

        # I guess fortunately all timestamp columns are called `timestamp`.

        df['timestamp'] = pd.to_datetime(df['timestamp'])

    return df



train_df = reduce_mem_usage(load_df('train.csv'))

building_metadata_df = reduce_mem_usage(load_df('building_metadata.csv'))

weather_train_df = reduce_mem_usage(load_df('weather_train.csv'))
def describe_df(df):

    print('Shape of data: ', df.shape)

    print('\nBasic info:')

    print(df.info())

    print('\nQuick peek at the data:')

    print(df.head())

    print('\nBasic description of the data:')

    print(df.describe())

    print('\nLooking at NAs')

    print(df.isna().sum())
describe_df(train_df)
# The duration of the training data

train_df['timestamp'].min(), train_df['timestamp'].max()
sns.relplot(x='timestamp', y='meter_reading', hue='meter', kind='line', 

            palette=sns.color_palette('hls', 4), aspect=16/9,

            data=(train_df[train_df['timestamp'] > pd.to_datetime('2016-12-01')]

                  .groupby(by=['meter', 'timestamp'])

                  .agg({'meter_reading': 'median'}).reset_index()))

plt.xticks(rotation=15)

plt.title('Median meter readings over time for different meter types (1 month)')

plt.gca().set(yscale='log')

plt.ylabel('Meter reading (log-scale)')

plt.show()
sns.relplot(x='month_of_year', y='meter_reading', hue='meter', kind='line', 

            palette=sns.color_palette('hls', 4), aspect=16/9,

            data=(train_df

                  .assign(month_of_year=train_df['timestamp'].dt.month)

                  .groupby(by=['meter', 'month_of_year'])

                  .agg({'meter_reading': 'median'}).reset_index()))

plt.xticks(rotation=15)

plt.title('Median meter readings over months for different meter types (1 year)')

plt.show()

plt.close()
train_df = train_df.assign(hour_of_day=train_df['timestamp'].dt.hour, 

                           day_of_week=train_df['timestamp'].dt.dayofweek,

                           day_of_year=train_df['timestamp'].dt.dayofyear)
describe_df(building_metadata_df)
g = sns.jointplot(x='square_feet', y='meter_reading', height=8,

                  data=(train_df.groupby(by='building_id')

                        .agg({'meter_reading': 'median'})

                        .join(building_metadata_df, on=['building_id'])))

plt.show()

plt.close()
g = sns.jointplot(x='square_feet', y='meter_reading', height=8, kind='reg',

                  data=(train_df.groupby(by='building_id')

                        .agg({'meter_reading': 'median'})

                        .join(building_metadata_df, on=['building_id'])

                        .pipe(lambda df: df.assign(meter_reading=np.log1p(df.meter_reading),

                                                   square_feet=np.log1p(df.square_feet)))))

g.ax_joint.set_xlabel('Meter reading (log-scale)')

g.ax_joint.set_ylabel('Square feet (log-scale)')

g.fig.suptitle('Relationship between Square footage and energy consumption in log-log space')

plt.show()

plt.close()
fig = plt.figure(figsize=(10, 20))

sns.violinplot(x='meter_reading', y='primary_use', orient='h', scale='count',

               data=(train_df.groupby(by='building_id')

                     .agg({'meter_reading': 'median'})

                     .join(building_metadata_df, on=['building_id'])))

plt.show()

plt.close()
plt.figure(figsize=(10, 5))

sns.lineplot(x='year_built', y='meter_reading',

             data=(train_df

                   .merge(building_metadata_df, on=['building_id'])

                   .groupby(by='year_built')

                   .agg({'meter_reading': 'median'})

                   .reset_index()))

plt.title('Meter readings for buildings built in different years')

plt.show()

plt.close()
building_metadata_enc = {

    'year_built': LabelEncoder(),

    'primary_use': LabelEncoder(),

}

building_metadata_df['year_built_enc'] = (building_metadata_enc['year_built']

                                          .fit_transform(building_metadata_df['year_built']))

building_metadata_df['primary_use_enc'] = (building_metadata_enc['primary_use']

                                           .fit_transform(building_metadata_df['primary_use']))
building_metadata_df['square_feet_log'] = np.log1p(building_metadata_df['square_feet'])
describe_df(weather_train_df)
merged_df = (train_df

             .merge(building_metadata_df, on=['building_id'])

             .merge(weather_train_df, on=['site_id', 'timestamp']))
merged_df.head()
merged_df['meter_reading_log'] = np.log1p(merged_df['meter_reading'])
print(sorted(merged_df.columns))
merged_df = merged_df.sort_values('timestamp')

feature_cols = ['building_id', 'day_of_week', 'day_of_year', 'floor_count', 

                'hour_of_day', 'meter', 'primary_use_enc', 'site_id', 

                'square_feet_log', 'year_built_enc']

categorical_features = ['building_id', 'day_of_week', 'day_of_year', 'hour_of_day',

                        'meter', 'primary_use_enc', 'site_id', 'year_built_enc']

X_df = merged_df[feature_cols]

y_df = merged_df[['meter_reading_log']]
# We can use `sklearn.model_selection.GroupKFold` and define "groups" to be based on

# the month. Our training dataset has 12 months, and we can choose 3 splits.

kfold = GroupKFold(n_splits=3)

groups = merged_df['timestamp'].dt.month
import sys



# These are the usual ipython objects, including this one you are creating

ipython_vars = ['In', 'Out', 'exit', 'quit', 'get_ipython', 'ipython_vars']



# Get a sorted list of the objects and their sizes

sorted([(x, sys.getsizeof(globals().get(x))) for x in dir() if not x.startswith('_') and x not in sys.modules and x not in ipython_vars], key=lambda x: x[1], reverse=True)
# Get rid of other dataframes and save some memory before the real show begins

del train_df

del weather_train_df

del merged_df



import gc

gc.collect()
import sys



# These are the usual ipython objects, including this one you are creating

ipython_vars = ['In', 'Out', 'exit', 'quit', 'get_ipython', 'ipython_vars']



# Get a sorted list of the objects and their sizes

sorted([(x, sys.getsizeof(globals().get(x))) for x in dir() if not x.startswith('_') and x not in sys.modules and x not in ipython_vars], key=lambda x: x[1], reverse=True)
models = []

params = {

    "objective": "regression",

    "boosting": "gbdt",

    "num_leaves": 1280,

    "learning_rate": 0.05,

    "feature_fraction": 0.85,

    "reg_lambda": 2,

    # The actual metric we'd be measured against is RMLSE 

    # (https://www.kaggle.com/c/ashrae-energy-prediction/overview/evaluation)

    # but since we've already taken the log of the meter reading as the target

    # we should just use the RMSE metric.

    "metric": "rmse",

}

for idx, (train_index, val_index) in enumerate(kfold.split(X_df, y_df, groups)):

    print(f'Training fold {idx}')

    X_train_df, y_train_df = X_df.loc[train_index], y_df.loc[train_index]

    X_val_df, y_val_df = X_df.loc[val_index], y_df.loc[val_index]

    train_dataset = lgb.Dataset(X_train_df, label=y_train_df, 

                                categorical_feature=categorical_features)

    val_dataset = lgb.Dataset(X_val_df, label=y_val_df,

                              categorical_feature=categorical_features)

    model = lgb.train(params=params, train_set=train_dataset, num_boost_round=1000,

                      valid_sets=[val_dataset],

                      early_stopping_rounds=50, verbose_eval=25)

    models.append(model)
for model in models:

    lgb.plot_importance(model)

    plt.show()
print(np.mean([model.best_score['valid_0']['rmse'] for model in models]))
# Load the test dataframes first

test_df = reduce_mem_usage(load_df('test.csv'))

weather_test_df = reduce_mem_usage(load_df('weather_test.csv'))
gc.collect()
test_df = test_df.assign(hour_of_day=test_df['timestamp'].dt.hour, 

                         day_of_week=test_df['timestamp'].dt.dayofweek,

                         day_of_year=test_df['timestamp'].dt.dayofyear)



merged_test_df = (test_df

                  .merge(building_metadata_df, how='left', on=['building_id'])

                  .merge(weather_test_df, how='left', on=['site_id', 'timestamp']))

X_test_df = merged_test_df[feature_cols]

row_ids = merged_test_df['row_id']
del test_df

del weather_test_df

del merged_test_df

gc.collect()
describe_df(X_test_df)
results = sum(np.expm1(model.predict(X_test_df)) / len(models) 

              for model in models)

results_df = pd.DataFrame({'row_id': row_ids, 

                           'meter_reading': np.clip(results, 0, None)})

results_df.head()
results_df.to_csv('submission.csv', index=False, float_format='%.4f')