import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from tqdm import tqdm_notebook

import seaborn as sns

import gc

import plotly.graph_objects as go



# /kaggle/input/ashrae-energy-prediction/weather_test.csv

# /kaggle/input/ashrae-energy-prediction/building_metadata.csv

# /kaggle/input/ashrae-energy-prediction/train.csv

# /kaggle/input/ashrae-energy-prediction/test.csv

# /kaggle/input/ashrae-energy-prediction/sample_submission.csv

# /kaggle/input/ashrae-energy-prediction/weather_train.csv
# Import datasets For train



train_df = pd.read_csv('/kaggle/input/ashrae-energy-prediction/train.csv')

weather_train_df = pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_train.csv')

building_meta_df = pd.read_csv('/kaggle/input/ashrae-energy-prediction/building_metadata.csv')



# Import datasets For test



# test_df = pd.read_csv('/kaggle/input/ashrae-energy-prediction/test.csv')

# weather_test_df = pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_test.csv')

# sample_submission = pd.read_csv('/kaggle/input/ashrae-energy-prediction/sample_submission.csv')

from pandas.api.types import is_datetime64_any_dtype as is_datetime

from pandas.api.types import is_categorical_dtype



def reduce_mem_usage_2(df, use_float16=False):

    """

    Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.        

    """

    

    start_mem = df.memory_usage().sum() / 1024**2

    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    

    for col in df.columns:

        if col == 'timestamp': continue

        if is_datetime(df[col]) or is_categorical_dtype(df[col]):

            continue

        col_type = df[col].dtype

        

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == "int":

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        else:

            df[col] = df[col].astype("category")



    end_mem = df.memory_usage().sum() / 1024**2

    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))

    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    

    return df
train_df = reduce_mem_usage_2(train_df ,use_float16=True)

weather_train_df = reduce_mem_usage_2(weather_train_df ,use_float16=True)

building_meta_df = reduce_mem_usage_2(building_meta_df ,use_float16=True)
train_df = train_df.merge(building_meta_df, on='building_id', how='left')

train_df = train_df.merge(weather_train_df, on=['site_id', 'timestamp'], how='left')

del weather_train_df, building_meta_df

gc.collect();
train_df.timestamp = pd.to_datetime(train_df.timestamp)
train_df['date'] = train_df.timestamp.dt.date

train_df ['hour'] = train_df.timestamp.dt.hour
# just first Let's see 5 first rows. this is my habit.

print(f'shape is {train_df.shape}')

train_df.head()
# check first hypothesis :

corr = train_df.corr()

# Don;t show upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11, 9))

sns.heatmap(corr, mask=mask)
train_df.isnull().sum() * 100 / train_df.shape[0]
fig, axes = plt.subplots(figsize=(14,6))

train_df[['timestamp', 'meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(ax=axes, label='By hour', alpha=0.8).set_ylabel('Meter reading', fontsize=14)
ax = sns.barplot(train_df.groupby(['primary_use']).size().reset_index(name='counts')['primary_use'], train_df.groupby(['primary_use']).size().reset_index(name='counts')['counts'])

ax.set(xlabel='Primary Usage', ylabel='# of records', title='Primary Usage vs. # of records')

ax.set_xticklabels(ax.get_xticklabels(), rotation=50, ha="right")

plt.show()
ax = sns.barplot(train_df.groupby('primary_use').meter_reading.agg('sum').reset_index()['primary_use'],

                train_df.groupby('primary_use').meter_reading.agg('sum').reset_index()['meter_reading'])

ax.set(xlabel='Primary Usage', ylabel='# of records', title='Primary Usage vs. # of records')

ax.set_xticklabels(ax.get_xticklabels(), rotation=50, ha="right")

plt.show()
train_df.groupby('primary_use').meter.agg(pd.Series.mode).reset_index()
ax = sns.barplot(['electricity', 'chiledwater', 'steam', 'hotwater'], train_df.meter.value_counts().to_frame()['meter'])

ax.set(xlabel='meter Types', ylabel='How much is used', title='meter Types vs. # of usage')
fig, axes = plt.subplots(nrows=4, ncols=4)

fig.set_figheight(12)

fig.set_figwidth(15)

# fig.tight_layout()

fig.subplots_adjust(wspace=0.5)

fig.subplots_adjust(hspace=0.5)

primary_uses = train_df.primary_use.unique()



for i, primary_use in enumerate(primary_uses):

    sns.countplot(x="meter", data = train_df[train_df.primary_use  == primary_use],ax=axes[i//4,i%4]).title.set_text(str(primary_use))



meter_types = { 0 : 'electricity', 1 : 'chilledwater', 2 : 'steam', 3 : 'hotwater' }





fig, axes = plt.subplots(nrows=4, ncols=4)

fig.set_figheight(12)

fig.set_figwidth(15)

# fig.tight_layout()

fig.subplots_adjust(wspace=0.5)

fig.subplots_adjust(hspace=0.5)

primary_uses = train_df.primary_use.unique()



for i, primary_use in enumerate(primary_uses):

    sns.barplot(train_df[train_df.primary_use == primary_use].groupby('meter').meter_reading.sum(),

               np.vectorize(meter_types.get)(train_df[train_df.primary_use == primary_use].groupby(['meter'])['meter_reading'].mean().keys()),

                ax=axes[i//4,i%4]).title.set_text(str(primary_use))

train_df.groupby('primary_use').square_feet.agg(['mean'])
train_df.groupby('primary_use').year_built.agg(['mean'])
train_df.groupby('primary_use').meter_reading.agg(['mean'])
fig, axes = plt.subplots(8,2,figsize=(14, 30), dpi=100)



for i in range(train_df['site_id'].nunique()):

    train_df[train_df.site_id == i][['timestamp', 'meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(ax = axes[i%8][i//8])

    axes[i%8][i//8].legend();

    axes[i%8][i//8].set_title('site_id {}'.format(i), fontsize=13);

    plt.subplots_adjust(hspace=0.45)
fig, axes = plt.subplots(8,2,figsize=(14, 30), dpi=100)

for i, use in enumerate(train_df['primary_use'].value_counts().index.to_list()):

    try:

        train_df[(train_df['site_id'] == 13) & (train_df['primary_use'] == use)][['timestamp', 'meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(ax=axes[i%8][i//8], alpha=0.8, label='By hour', color='tab:blue').set_ylabel('Mean meter reading', fontsize=13);

        train_df[(train_df['site_id'] == 13) & (train_df['primary_use'] == use)][['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(ax=axes[i%8][i//8], alpha=1, label='By day', color='tab:orange').set_xlabel('');

        axes[i%8][i//8].legend();

    except TypeError:

        pass

    axes[i%8][i//8].set_title(use, fontsize=13);

    plt.subplots_adjust(hspace=0.45)
train_df[(train_df['site_id'] == 13) & (train_df['primary_use'] == 'Education')][['timestamp', 'meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot()
meter_types = { 0 : 'electricity', 1 : 'chilledwater', 2 : 'steam', 3 : 'hotwater' }

country=train_df[(train_df.site_id == 13) & (train_df.primary_use == 'Education') ].groupby('meter')['meter_reading'].sum()



fig = go.Figure(go.Treemap(

    labels=np.vectorize(meter_types.get)(country.keys()),

    parents = ["Total Energy Usage Type "]*len(country),

    values =  country

))



fig.show()


fig, axes = plt.subplots(9,2,figsize=(14, 36), dpi=100)

for i, building in enumerate(train_df[(train_df['site_id'] == 13) & (train_df['primary_use'] == 'Education') & (train_df['meter'] == 2)]['building_id'].value_counts(dropna=False).index.to_list()):

    train_df[(train_df['site_id'] == 13) & (train_df['primary_use'] == 'Education') & (train_df['meter'] == 2) & (train_df['building_id'] == building)][['timestamp', 'meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(ax=axes[i%9][i//9], alpha=0.8, label='By hour', color='tab:blue').set_ylabel('Mean meter reading', fontsize=13);

    train_df[(train_df['site_id'] == 13) & (train_df['primary_use'] == 'Education') & (train_df['meter'] == 2) & (train_df['building_id'] == building)][['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(ax=axes[i%9][i//9], alpha=1, label='By day', color='tab:orange').set_xlabel('');

    axes[i%9][i//9].legend();

    axes[i%9][i//9].set_title('building_id: ' + str(building), fontsize=13);

    plt.subplots_adjust(hspace=0.45)