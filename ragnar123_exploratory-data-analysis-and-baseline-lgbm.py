# Libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import lightgbm as lgb

import warnings

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 1000)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
print('Reading train set...')

train = pd.read_csv('/kaggle/input/ashrae-energy-prediction/train.csv')

print('Reading test set...')

test = pd.read_csv('/kaggle/input/ashrae-energy-prediction/test.csv')

print('Train set has {} rows and {} columns'.format(train.shape[0], train.shape[1]))

print('Test set has {} rows and {} columns'.format(test.shape[0], test.shape[1]))
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
train = reduce_mem_usage(train)

test = reduce_mem_usage(test)
train.isnull().sum()
test.isnull().sum()
train.head()
print('We have {} time series in our training set'.format((train['building_id'].astype(str) + train['meter'].astype(str)).nunique()))

print('We have {} time series in our test set'.format((test['building_id'].astype(str) + test['meter'].astype(str)).nunique()))
train_series = list((train['building_id'].astype(str) + train['meter'].astype(str)).unique())

test_series = list((test['building_id'].astype(str) + test['meter'].astype(str)).unique())

print('Number of series that are in the training set and are also contained in the test set {}'.format(len([x for x in train_series if x in test_series])))
def plot_count(df, col):

    total = len(df)

    plt.figure(figsize = (12,8))

    plot_me = sns.countplot(df[col])

    plot_me.set_xlabel('{} type'.format(col), fontsize = 16)

    plot_me.set_ylabel('frequency', fontsize = 16)

    for p in plot_me.patches:

        height = p.get_height()

        plot_me.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(height/total*100),

                ha="center", fontsize=15)

        

plot_count(train, 'meter')
plot_count(test, 'meter')
train['timestamp'] = pd.to_datetime(train['timestamp'])

test['timestamp'] = pd.to_datetime(test['timestamp'])

train_el = train[train['meter']==0]

train_ch = train[train['meter']==1]

train_st = train[train['meter']==2]

train_ho = train[train['meter']==3]

test_el = test[test['meter']==0]

test_ch = test[test['meter']==1]

test_st = test[test['meter']==2]

test_ho = test[test['meter']==3]
# count the number of building for each timestamp for electicity meter

def plot_time_freq(df, name = 'electricity', se = 'train'):

    print('We have {} series'.format(df['building_id'].nunique()))

    print('Min date: ', df.timestamp.min())

    print('Max date: ', df.timestamp.max())

    print('Time behaviour for {} meter for the {} set'.format(name, se))

    df['date'] = df['timestamp'].dt.date 

    df['week'] = df['timestamp'].dt.week

    df['dayofmonth'] = df['timestamp'].dt.day

    df['month'] = df['timestamp'].dt.month

    df['dayofweek'] = df['timestamp'].dt.dayofweek

    df['hour'] = df['timestamp'].dt.hour

    tmp1 = df.groupby(['date'])['building_id'].count().reset_index().rename(columns = {'building_id': 'frequency'})

    tmp2 = df.groupby(['week'])['building_id'].count().reset_index().rename(columns = {'building_id': 'frequency'})

    tmp3 = df.groupby(['dayofmonth'])['building_id'].count().reset_index().rename(columns = {'building_id': 'frequency'})

    tmp4 = df.groupby(['hour'])['building_id'].count().reset_index().rename(columns = {'building_id': 'frequency'})

    tmp5 = df.groupby(['month'])['building_id'].count().reset_index().rename(columns = {'building_id': 'frequency'})

    tmp6 = df.groupby(['dayofweek'])['building_id'].count().reset_index().rename(columns = {'building_id': 'frequency'})

    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, figsize = (12, 12))

    sns.lineplot(tmp1['date'], tmp1['frequency'], ax = ax1)

    ax1.set_title('Date Frequency')

    ax1.set_xlabel('Date', fontsize = 10)

    ax1.set_ylabel('Frequency', fontsize = 10)

    sns.lineplot(tmp2['week'], tmp2['frequency'], ax = ax2)

    ax2.set_title('Week Frequency')

    ax2.set_xlabel('Week', fontsize = 10)

    ax2.set_ylabel('Frequency', fontsize = 10)

    sns.lineplot(tmp3['dayofmonth'], tmp3['frequency'], ax = ax3)

    ax3.set_title('Day of month frequency')

    ax3.set_xlabel('Day of month', fontsize = 10)

    ax3.set_ylabel('Frequency', fontsize = 10)

    sns.lineplot(tmp4['hour'], tmp4['frequency'], ax = ax4)

    ax4.set_title('Hour frequency')

    ax4.set_xlabel('Hour', fontsize = 10)

    ax4.set_ylabel('Frequency', fontsize = 10)

    sns.lineplot(tmp5['month'], tmp5['frequency'], ax = ax5)

    ax5.set_title('Month frequency')

    ax5.set_xlabel('Month', fontsize = 10)

    ax5.set_ylabel('Frequency', fontsize = 10)

    sns.lineplot(tmp6['dayofweek'], tmp6['frequency'], ax = ax6)

    ax6.set_title('Day of week frequency')

    ax6.set_xlabel('Day of week', fontsize = 10)

    ax6.set_ylabel('Frequency', fontsize = 10)

    plt.tight_layout()

    plt.show()



plot_time_freq(train_el, 'electricity', 'train')
plot_time_freq(test_el, 'electricity', 'test')
# let's check a month, in this case Frebuary and March

def build_sw(df, cols, p_cols, value):

    sw = df.groupby(cols)['meter'].count().reset_index()

    sw1 = sw[sw[p_cols]==value]

    plt.figure(figsize = (10,8))

    plt.scatter(sw1[cols[2]], sw1[cols[0]])

    plt.title('Observation for each serie for {} {}'.format(p_cols, value))

    plt.show()

build_sw(train_el, ['building_id', 'month', 'dayofmonth'], 'month', 2)
build_sw(train_el, ['building_id', 'month', 'dayofmonth'], 'month', 3)
def check_hour(df):

    tmp = df.groupby(['building_id', 'date'])['meter'].count().reset_index()

    return tmp[tmp['meter']!=24].iloc[::10].head(10)

check_hour(train_el)
check_hour(test_el)
def start_date(df):

    b_id = []

    min_date = []

    for i in list(df['building_id'].unique()):

        b_id.append(i)

        min_date.append(df[df['building_id']==i]['date'].min())

    tmp = pd.DataFrame({'building_id': b_id, 'min_date': min_date})

    tmp['min_date'] = tmp['min_date'].astype(str)

    print('There are {} series that start after 2016-01-01'.format(tmp[tmp['min_date']!='2016-01-01'].shape[0]))

start_date(train_el)
def end_date(df):

    b_id = []

    max_date = []

    for i in list(df['building_id'].unique()):

        b_id.append(i)

        max_date.append(df[df['building_id']==i]['date'].max())

    tmp = pd.DataFrame({'building_id': b_id, 'max_date': max_date})

    tmp['max_date'] = tmp['max_date'].astype(str)

    print('There are {} series that finish before 2018-12-31'.format(tmp[tmp['max_date']!='2018-12-31'].shape[0]))

end_date(test_el)
def check_series(df, n_years = 1):

    n_day_month = {1 : 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9:30, 10:31, 11:30, 12:31}

    df1 = df.groupby('month')['meter'].count().reset_index()

    df1['n_days'] = df1['month'].map(n_day_month)

    df1['meter_'] = df1['n_days'] * df.building_id.nunique() * n_years * 24

    df1['missing_observations_%'] = 100 - (df1['meter'] / df1['meter_']) * 100

    df1['missing_observations_%'] = df1['missing_observations_%'].astype(str) + '%'

    return df1



check_series(train_el, 1)
check_series(test_el, 2)
plot_time_freq(train_ch, 'chilled water', 'train')
plot_time_freq(test_ch, 'chilled water', 'test')
build_sw(train_ch, ['building_id', 'month', 'dayofmonth'], 'month', 2)
build_sw(train_ch, ['building_id', 'month', 'dayofmonth'], 'month', 3)
check_hour(train_ch)
check_hour(test_ch)
start_date(train_ch)
check_series(train_ch, 1)
check_series(test_ch, 2)
plot_time_freq(train_st, 'steam', 'train')
plot_time_freq(test_st, 'steam', 'test')
check_series(train_st, 1)
check_series(test_st, 2)
plot_time_freq(train_ho, 'hot water', 'train')
plot_time_freq(test_ho, 'hot water', 'test')
check_series(train_ho, 1)
check_series(test_ho, 2)
cross_series = train.groupby(['building_id'])['meter'].nunique().reset_index()

cross_series.columns = ['building_id', 'n_meter']

print('{} series are in the 4 types of meters'.format(cross_series[cross_series['n_meter']==4].shape[0]))

print('{} series are in the 3 types of meters'.format(cross_series[cross_series['n_meter']==3].shape[0]))

print('{} series are in the 2 types of meters'.format(cross_series[cross_series['n_meter']==2].shape[0]))

print('{} series are only in 1 meter'.format(cross_series[cross_series['n_meter']==1].shape[0]))
fig, ax = plt.subplots(2, 2, figsize = (12, 12))

sns.distplot(np.log1p(train_el['meter_reading']), ax = ax[0,0])

ax[0,0].set_title('Distribution for electricity meter')     

sns.distplot(np.log1p(train_ch['meter_reading']), ax = ax[0,1])

ax[0,1].set_title('Distribution for chilled water meter') 

sns.distplot(np.log1p(train_st['meter_reading']), ax = ax[1,0])

ax[1,0].set_title('Distribution for steam meter') 

sns.distplot(np.log1p(train_ho['meter_reading']), ax = ax[1,1])

ax[1,1].set_title('Distribution for hot water meter')
def plot_series(df, building_id, meter_type):

    plt.figure(figsize = (8, 8))

    df1 = df[df['building_id']==building_id]

    df1.groupby(['date'])['meter_reading'].sum().reset_index()

    sns.lineplot(df1['date'], df1['meter_reading'])

    plt.xlabel('Date', fontsize = 10)

    plt.ylabel('Meter reading (sum of the day)')

    plt.suptitle('Meter reading for building_id {} for {} meter type'.format(building_id, meter_type))

    plt.show()

plot_series(train_el, 0, 'electricity')
plot_series(train_el, 1, 'electricity')
plot_series(train_el, 1400, 'electricity')
plot_series(train_el, 800, 'electricity')
plot_series(train_el, 300, 'electricity')
plot_series(train_ch, 1000, 'chilled water')
plot_series(train_ch, 1350, 'chilled water')
def plot_time_variables(df1, df2, df3, df4, col, meter_type):

    df1 = df1.groupby([col])['meter_reading'].sum().reset_index()

    df2 = df2.groupby([col])['meter_reading'].sum().reset_index()

    df3 = df3.groupby([col])['meter_reading'].sum().reset_index()

    df4 = df4.groupby([col])['meter_reading'].sum().reset_index()

    fig, ax = plt.subplots(2, 2, figsize = (12, 12))

    sns.lineplot(df1[col], df1['meter_reading'], ax = ax[0,0])

    sns.lineplot(df2[col], df2['meter_reading'], ax = ax[0,1])

    sns.lineplot(df3[col], df3['meter_reading'], ax = ax[1,0])

    sns.lineplot(df4[col], df4['meter_reading'], ax = ax[1,1])

     

plot_time_variables(train_el, train_ch, train_st, train_ho, 'hour', 'electricity')
plot_time_variables(train_el, train_ch, train_st, train_ho, 'week', 'electricity')
plot_time_variables(train_el, train_ch, train_st, train_ho, 'dayofweek', 'electricity')
bm = pd.read_csv('/kaggle/input/ashrae-energy-prediction/building_metadata.csv')

bm = reduce_mem_usage(bm)

bm.head()
# check if we have all the building metadata

len(list(set(train['building_id'].unique()).intersection(set(bm['building_id'].unique()))))
# check for missing values

def missing_values(df):

    df1 = pd.DataFrame(bm.isnull().sum()).reset_index()

    df1.columns = ['feature', 'n_missing_values']

    df1['ratio'] = df1['n_missing_values'] / df.shape[0]

    df1['unique'] = df.nunique().values

    df1['max'] = df.max().values

    df1['min'] = df.min().values

    return df1

missing_values(bm)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 5))

sns.distplot(bm['square_feet'], ax = ax1)

ax1.set_title('Square feet distribution')

sns.distplot(bm['year_built'].dropna(), ax = ax2)

ax2.set_title('Year built distribution')
train_el = train_el.merge(bm, on = 'building_id')

train_el.head()
# is there a relation between square feet and floor count

bm[['square_feet', 'floor_count']].corr()
def plot_group_meta(cols, df, name):

    df1 = df.groupby(cols)['meter_reading'].sum().reset_index()

    for i in list(df1[cols[0]].unique()):

        df2 = df1[df1[cols[0]]==i]

        plt.figure(figsize = (9, 9))

        sns.lineplot(df2[cols[1]], df2['meter_reading'])

        plt.title('Meter readings for {} meter for {} {}'.format(name, cols[0], i))

        

        

plot_group_meta(['site_id', 'date'], train_el, 'electricity')
build_sw(train_el[train_el['site_id']==15], ['building_id', 'month', 'dayofmonth'], 'month', 2)
build_sw(train_el[train_el['site_id']==15], ['building_id', 'month', 'dayofmonth'], 'month', 3)
plot_group_meta(['primary_use', 'date'], train_el, 'electricity')
def plot_corr(df, files):

    plt.figure(figsize = (10,8))

    sns.heatmap(df.corr(), annot = True, cmap="YlGnBu")

    plt.title('Correlation analysis between target variable and {}'.format(files))

    plt.show

corr_frame = train_el[['meter_reading', 'year_built', 'square_feet', 'floor_count']]

plot_corr(corr_frame, 'building metadata')