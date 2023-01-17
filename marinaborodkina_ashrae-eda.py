import numpy as np

import pandas as pd



import time

import math



# Visualiazation

import seaborn as sns

import matplotlib.pyplot as plt



# Import and suppress warnings

import warnings

warnings.filterwarnings('ignore')
def SecondsToStr(time_taken):

    ''' Function return hours, minutes, seconds '''

    ''' from the time in string format. '''



    hours, rest = divmod(time_taken, 3600)

    minutes, seconds = divmod(rest, 60)

    h_ = str(math.trunc(hours))

    m_ = str(math.trunc(minutes))

    s_ = str(round(seconds, 2))

    time_taken_str = ':'.join([h_, m_, s_])



    # return hours, minutes, seconds from the time taken

    return time_taken_str
def df_eda(df_, with_stat_=False):

    

    # Columns of the DataFrame

    print('columns:')

    print(df_.columns.to_list())

    # Shape (number of columns, rows)

    print('\nshape:')

    print(df_.shape)

    # Types of the columns

    print('\ntypes:')

    print(df_.dtypes)

    if with_stat_:

        # Statistic for numerical columns

        print('\nstat:')

        print(df_.describe())

        

def column_info(df_, col_):

    print(col_)

    print('')

    desc_ = df_[col_].describe()

    print(round(desc_.drop(['count']), 2))

    

def column_visualizatin(df_, col_, target_, koef_, with_target_=False):

    fig, ax = plt.subplots() 

    ax.hist(df_[col_], color='g', alpha=0.5, normed=True, label=col_) 

    

    if with_target_:

        df_grouped_ = df_[[col_, target_]].groupby([col_]).mean().reset_index()

        df_grouped_[target_] = df_grouped_[target_].astype('float')/koef_

        ax.plot(df_grouped_[col_], df_grouped_[target_], color='r', label=target_)



    ax.set(title=col_)

    ax.legend(loc='best')

    plt.show()

    

def value_distribution(df_, col_, n_):

    

    ''' Check the share '''



    print(round(df_[col_].value_counts(normalize=True)*100, 2)[:n_])

    sns.countplot(x=col_, data=df_)

    plt.xticks(rotation=90)

    

def share_of_missing_per_column(df_, df_name_):

    

    print('')

    print(df_name_)

    print('Share of missing per column:\n')

    data = []



    for col in df_.columns.to_list():

        if (df_[col].isnull().sum() > 0):

            data.append([col, '{}%'.format(round(100*df_[col].isnull().sum()/df_[col].shape[0], 2))])

    return pd.DataFrame.from_records(data, columns=['Column', 'Missing_share'])
# import train Dataset

start_time = time.time()



train = pd.read_csv('../input/ashrae-energy-prediction/train.csv')

building_metadata = pd.read_csv('../input/ashrae-energy-prediction/building_metadata.csv')

weather_train = pd.read_csv('../input/ashrae-energy-prediction/weather_train.csv')



print('Total time: {}'.format(SecondsToStr(time.time() - start_time)))
# import test Dataset

start_time = time.time()



test = pd.read_csv('../input/ashrae-energy-prediction/test.csv')

weather_test = pd.read_csv('../input/ashrae-energy-prediction/weather_test.csv')

sample_submission = pd.read_csv('../input/ashrae-energy-prediction/sample_submission.csv')



print('Total time: {}'.format(SecondsToStr(time.time() - start_time)))
# merge Datasets

start_time = time.time()



# Temporary merge for EDA

train_df = (train.merge(building_metadata, on='building_id', how='left')).merge(weather_train, on=['site_id', 'timestamp'], how='left')

test_df = (test.merge(building_metadata, on='building_id', how='left')).merge(weather_test, on=['site_id', 'timestamp'], how='left')



# Temporary merge for EDA

weather_df = weather_train.append(weather_test, ignore_index=True)

weather_columns = weather_df.columns.to_list()



print('Total time: {}'.format(SecondsToStr(time.time() - start_time)))
# df_eda(train_df, False)

# df_eda(test_df, False)



cat_cols, num_cols = [], []



for col in train_df.columns:

    if train_df[col].dtype == object:

        cat_cols.append(col)

    else:

        num_cols.append(col)

print('Numerical columns {}, categorical columns {}'.format(len(num_cols), len(cat_cols)))
print('Train shape:', train.shape)

print('Train TimeBorder:', train['timestamp'].min(), ':', train['timestamp'].max())

print('Train timestamp. Number of missing values:', train['timestamp'].isnull().sum())

train.head(2)
share_of_missing_per_column(train, 'Train')
target = 'meter_reading'
# print(round(train[target].value_counts(normalize=True)*100, 2))

column_info(train, target)

train[target].plot()

plt.show()
# Check the share of meter

value_distribution(train, 'meter', 4)
print('Test shape:', test.shape)

print('Test TimeBorder:', test['timestamp'].min(), ':', test['timestamp'].max())

print('Test timestamp. Number of missing values:', test['timestamp'].isnull().sum())

test.head(2)
share_of_missing_per_column(test, 'Test')
data = []



lst = train_df.columns.to_list()

lst.pop(lst.index(target))



for col in lst:

    if col in num_cols:

        eql_median = False



        if train_df[col].median() == test_df[col].median():

            eql_median = True



        data.append([col, 

                     train_df[col].median(),

                     test_df[col].median(),

                     eql_median,

                     train_df[col].mean(),

                     test_df[col].mean(),

                     train_df[col].var(),

                     test_df[col].var()

                     ])

pd.DataFrame.from_records(data, columns=['Column', 

                                         'Train_median', 'Test_median', 

                                         'Equal_medians',

                                         'Train_mean', 'Test_mean', 

                                         'Train_var', 'Test_var', 

                                        ])
print('Weather shape:', weather_df.shape)

print('Weather TimeBorder:', weather_df['timestamp'].min(), ':', weather_df['timestamp'].max())

print('Weather timestamp. Number of missing values:', weather_df['timestamp'].isnull().sum())



weather_df.head(2)
weather_df.columns
weather_df.dtypes
weather_df.describe()
plt.figure(figsize=(8, 8))

sns.heatmap(weather_df.corr(), square=True, annot=True)
share_of_missing_per_column(weather_df, 'Weather')
# Check the share of site_id

value_distribution(weather_df, 'site_id', 3)
column_visualizatin(train_df, 'air_temperature', target, 100000, True)
weather_df['air_temperature'].hist(color='salmon', alpha=0.5) 
column_visualizatin(train_df, 'cloud_coverage', target, 100000, True)
column_visualizatin(train_df, 'dew_temperature', target, 100000, True)
weather_df['dew_temperature'].hist(color='salmon', alpha=0.5) 
column_visualizatin(train_df, 'precip_depth_1_hr', target, 1000000, True)
column_visualizatin(train_df, 'sea_level_pressure', target, 100000, True)
column_visualizatin(train_df, 'wind_direction', target, 1000000, True)
column_visualizatin(train_df, 'wind_speed', target, 100000, True)
print(building_metadata.shape)

building_metadata.head(2)
building_metadata.dtypes
building_metadata.describe()
plt.figure(figsize=(8, 8))

sns.heatmap(building_metadata.corr(), square=True, annot=True)
share_of_missing_per_column(building_metadata, 'Buildings')
# Check the share of site_id

value_distribution(building_metadata, 'site_id', 3)
# Check the share of primary_use

value_distribution(building_metadata, 'primary_use', 10)
column_visualizatin(train_df, 'year_built', target, 100000, True)
column_visualizatin(train_df, 'floor_count', target, 10000, True)