import warnings

import os

import numpy as np

import pandas as pd

import math

import matplotlib.pyplot as plt

import seaborn as sns



warnings.simplefilter("ignore")

%matplotlib inline



print(os.listdir("../input"))
df_train = pd.read_csv(os.path.join('..', 'input', 'train.csv'), index_col=0)
df_train.head()
df_train.dtypes
df_train.duplicated().sum()
df_train = df_train.drop_duplicates()

df_train.duplicated().sum()
df_train.isna().sum()
describe = df_train.describe()



def seconds_to_pretty(seconds):

    seconds = int(float(seconds))

    return '{} h {} m {} s ({} s)'.format(seconds // 3600, (seconds % 3600) // 60, (seconds % 3600) % 60, seconds)



describe = describe.drop('count') # to disable scientific notation

describe['trip_duration'] = describe['trip_duration'].apply(str)

describe['trip_duration'].loc[['mean', 'min', '25%', '50%', '75%', 'max']] = describe['trip_duration'].loc[['mean', 'min', '25%', '50%', '75%', 'max']].apply(seconds_to_pretty)

describe
# def pretty_print_max_trip_duration(df):   

#     max_trip_duration = df_train['trip_duration'].max()

#     print('Max duration trip: ({} sec) -> {} hours {} minutes {} secondes'.format(max_trip_duration, max_trip_duration // 3600, (max_trip_duration % 3600) // 60, (max_trip_duration % 3600) % 60))



# pretty_print_max_trip_duration(df_train)
# let's make a boxplot to better trip_duration outliers visualization

# we need to see if there is any correlation between trip_duration and store_and_fwd_flag

def boxplot_trip_duration(df, ylim=None):

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

    plt.subplots_adjust(wspace=1)

    if ylim:

        ax.set_ylim(ylim)

    sns.boxplot(data=df_train, y='trip_duration', x='store_and_fwd_flag', fliersize=5, ax=ax).axes.set_title(label='Trip duration outliers visualization \n (store_and_fwd_flag = N)', fontsize=16, pad=25)

#     sns.boxplot(y=df_train[df_train['store_and_fwd_flag'] == 'N']['trip_duration'], fliersize=5, ax=ax1).axes.set_title(label='Trip duration outliers visualization \n (store_and_fwd_flag = N)', fontsize=16, pad=25)

#     sns.boxplot(y=df_train[df_train['store_and_fwd_flag'] == 'Y']['trip_duration'], fliersize=5, ax=ax2).axes.set_title(label='Trip duration outliers visualization \n (store_and_fwd_flag = Y)', fontsize=16, pad=25)

    

boxplot_trip_duration(df_train)
with_n = len(df_train[df_train['store_and_fwd_flag'] == 'N'].index)

with_y = len(df_train[df_train['store_and_fwd_flag'] == 'Y'].index)



len(df_train[df_train['store_and_fwd_flag'] == 'N'].index), len(df_train[df_train['store_and_fwd_flag'] == 'Y'].index)
seconds_to_pretty(df_train[df_train['store_and_fwd_flag'] == 'N']['trip_duration'].max()), seconds_to_pretty(df_train[df_train['store_and_fwd_flag'] == 'Y']['trip_duration'].max())
boxplot_trip_duration(df_train, [0, df_train[df_train['store_and_fwd_flag'] == 'N']['trip_duration'].mean()])
_ = sns.distplot(df_train['passenger_count'], hist=False, rug=True)
_ = sns.jointplot(data=df_train, x="pickup_latitude", y="pickup_longitude")
_ = sns.jointplot(data=df_train, x="dropoff_latitude", y="dropoff_longitude")
df_test = pd.read_csv(os.path.join('..', 'input', 'test.csv'), index_col=0)

df_test.head()
len(df_test.index)
df_train.describe().drop('count')
df_test.describe().drop('count')
_ = sns.distplot(df_train['passenger_count'], hist=False, rug=True).set_title(label='Train dataset', fontsize=16, pad=25)
_ = sns.distplot(df_test['passenger_count'], hist=False, rug=True).set_title(label='Test dataset', fontsize=16, pad=25)
_ = sns.jointplot(data=df_test, x="pickup_latitude", y="pickup_longitude")
_ = sns.jointplot(data=df_test, x="dropoff_latitude", y="dropoff_longitude")
max_trip_duration_flag_y = df_train[df_train['store_and_fwd_flag'] == 'Y']['trip_duration'].max()

row_count_before_deletion = len(df_train.index)

print('{} lines before deletion'.format(row_count_before_deletion))

df_train = df_train[~((df_train['store_and_fwd_flag'] == 'N') & (df_train['trip_duration'] > max_trip_duration_flag_y))]

row_count_after_deletion = len(df_train.index)

print('{} lines after deletion'.format(row_count_after_deletion))

print('{} lines deleted'.format(row_count_before_deletion - row_count_after_deletion))
boxplot_trip_duration(df_train)
_ = sns.distplot(df_train[df_train['store_and_fwd_flag'] == 'Y']['trip_duration'], hist=False, rug=True)
df_train[((df_train['store_and_fwd_flag'] == 'Y') & (df_train['trip_duration'] > 6500))]['trip_duration'].count()
# row_count_before_deletion = len(df_train.index)

# print('{} lines before deletion'.format(row_count_before_deletion))

# df_train = df_train[df_train['trip_duration'] <= 6500]

# row_count_after_deletion = len(df_train.index)

# print('{} lines after deletion'.format(row_count_after_deletion))

# print('{} lines deleted'.format(row_count_before_deletion - row_count_after_deletion))
# _ = sns.distplot(df_train['trip_duration'], hist=False, rug=True)
row_count_before_deletion = len(df_train.index)

print('{} lines before deletion'.format(row_count_before_deletion))

df_train = df_train[df_train['trip_duration'] >= 10]

row_count_after_deletion = len(df_train.index)

print('{} lines after deletion'.format(row_count_after_deletion))

print('{} lines deleted'.format(row_count_before_deletion - row_count_after_deletion))
# df_train[df_train['passenger_count'] < 1]['passenger_count'].count()
# row_count_before_deletion = len(df_train.index)

# print('{} lines before deletion'.format(row_count_before_deletion))

# df_train = df_train[df_train['passenger_count'] >= 1]

# row_count_after_deletion = len(df_train.index)

# print('{} lines after deletion'.format(row_count_after_deletion))

# print('{} lines deleted'.format(row_count_before_deletion - row_count_after_deletion))
# max_test_pickup_latitude = df_test['pickup_latitude'].max()

# min_test_pickup_latitude = df_test['pickup_latitude'].min()



# max_test_pickup_longitude = df_test['pickup_longitude'].max()

# min_test_pickup_longitude = df_test['pickup_longitude'].min()





# max_test_dropoff_latitude = df_test['dropoff_latitude'].max()

# min_test_dropoff_latitude = df_test['dropoff_latitude'].min()



# max_test_dropoff_longitude = df_test['dropoff_longitude'].max()

# min_test_dropoff_longitude = df_test['dropoff_longitude'].min()



# (max_test_pickup_latitude, min_test_pickup_latitude, '---',

#  max_test_pickup_longitude, min_test_pickup_longitude, '---',

#  max_test_dropoff_latitude, min_test_dropoff_latitude, '---',

#  max_test_dropoff_longitude, min_test_dropoff_longitude)
# row_count_before_deletion = len(df_train.index)

# print('{} lines before deletion'.format(row_count_before_deletion))



# df_train['pickup_latitude'] = df_train[df_train['pickup_latitude'] <= max_test_pickup_latitude]['pickup_latitude']

# df_train['pickup_latitude'] = df_train[df_train['pickup_latitude'] >= min_test_pickup_latitude]['pickup_latitude']



# df_train['pickup_longitude'] = df_train[df_train['pickup_longitude'] <= max_test_pickup_longitude]['pickup_longitude']

# df_train['pickup_longitude'] = df_train[df_train['pickup_longitude'] >= min_test_pickup_longitude]['pickup_longitude']





# df_train['dropoff_latitude'] = df_train[df_train['dropoff_latitude'] <= max_test_dropoff_latitude]['dropoff_latitude']

# df_train['dropoff_latitude'] = df_train[df_train['dropoff_latitude'] >= min_test_dropoff_latitude]['dropoff_latitude']



# df_train['dropoff_longitude'] = df_train[df_train['dropoff_longitude'] <= max_test_dropoff_longitude]['dropoff_longitude']

# df_train['dropoff_longitude'] = df_train[df_train['dropoff_longitude'] >= min_test_dropoff_longitude]['dropoff_longitude']



# row_count_after_deletion = len(df_train.index)

# print('{} lines after deletion'.format(row_count_after_deletion))

# print('{} lines deleted'.format(row_count_before_deletion - row_count_after_deletion))
from geopy.distance import geodesic



def create_datetime_based_columns(df):

    df['datetime'] =  pd.to_datetime(df['pickup_datetime'])

    

    df['year'] = df['datetime'].dt.year # year seems to help

    df['month'] = df['datetime'].dt.month # month doesn't help at all

    df['day'] = df['datetime'].dt.day

    df['dayofweek'] = df['datetime'].dt.weekday

    df['hour'] = df['datetime'].dt.hour

    

    return df
def filter_feature_columns(df):

    selected_columns = []

    selected_columns = ['pickup_longitude', 'pickup_latitude'] 

    selected_columns += ['dropoff_longitude', 'dropoff_latitude']

    selected_columns += ['month', 'dayofweek', 'hour']

#     selected_columns += ['store_and_fwd_flag']

#     selected_columns += ['lat_distance', 'long_distance']

    return df[selected_columns]



def filter_target_column(df):

    return df['trip_duration']

    



def filter_split_dataset(df):

    X = filter_feature_columns(df)

    y = filter_target_column(df)

    

    return X, y

 
df_train_copy = df_train.copy() # we have to work on a copy of df_train to be able to repeat the operations from 0 without having to reload the dataset

df_train_copy = create_datetime_based_columns(df_train_copy)

# tmp_df_train = normalize_store_and_fwd_flag(tmp_df_train)

# df_train_copy = create_distance_column(df_train_copy)



df_train_copy.head()
X_train, y_train = filter_split_dataset(df_train_copy)

X_train.shape, y_train.shape
X_train.head()
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import RandomizedSearchCV # test des combinaisons aléatoirent de parametres

from sklearn.model_selection import GridSearchCV # test toutes les combinaisons
# thi part take to much time, so I comment it

# rfr_gcv = RandomForestRegressor()

# param_distributions = {

#     'n_estimators' = [10, 100, 200, 300]

#     'min_samples_leaf' = [1, 5, 10]

#     'min_samples_split' = [2, 10, 15]

#     'max_depth' = [10, 40, 80, 90]

# }



# rs = GridSearchCV(rfr_gcv, param_distributions, scoring='neg_mean_squared_log_error')

# rs.fit(X_train, y_train)

# rs.best_params_
# rfr = RandomForestRegressor(n_estimators=19, min_samples_leaf=10, 

#                             min_samples_split=15, max_features='auto', max_depth=80, bootstrap=True)



# RandomForestRegressor with best_params

# rfr = RandomForestRegressor(n_estimators=300, min_samples_leaf=10, min_samples_split=15, 

#                             max_features='auto', max_depth=90, bootstrap=True) # that take too long time :/



rfr = RandomForestRegressor(n_estimators=30, min_samples_leaf=10, min_samples_split=15, 

                            max_features='auto', max_depth=90, bootstrap=True)



s_split = ShuffleSplit(n_splits=4, train_size=.12, test_size=.6) # allows to test on less data, so the cross validation takes less time



# I commented this line to improve kernel execution time

# np.sqrt(-cross_val_score(rfr, X_train, y_train, cv=s_split, scoring='neg_mean_squared_log_error', n_jobs=-1)).mean()
rfr.fit(X_train, y_train)
df_test_copy = df_test.copy()

df_test_copy = create_datetime_based_columns(df_test_copy)

# tmp_df_test = normalize_store_and_fwd_flag(tmp_df_test)

X_test = filter_feature_columns(df_test_copy)



y_test_pred = rfr.predict(X_test)

# 

submission = pd.DataFrame({'id': df_test.index, 'trip_duration': y_test_pred})

submission.head()
submission.to_csv('submission.csv', index=False)

!ls