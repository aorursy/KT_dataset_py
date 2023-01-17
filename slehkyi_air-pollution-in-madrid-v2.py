import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()



import missingno as mno

from sklearn import linear_model



import os

print(os.listdir("../input"))
partials = list()

stations = list()



with pd.HDFStore('../input/madrid.h5') as data:

    stations = [k[1:] for k in data.keys() if k != '/master']

    for station in stations:

        df = data[station]

        df['station'] = station

        partials.append(df)
measures = pd.concat(partials, sort=False).sort_index()

measures.head()
measures.describe()
measures.isnull().sum().apply(lambda x: (x / len(measures) * 100))
mno.matrix(measures, figsize = (24, 8))
plt.figure(figsize=(12,12))

sns.heatmap(measures.drop('station', axis=1).corr(), square=True, annot=True, cmap='rainbow')
# get list of unique timestamps

RANDOM_STATION = '28079017'

one_station = measures[measures['station'] == RANDOM_STATION]

timestamps = one_station.index.to_series()
# only for EDA purposes to understand what we're dealing with

# for station in stations:

#     print("Station: {}".format(station))

#     total_observations = len(measures[measures['station'] == station])

#     print("Total observations per station: {}".format(total_observations))

#     print(measures[measures['station'] == station].notnull().sum().apply(lambda x: (x / total_observations * 100)))
column_to_check = 'NMHC'

list_of_dfs = list()

for station in stations:

    station_data = measures[measures['station'] == station]

    total_observations = len(station_data)

#     print("Total observations per station: {}".format(total_observations))

#     print(station_data[column_to_check].notnull().sum())

    if station_data[column_to_check].notnull().sum() > 0:

        list_of_dfs.append(station_data[['station',column_to_check]])



df_nmhc = pd.concat(list_of_dfs)
f = plt.figure(figsize=(30,8))

sns.scatterplot(x='date', y='NMHC', data=df_nmhc.reset_index())
f = plt.figure(figsize=(30,8))

sns.boxplot(x='station', y='NMHC', data=df_nmhc.dropna(axis=0).reset_index())
f = plt.figure(figsize=(30,8))

sns.violinplot(x='station', y='NMHC', data=df_nmhc.dropna(axis=0).reset_index())
df_nmhc_8h = df_nmhc.dropna(axis=0).groupby('station').resample('8H').mean().reset_index()

df_nmhc_8h.head()
f = plt.figure(figsize=(30,16))

ax = f.add_subplot(3,1,1)

sns.scatterplot(x='date', y='NMHC', data=df_nmhc_8h, ax=ax)

ax = f.add_subplot(3,1,2)

palette = sns.color_palette("rainbow", 16)

sns.boxplot(x='station', y='NMHC', data=df_nmhc_8h, palette=palette, ax=ax)

ax = f.add_subplot(3,1,3)

palette = sns.color_palette("rainbow", 16)

sns.violinplot(x='station', y='NMHC', data=df_nmhc_8h, palette=palette)

df_nmhc_year = df_nmhc.dropna(axis=0).resample('8H').mean()

df_nmhc_year['year'] = df_nmhc_year.index.year

df_nmhc_year.reset_index(inplace=True)
f = plt.figure(figsize=(30,16))

ax = f.add_subplot(2,1,1)

palette = sns.color_palette("rainbow", 16)

sns.boxplot(x='year', y='NMHC', data=df_nmhc_year, palette=palette, ax=ax)

ax = f.add_subplot(2,1,2)

palette = sns.color_palette("rainbow", 16)

sns.violinplot(x='year', y='NMHC', data=df_nmhc_year, palette=palette)
nan_columns = ['CO']
def random_imputation(df, feature):



    number_missing = df[feature].isnull().sum()

    observed_values = df.loc[df[feature].notnull(), feature]

    df.loc[df[feature].isnull(), feature + '_imp'] = np.random.choice(observed_values, number_missing, replace = True)

    

    return df
co_mean = measures['CO'].groupby('date').mean()
type(co_mean)
df_mean = pd.DataFrame(data=co_mean, columns=['CO'])
# group data by different time periods

df_gr_D = df_mean.groupby(pd.Grouper(freq='D')).transform(np.mean).resample('D').mean()

df_gr_M = df_gr_D.groupby(pd.Grouper(freq='M')).transform(np.mean).resample('M').mean()
# prepare final dataset

df_detailed = df_gr_D.copy()

df_detailed['year'] = df_detailed.index.year

df_detailed['month'] = df_detailed.index.month

df_detailed['day'] = df_detailed.index.day

df_detailed.head()
# finding max and min values per each month

max_vals_df = pd.DataFrame(columns=['year', 'month', 'volume'])

MIN_YEAR = min(df_detailed['year'])

MAX_YEAR = max(df_detailed['year'])

for i in range(MIN_YEAR, MAX_YEAR+1):

    max_val = max(df_detailed['CO'][df_detailed['year'] == i])

    if max_val > 0.0:

        month = df_detailed[(df_detailed['CO'] == max_val) & (df_detailed['year'] == i)]['month'].values[0]

        to_add = pd.DataFrame([[i, month, max_val]], columns=['year', 'month', 'volume'])

        max_vals_df = max_vals_df.append(to_add)



min_vals_df = pd.DataFrame(columns=['year', 'month', 'volume'])

for i in range(MIN_YEAR, MAX_YEAR+1):

    min_val = min(df_detailed['CO'][df_detailed['year'] == i])

    if min_val > 0.0:

        month = df_detailed[(df_detailed['CO'] == min_val) & (df_detailed['year'] == i)]['month'].values[0]

        to_add = pd.DataFrame([[i, month, min_val]], columns=['year', 'month', 'volume'])

        min_vals_df = min_vals_df.append(to_add)
max_vals_df['date'] = pd.to_datetime(max_vals_df.year.astype('str')+"-"+max_vals_df.month.astype('str'), format="%Y-%m")

max_vals_df = max_vals_df.set_index('date', drop=True)

max_vals_df.drop(['year', 'month'], axis=1, inplace=True)

max_vals_df
min_vals_df['date'] = pd.to_datetime(min_vals_df.year.astype('str')+"-"+min_vals_df.month.astype('str'), format="%Y-%m")

min_vals_df = min_vals_df.set_index('date', drop=True)

min_vals_df.drop(['year', 'month'], axis=1, inplace=True)

min_vals_df
# really useful function

def zero_to_nan(values):

    """Replace every 0 with 'nan' and return a copy."""

    return [float('nan') if x == 0 else x for x in values]





df_detailed['CO'] = zero_to_nan(df_detailed['CO'])
# plot 1 - difference between max and min values through time

f = plt.figure(figsize=(18,6))

ax = f.add_subplot(2,1,1)

sns.lineplot(data = min_vals_df.reset_index(), x='date', y='volume')

plt.xlabel('')

plt.ylabel('MIN CO pollution, mg/m3')

ax = f.add_subplot(2,1,2)

sns.lineplot(data = max_vals_df.reset_index(), x='date', y='volume')

plt.xlabel('Year')

plt.ylabel('MAX CO pollution, mg/m3')
# plot 2 - changes in air pollution through years

years = []

for i in range(MIN_YEAR, MAX_YEAR+1, 2):

    years.append(i)



f = plt.figure(figsize=(18,6))

plt.plot(df_detailed.index, df_detailed['CO'], c='r')

plt.xlabel('Year')

plt.ylabel('mg/m3')

plt.title('CO pollution in Madrid 2001-2018')