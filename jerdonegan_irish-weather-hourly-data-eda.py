import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # plotting
print(os.listdir('../input'))
df = pd.read_csv('../input/hourly_irish_weather.csv', parse_dates=['date'])

df.drop('Unnamed: 0', axis=1, inplace=True)

nRow, nCol = df.shape

print(f'There are {nRow} rows and {nCol} columns')
df.head(5)
df.info()
df_objects = df.select_dtypes(include='object')

object_memory_usage = df_objects.memory_usage(deep=True).sum()/ 1024**2

print(f"Total memory usage for Objects is {object_memory_usage :3.2f}mb's")



total_memory_usage = df.memory_usage(deep=True).sum() / 1024 ** 2

print(f"Total memory usage is {total_memory_usage :3.2f}mb's")
df_category = df_objects.astype('category')



category_memory_usage = df_category.memory_usage(deep=True).sum()/ 1024**2

print(f"Total memory usage for Categories is {category_memory_usage:3.2f}mb's")
object_columns = df.select_dtypes(include='object').columns



df[object_columns] = df[object_columns].astype('category')
df.isna().mean()
def heatmap(df):

    sns.set(rc={'figure.figsize':(20,20)})

    sns.set_context('talk')

    sns.heatmap(df.round(2), cbar=None, annot=True, linewidth=.5, cmap='YlGnBu')
station_na_dict = {}

for station in df.station.cat.categories:

    station_na_dict[station] = df[df.station == station].isna().mean()



station_na_df = pd.DataFrame(station_na_dict)



heatmap(station_na_df)

plt.title('Proportion of missing values by Weather Station')
yearly_na_dict = {}



for year in df.date.dt.year.unique():

    yearly_na_dict[year] = df[df.date.dt.year == year].isna().mean()



yearly_na_df = pd.DataFrame(yearly_na_dict)



heatmap(yearly_na_df)

plt.title('Proportion of missing values by Year')
sun_year_station = df.groupby([df.date.dt.year, 'station']).count()['sun'].reset_index()

sun_pivoted = sun_year_station.pivot_table(values='sun', index='station', columns='date', fill_value=0)



heatmap(sun_pivoted/1000)

plt.yticks(rotation=0)

plt.title('Sun data Points by year and station (1000\'s) ')
def aggerate_data(grouper):

    # Drop Lat, Long, w and ww columns

    agg_df = df.drop(['latitude', 'longitude', 'w', 'ww'], axis=1).groupby(grouper).agg([np.min, np.max, np.mean, np.std])

    agg_df.columns = ['_'.join(val) for val in agg_df.columns.values]

    return agg_df



month_year_agg_df = aggerate_data([df.date.dt.year, df.date.dt.month])



# Rename and reset index

month_year_agg_df.index.rename(['year', 'month'], inplace = True)

month_year_agg_df.reset_index(inplace=True)



# Combine year and Month

month_year_agg_df['day'] = 15

month_year_agg_df['date'] = pd.to_datetime(month_year_agg_df[['day','month', 'year']])

month_year_agg_df.drop(['day', 'month', 'year'], axis=1, inplace=True)



#create new date index

month_year_agg_df.index = month_year_agg_df.date

month_year_agg_df.drop('date', axis=1, inplace=True)



month_year_agg_df.head()
def plot_aggerated_data_timeseries(df, variable, title='Monthly'):

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(30, 20))

    fig.suptitle(f'{title} Average - {variable.capitalize()}')

    

    ax1.plot(df.index, df[f'{variable}_mean'])

    upper_sd = df[f'{variable}_mean'] + df[f'{variable}_std']

    lower_sd = df[f'{variable}_mean'] - df[f'{variable}_std']

    ax1.fill_between(df.index, y1=upper_sd, y2=lower_sd, alpha=.2)

    ax1.set_title(f'{variable.capitalize()} - Average and Standard Deviation')

   

    ax2.plot(df.index, df[f'{variable}_mean'])

    ax2.fill_between(df.index, df[f'{variable}_amin'], df[f'{variable}_amax'], alpha=.2)

    ax2.set_title(f'{variable.capitalize()} - Average, Min, Max')

    plt.show()
cols = set(c[0] for c in month_year_agg_df.columns.str.split('_'))



for col in cols:

    plot_aggerated_data_timeseries(month_year_agg_df, col, title='Month, Year')
monthly_df = aggerate_data(df.date.dt.month)



for col in cols:

    plot_aggerated_data_timeseries(monthly_df, col, title='Monthly')