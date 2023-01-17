!pip install pygeohash

import pygeohash as pgh

import seaborn as sns

import numpy as np 

import pandas as pd 

import datetime

import calendar

import matplotlib.pyplot as plt

plt.style.use('ggplot')



# from tsfresh.examples.har_dataset import download_har_dataset, load_har_dataset, load_har_classes

# from tsfresh import extract_features, extract_relevant_features, select_features

# from tsfresh.utilities.dataframe_functions import impute

# from tsfresh.feature_extraction import settings

# import geopandas

# import geopy

# from geopy.extra.rate_limiter import RateLimiter

# from geopy.geocoders import Nominatim



import folium

import os

from tqdm import tqdm

from IPython.display import display
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
files = []



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        if 'addresses' in filename:

            continue

        files.append(os.path.join(dirname, filename))

cols = pd.read_csv(files[0]).columns

df = pd.concat([pd.read_csv(file, header=None, skiprows=1) for file in files])

df.columns = cols
df['Year'] = pd.to_datetime(df['Order Date'], errors='coerce').dt.year

df['Month'] = pd.to_datetime(df['Order Date'], errors='coerce').dt.month

df['Day'] = pd.to_datetime(df['Order Date'], errors='coerce').dt.dayofyear

df['Date'] = pd.to_datetime(df['Order Date'], errors='coerce').dt.date

df['Week'] = pd.to_datetime(df['Order Date'], errors='coerce').dt.week

df['Time'] = pd.to_datetime(df['Order Date'], errors='coerce').dt.time

df.drop('Order Date', inplace=True, axis=1)

df.drop(df['Product'] == 'Product', inplace=True)



df.dropna(inplace=True)

df[['Price Each', 'Quantity Ordered', 'Day', 'Week']] = df[['Price Each', 'Quantity Ordered', 'Day', 'Week']].astype('float32')

df['total'] = df['Quantity Ordered'] * df['Price Each']



df.to_csv('full_df.csv')

df.head()
plt.barh(np.arange(len(df['Product'].value_counts())), df['Product'].value_counts(ascending=True))

plt.yticks(np.arange(len(df['Product'].value_counts())), df['Product'].value_counts(ascending=True).index);
top_7 = list(df['Product'].value_counts(ascending=True).index[12:])
month_sales = df.pivot_table(values='Quantity Ordered', index='Month', columns='Product', aggfunc='sum')

month_sales
displays = ['20in Monitor', '27in 4K Gaming Monitor', '27in FHD Monitor', '34in Ultrawide Monitor']

apple = ['Apple Airpods Headphones', 'Lightning Charging Cable', 'Macbook Pro Laptop', 'iPhone']

batteries = ['AA Batteries (4-pack)', 'AAA Batteries (4-pack)']

others = ['Bose SoundSport Headphones', 'Flatscreen TV', 'Google Phone', 'LG Dryer', 

          'LG Washing Machine', 'ThinkPad Laptop', 'USB-C Charging Cable','Vareebadd Phone', 

          'Wired Headphones']
fig, axes = plt.subplots(2, 2, figsize=(20, 10))

axes = axes.flatten()

for ax, product in zip(axes, [displays, apple, batteries, others]):

    ax.plot(month_sales[product]); 



    ax.legend(month_sales[product], ncol=3, fontsize=11)

    ax.set_ylim(0, 4500)

    ax.set_xticks(np.arange(13))

    ax.set_xticklabels(calendar.month_name, rotation=40)

fig.suptitle('Monthly sales with rollin mean and std', y=.91, x=.51);
def moving_average(product_name, n):

    series = df[df['Product'] == product_name].groupby('Week').count()['Quantity Ordered']

    

    rolling_mean = series.rolling(window=n).mean()



    rolling_std =  series.rolling(window=n).std()

    upper_bound = rolling_mean+1.96*rolling_std

    lower_bound = rolling_mean-1.96*rolling_std

    return [series[n:], rolling_mean, upper_bound, lower_bound]
fig, axes = plt.subplots(6, 3, figsize=(20, 20))

axes = axes.flatten()

for product, ax in zip(df.Product.unique(), axes):

#     pass

    sales, rol, ub, lb = moving_average(product, 3)

    ax.plot(sales, 'r');

    ax.plot(rol, 'b--')

    ax.plot(ub, c='gray', ls='-.')

    ax.plot(lb, c='gray', ls='-.')

    ax.set_title(product, fontsize=10)

fig.suptitle('Weekly sales (red) with rollin mean (blue) and std (gray)', y=.91, x=.51);



def exp_smoothing(product, alpha):

    series = df[df['Product'] == product].groupby('Week').count()['Quantity Ordered'].values

    result = [series[0]] 

    for n in range(1, len(series)):

        result.append(alpha * series[n] + (1 - alpha) * result[n-1])

    return result



def double_exp_smoothing(product, alpha, beta):

    series = df[df['Product'] == product].groupby('Week').count()['Quantity Ordered'].values

    result = [series[0]]

    for n in range(1, len(series)+1):

        if n == 1:

            level, trend = series[0], series[1] - series[0]

        if n >= len(series):

            value = result[-1]

        else:

            value = series[n]

        last_level, level = level, alpha*value + (1-alpha)*(level+trend)

        trend = beta*(level-last_level) + (1-beta)*trend

        result.append(level+trend)

    return result
fig, axes = plt.subplots(6, 3, figsize=(20, 20))

axes = axes.flatten()

for product, ax in zip(df.Product.unique(), axes):

#     pass

    exp_sm = exp_smoothing(product, .5)

    ax.plot(exp_sm, 'b--');

    ax.plot(df[df['Product'] == product].groupby('Week').count()['Quantity Ordered'], 'r')

    ax.set_title(product, fontsize=10)

fig.suptitle('Weekly sales (red) with rollin mean (blue): exp_smoothing', y=.91, x=.51);



fig, axes = plt.subplots(6, 3, figsize=(20, 20))

axes = axes.flatten()

for product, ax in zip(df.Product.unique(), axes):

#     pass

    double_exp_sm = double_exp_smoothing(product, .5, .5)

    ax.plot(double_exp_sm, 'b--');

    ax.plot(df[df['Product'] == product].groupby('Week').count()['Quantity Ordered'], 'r')

    ax.set_title(product, fontsize=10)

fig.suptitle('Weekly sales (red) with rollin mean (blue): double exp smoothing', y=.91, x=.51);
fig, ax1 = plt.subplots(figsize=(25, 5))



ax1.set_xlabel('Days')

ax1.set_ylabel('Total cost per day', color='tab:red')

ax1.plot(df.groupby('Day')['total'].sum(), alpha=.5)

ax1.tick_params(axis='y')

ax1.grid(False)

ax2 = ax1.twinx()  

ax2.set_ylabel('Mean price per day', color='tab:blue')  # we already handled the x-label with ax1

ax2.plot(df.groupby('Day')['Price Each'].mean(), color='tab:blue', alpha=.5)

ax2.grid(False)

ax2.tick_params(axis='y', labelcolor='tab:blue')

plt.title('Mean price and total cost per day');
fig, ax1 = plt.subplots(figsize=(25, 5))



ax1.set_xlabel('Days')

ax1.set_ylabel('Total cost per day', color='tab:red')

ax1.plot(df.groupby('Day')['total'].sum(), alpha=.5)

ax1.tick_params(axis='y')

ax1.grid(False)

ax2 = ax1.twinx()  

ax2.set_ylabel('Number of items', color='tab:blue')

ax2.plot(df.groupby('Day')['Quantity Ordered'].count(), color='tab:blue', alpha=.5)

ax2.grid(False)

ax2.tick_params(axis='y', labelcolor='tab:blue')

plt.title('Number of items and total cost per day');
fig, ax1 = plt.subplots(figsize=(25, 5))



ax1.set_xlabel('Week')

ax1.set_ylabel('Total cost per week', color='tab:red')

ax1.plot(df.groupby('Week')['total'].sum(), alpha=.5)

ax1.tick_params(axis='y')

ax1.grid(False)

ax2 = ax1.twinx()  

ax2.set_ylabel('Mean price per week', color='tab:blue')  # we already handled the x-label with ax1

ax2.plot(df.groupby('Week')['Price Each'].mean(), color='tab:blue', alpha=.5)

ax2.grid(False)

ax2.tick_params(axis='y', labelcolor='tab:blue')

plt.title('Mean price and total cost per week');
fig, ax1 = plt.subplots(figsize=(25, 5))



ax1.set_xlabel('Week')

ax1.set_ylabel('Total cost per week', color='tab:red')

ax1.plot(df.groupby('Week')['total'].sum(), alpha=.5)

ax1.tick_params(axis='y')

ax1.grid(False)

ax2 = ax1.twinx()  

ax2.set_ylabel('Number of items', color='tab:blue')

ax2.plot(df.groupby('Week')['Quantity Ordered'].count(), color='tab:blue', alpha=.5)

ax2.grid(False)

ax2.tick_params(axis='y', labelcolor='tab:blue')

plt.title('Number of items and total cost per week');
addresses = pd.read_csv('/kaggle/input/addresses-for-sales-analysis-2019/addresses.csv', index_col='Unnamed: 0')

addresses.head()
df_add = pd.merge(df, addresses, left_on='Purchase Address', right_on='address')
#worst way. May be in future



lat = []

lon = []

for i in tqdm(np.arange(df_add.shape[0])):

    long = float(df_add.iloc[i]['location'][1:-1][:df_add.iloc[i]['location'][1:-1].find(', ')])

    lati = float(df_add.iloc[i]['location'][1:-1][df_add.iloc[i]['location'][1:-1].find(', ')+1:])

    lat.append(lati)

    lon.append(long)

df_add['lat'] = lat

df_add['lon'] = lon

df_add.drop('location', axis=1)

df_add.head()
m = folium.Map(location=[29.627060, -96.052370], tiles="OpenStreetMap", zoom_start=4)

sample = df_add.sample(1000)

for i in range(sample.shape[0]):

    folium.Marker([sample.iloc[i]['lon'], sample.iloc[i]['lat']], 

                  popup=sample.iloc[i]['Purchase Address']).add_to(m)

    

m
df_add['gh'] = df_add.apply(lambda x: pgh.encode(x['lat'], x['lon'], precision=6), axis=1)
plt.figure(figsize=(15, 5))

sns.heatmap(pd.crosstab(df_add[df_add['Product'].isin(top_7)]['Product'], 

                        df_add[df_add['Product'].isin(top_7)]['gh']), 

            cmap="gist_earth")
plt.figure(figsize=(14, 4))

plt.bar(df_add['Quantity Ordered'].groupby(df_add['gh']).sum().sort_values(ascending=False).index[:10], 

        df_add['Quantity Ordered'].groupby(df_add['gh']).sum().sort_values(ascending=False)[:10], align='center')

plt.xticks(rotation=40, ha='right');
df_add.to_csv('df_add_gh.csv')