import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import seaborn as sns

import geopandas as gpd

import folium

from folium.plugins import HeatMap

import re



%matplotlib inline

plt.rcParams['figure.figsize'] = (12, 5);

sns.set_style('whitegrid')
with open('../input/restaurant-recommendation-challenge/VariableDefinitions.txt') as f:

    print(f.read())
chunk_size=10000

root_dir = '../input/restaurant-recommendation-challenge'

train_full = pd.read_csv(os.path.join(root_dir, 'train_full.csv'))

orders = pd.read_csv(os.path.join(root_dir, 'orders.csv'))

vendors = pd.read_csv(os.path.join(root_dir, 'vendors.csv'))
train_full = train_full.sample(chunk_size)

orders = orders.sample(chunk_size)
train_full.head()
orders.head()
train_full.isna().sum().sum()
fig, ax = plt.subplots(1, 2)

train_full['gender'].hist(ax=ax[0], color='yellow')

train_full['location_type'].hist(ax=ax[1])
train_full['country_id'].value_counts()
locs = gpd.read_file(os.path.join(root_dir, 'train_locations.csv'))

locs.dropna(subset=['latitude'], inplace=True)

locs.head()
def check_num(string):

    regex = r'-?[0-9]*.[0-9]*'

    m = re.match(regex, string)

    if m is None:

        return float(0)

    return float(string[:6])



locs['latitude'] = locs['latitude'].apply(check_num)

locs['longitude'] = locs['longitude'].apply(check_num)

locs['geometry'] = gpd.points_from_xy(locs['longitude'], locs['latitude'])
m = folium.Map(location=[50,-85], zoom_start=2)

for i in list(locs.index)[:50]:

    folium.Marker([locs.loc[i, 'latitude'], locs.loc[i, 'longitude']]).add_to(m)

m
sorted(train_full['location_number'].unique())
train_full[['status_x', 'status_y']].hist(color='magenta')
train_full['discount_percentage'].value_counts()
train_full['commission'].unique()
train_full['display_orders'].value_counts()
train_full['target'].sum()
train_full['rank'].hist()
train_full['prepration_time'].hist(color='gold')
orders.head()
orders.describe()
plt.hist(orders['payment_mode']);
sns.heatmap(orders.corr(), cmap="YlGnBu")
fig, ax = plt.subplots(1, 2)

sns.distplot(orders['grand_total'], ax=ax[0], color='purple')

sns.distplot(orders['item_count'], ax=ax[1])
fig, ax = plt.subplots(1, 2)

orders.loc[:, 'delivery_date'] = pd.to_datetime(orders['delivery_date'])

ax[0].scatter(orders.set_index('delivery_date').index, orders['item_count'], 

              label='items', alpha=0.6, color='red')

ax[0].legend();

ax[1].scatter(orders.set_index('delivery_date').index, orders['grand_total'], 

              label='total pay', alpha=0.6, color='green')

ax[1].legend();
orders.loc[:, 'delivery_time'] = pd.to_datetime(orders['delivery_time'], errors='coerce')

for i in range(0, 24):

    df = orders[orders['delivery_time'].dt.hour==i]

    orders.loc[df.index, 'delivery_hour'] = i

orders['delivery_hour'].hist(bins=24, label='orders by hour of day')

plt.legend();
orders.groupby('customer_id').mean()['grand_total'].plot(marker='.', linestyle='none', color='orange')

plt.title('total cost');
orders[orders['grand_total']==0.0]
orders[orders['grand_total']==0.0]['promo_code'].isna().sum(), orders[orders['grand_total']==0.0].shape
customers = pd.read_csv('../input/restaurant-recommendation-challenge/train_customers.csv')

customers.head()
customers['akeed_customer_id'].nunique(), customers.shape[0]
dists = ['gender', 'language', 'status', 'verified']

d=0

fig, ax = plt.subplots(2, 2)

for i in range(2):

    for j in range(2):

        customers[dists[d]].dropna().hist(ax=ax[i][j], label=dists[d], color='aqua')

        if dists[d] == 'gender':

            ax[i][j].tick_params(rotation=45)

        ax[i][j].legend();

        plt.tight_layout();

        d+=1
def clean_string(string):

    string = str(string)

    if '?' in string or string=='nan' or string.strip(' ')=='':

        return np.nan

    string = string.strip(' ').lower()

    return string



customers.loc[:, 'gender'] = customers['gender'].apply(clean_string)

customers['gender'].hist(color='chocolate')
def calc_age(year):

    if len(str(year))==2:

        if str(year).startswith('0'):

            year = '20'+str(year)

        else:

            year = '19'+str(year)

        year = int(year)

    if year is None:

        return np.nan

    return 2020-year



customers.loc[:, 'age'] = customers['dob'].apply(calc_age)

customers[customers['age']<16]
ages = customers[customers['age']>16]

ages = ages[ages['age']<110]

ages['age'].dropna().hist(bins=20, label='customers by age', color='brown')