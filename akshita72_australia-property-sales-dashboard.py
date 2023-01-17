# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/aus-real-estate-sales-march-2019-to-april-2020/aus-property-sales-sep2018-april2020.csv')

data.head()
data.dtypes
data['date_sold'] = pd.to_datetime(data['date_sold'])
data_city_price = data[['city_name','price','property_type']]
import matplotlib.pyplot as plt



per = (data_city_price.isnull().sum()/data.shape[0])*100



per.plot.barh()

plt.title('Missing Values')

plt.xlabel('Percentage')

plt.show()
data_cp = data_city_price.dropna()

data_cp.shape
import seaborn as sns



fig_dims = (14, 10)

fig, ax = plt.subplots(figsize=fig_dims)

sns.barplot(x='city_name', y='price', hue='property_type', ax=ax, data=data_cp)

ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

ax.set_xlabel('City')

ax.set_ylabel('Prices')

plt.show()
sydney = data[data['city_name']=='Sydney'][['date_sold','price','property_type']]

sydney.shape
sydney = sydney.dropna()

sydney.shape
sydney.sort_values('date_sold')
sydney_unit = sydney[sydney['property_type']=='unit'][['date_sold','price']]

sydney_house = sydney[sydney['property_type']=='house'][['date_sold','price']]

sydney_townhouse = sydney[sydney['property_type']=='townhouse'][['date_sold','price']]
sales_by_month_u = sydney_unit['date_sold'].groupby([sydney_unit['date_sold'].dt.year, 

                                                   sydney_unit['date_sold'].dt.month]).agg('count') 

sales_by_month_u = sales_by_month_u.to_frame()

sales_by_month_u['date'] = sales_by_month_u.index

sales_by_month_u = sales_by_month_u.rename(columns={sales_by_month_u.columns[0]:"sales"})

sales_by_month_u['date'] = pd.to_datetime(sales_by_month_u['date'], format="(%Y, %m)")

sales_by_month_u = sales_by_month_u.reset_index(drop=True)

sales_by_month_u['month'] = sales_by_month_u.date.dt.month
sales_by_month_h = sydney_house['date_sold'].groupby([sydney_house['date_sold'].dt.year, 

                                                   sydney_house['date_sold'].dt.month]).agg('count') 

sales_by_month_h = sales_by_month_h.to_frame()

sales_by_month_h['date'] = sales_by_month_h.index

sales_by_month_h = sales_by_month_h.rename(columns={sales_by_month_h.columns[0]:"sales"})

sales_by_month_h['date'] = pd.to_datetime(sales_by_month_h['date'], format="(%Y, %m)")

sales_by_month_h = sales_by_month_h.reset_index(drop=True)

sales_by_month_h['month'] = sales_by_month_h.date.dt.month
sales_by_month_th = sydney_townhouse['date_sold'].groupby([sydney_townhouse['date_sold'].dt.year, 

                                                   sydney_townhouse['date_sold'].dt.month]).agg('count') 

sales_by_month_th = sales_by_month_th.to_frame()

sales_by_month_th['date'] = sales_by_month_th.index

sales_by_month_th = sales_by_month_th.rename(columns={sales_by_month_th.columns[0]:"sales"})

sales_by_month_th['date'] = pd.to_datetime(sales_by_month_th['date'], format="(%Y, %m)")

sales_by_month_th = sales_by_month_th.reset_index(drop=True)

sales_by_month_th['month'] = sales_by_month_th.date.dt.month
fig_dims = (14, 10)

fig, ax = plt.subplots(figsize=fig_dims)

sns.lineplot(x='date', y='sales', ax=ax, data=sales_by_month_u,label='unit',linewidth=5)

sns.lineplot(x='date', y='sales', ax=ax, data=sales_by_month_h, label='house',linewidth=5)

sns.lineplot(x='date', y='sales', ax=ax, data=sales_by_month_th, label='townhouse',linewidth=5)

ax.set_xlabel('Date')

ax.set_ylabel('Number of Property Sales')

ax.set_title('Sydney Property Sales Trends')

plt.legend()

plt.show()
d2020 = data[data['date_sold'].dt.year == 2020][['state','lat','lon']]
d2020.dropna(inplace=True)
from mpl_toolkits.basemap import Basemap





fig, ax = plt.subplots(figsize=(10,20))

m = Basemap(resolution='l', # c, l, i, h, f or None

            projection='merc',

            lat_0=25.27, lon_0=133.77,

            llcrnrlon=110., llcrnrlat=-45, urcrnrlon=155., urcrnrlat=-10)

m.etopo(scale=0.5, alpha=0.5)



x, y = m(d2020['lon'].to_list(), d2020['lat'].to_list())

plt.plot(x, y, 'ok', markersize=5)
