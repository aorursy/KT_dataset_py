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
        
import matplotlib.pyplot as plt
import seaborn as s
s.set()
from datetime import datetime

#geo exploratory 

geo_data = pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_geolocation_dataset.csv')
print(geo_data.head())

from mpl_toolkits.basemap import Basemap
import os
import conda

conda_file_dir = conda.__file__
conda_dir = conda_file_dir.split('lib')[0]
proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
os.environ["PROJ_LIB"]=proj_lib

lat=geo_data['geolocation_lat']
lon=geo_data['geolocation_lng']

plt.figure(figsize=(10,10))
m = Basemap(llcrnrlat=-55.401805,llcrnrlon=-92.269176,urcrnrlat=13.884615,urcrnrlon=-27.581676)
m.bluemarble()
m.drawmapboundary(fill_color='#46bcec')
m.fillcontinents(color='#f2f2f2', lake_color = '#46bcec')
m.drawcountries()
m.scatter(lon,lat,zorder=10,alpha =0.5, color='tomato')

plt.figure(figsize=(10,10))
s.countplot(x='geolocation_state',data=geo_data, order=geo_data['geolocation_state'].value_counts().sort_values().index)

# orders_dataset
order_data = pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_orders_dataset.csv')
order_data.isnull().sum()

#feature engineering with time

order_data['order_delivered_customer_date']= order_data['order_delivered_customer_date'].fillna(method='ffill')
order_data['order_delivered_customer_date'].isnull().sum()

order_data['delivered_time']=pd.to_datetime(order_data['order_delivered_customer_date'],
                                            format='%Y-%m-%d').dt.date
order_data['estimate_time']=pd.to_datetime(order_data['order_estimated_delivery_date'], 
                                           format='%Y-%m-%d').dt.date

order_data['Weekly']= pd.to_datetime(order_data['order_delivered_customer_date'], format='%Y-%m-%d').dt.week

order_data['yearly']=pd.to_datetime(order_data['order_delivered_customer_date'])\
                    .dt.to_period('M')
order_data['yearly']= order_data['yearly'].astype(str)

#finding different ways of delivered and estimated times

order_data['diff_days']=order_data['delivered_time']-order_data['estimate_time']
order_data['diff_days']=order_data['diff_days'].dt.days

plt.figure(figsize=(20,10))
s.lineplot(x='Weekly', y='diff_days',data=order_data, color='coral',linewidth=5,markers=True,dashes=False,estimator='mean')
plt.xlabel('Weeks',size=14)
plt.ylabel('Difference Days',size=14)
plt.title('Average Difference Days per Week', size=15, weight='bold')
plt.show()

# customer top 10 product 

order_item_data=pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_order_items_dataset.csv')
product_data = pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_products_dataset.csv')

#Merge Data 

total_orders = pd.merge(order_data, order_item_data)
product_orders=pd.merge(total_orders, product_data, on='product_id')
product_orders.info()

#shorten product_id

len(product_orders['product_id'].unique())
len(product_orders['product_id'].str[-8:].unique())
product_orders['productid_shorten']=product_orders['product_id'].str[-8:]

#top 10 product graph

plt.figure(figsize=(20,10))
s.countplot(x='product_category_name',data=product_orders, palette='gist_earth',order=product_orders['product_category_name'].value_counts()[:10].sort_values().index).set_title('Top 10 Products', fontsize=20, weight='bold')
plt.show()

# top 10 seller 

sellers_data = pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_sellers_dataset.csv')

#merge product data

sellers_product = pd.merge(product_orders, sellers_data, on='seller_id')
sellers_product.info()

len(sellers_product['seller_id'].unique())

sellers_product['sellerid_shorten']=sellers_product['seller_id'].str[-6:]

# top 10 sellers

plt.figure(figsize=(20,10))
sellers_product['sellerid_shorten'].value_counts()[:10].plot.pie(autopct='%1.1f%%',
        shadow=True, startangle=90, cmap='tab20')
plt.title("Top 10 Seller",size=14, weight='bold')

# payments

payment_data = pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_order_payments_dataset.csv')

#merge with sellers_product

payments = pd.merge(sellers_product, payment_data, on ='order_id')
payments.info()

# drop some unused columns
payments = payments.drop(columns=['product_name_lenght','product_description_lenght',
                                 'product_photos_qty','product_weight_g','product_length_cm',
                                 'product_height_cm','product_width_cm'])
price_details= payments.groupby(['order_id','price','product_category_name',
                                 'yearly','Weekly'])[['freight_value','payment_value']].sum().reset_index()

price_details['total_order_value'] = price_details['price'] + price_details['freight_value']

price_details['gross_profit'] = price_details['payment_value']- price_details['total_order_value']
price_details['profit_margin'] = price_details['gross_profit']/price_details['payment_value']
price_details['profit_margin'] = price_details['profit_margin'].astype('int64')

plt.figure(figsize=(25,15))

s.lineplot(x='yearly',y='gross_profit',
             data=price_details[price_details['product_category_name']\
             =='cama_mesa_banho'], label='bed_bath_table',color="green")
s.lineplot(x='yearly', y='gross_profit',
             data=price_details[price_details['product_category_name']\
             =='beleza_saude'], label='beauty_health', color="blue")
s.lineplot(x='yearly', y='gross_profit',
             data=price_details[price_details['product_category_name']\
             =='esporte_lazer'], label='sports_leisure', color="red")
s.lineplot(x='yearly', y='gross_profit',
             data=price_details[price_details['product_category_name']\
             =='moveis_decoracao'], label='home_decoration', color="orange")
s.lineplot(x='yearly', y='gross_profit',
             data=price_details[price_details['product_category_name']\
             =='informatica_acessorios'], label='Informatic_accessories', color="purple")
plt.title("Gross Profit of Top 5 Products (2016-2018)",fontweight='bold')

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
