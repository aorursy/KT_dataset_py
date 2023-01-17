# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)= pd 

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/summer-products-and-sales-in-ecommerce-wish/summer-products-with-rating-and-performance_2020-08.csv')
df.head()
df.isnull().sum()
df.drop(['has_urgency_banner','urgency_text','merchant_profile_picture','merchant_has_profile_picture','merchant_info_subtitle','product_picture'],axis=1, inplace=True)
df.head()
df.drop_duplicates(inplace=True)
df
df.origin_country.hist()
units_sold_by_country = df.groupby('origin_country')['units_sold'].sum()
units_sold_by_country
plt.bar(units_sold_by_country.index, height = units_sold_by_country.values)
plt.yscale('log')
df['discount']= df['retail_price']- df['price']
df['revenue']= df['retail_price']*  df['units_sold'] 
df['profit']= df['revenue']-(df['price']* df['units_sold'])
df.groupby('merchant_id')['revenue'].sum() # Total Revenue earned by merchant
df.groupby('merchant_id')['profit'].sum() # Total Profit by merchant. Some has negative values that means they had provided product with huge discount and below retail price
Loss_by_merchant = df[df['profit']<0][['merchant_id','profit']] # Find which Merchant did loss
Loss_by_merchant # total 477 products have been sold at huge discounted price
Loss_by_merchant.set_index('merchant_id', inplace=True)
Loss_by_merchant
Total_Loss_merchant_by= Loss_by_merchant.groupby('merchant_id')['profit'].sum() # Total loss by merchant
Total_Loss_merchant_by  # Total 377 Unique merchant are selling products below retail price
Total_Loss_merchant_by.sort_values().head() # Top 5 merchant loss highest
price_by= df[['price', 'discount','retail_price', 'origin_country']].groupby('origin_country').mean()
revenue_by_country = df.groupby('origin_country')['revenue'].sum()
sns.barplot(revenue_by_country.index, revenue_by_country.values)  # Country wise revenue. China is earning highest revenue to compare other countries
plt.yscale('log')
df['uses_ad_boosts'].value_counts()
sns.countplot('uses_ad_boosts', data=df)
df.groupby('uses_ad_boosts')['revenue'].sum()  # Without ad_boosts, good revenue have been achieved. 
df[df['merchant_id'].isin(Total_Loss_merchant_by)]
df['title_orig'] 
df[df['title_orig']== [a for a in df['title_orig']  if 'women' in a or 'Women' in a or 'women'.upper()]]   # It returns all rows so means all products are for women
df['shipping_option_name'].value_counts() # Customer selected Livraison Standard shipping option
df[df['shipping_is_express']==1] # Very few customer selected for express shipping
df['badges_count'].value_counts()
df[df['badge_fast_shipping']==1]  
df[df['merchant_rating']>=4.5][['merchant_id','merchant_rating']]  # Few merchant received ratings above 4.5
