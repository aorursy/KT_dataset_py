# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# read order data
olist_order = pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_orders_dataset.csv')
# read order payment data
olist_payment = pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_order_payments_dataset.csv')
# read order item data
olist_item_order = pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_order_items_dataset.csv')
# read customer data
olist_customer = pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_customers_dataset.csv')
# read products data
olist_products = pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_products_dataset.csv')
# read products category translation data
olist_category = pd.read_csv('/kaggle/input/brazilian-ecommerce/product_category_name_translation.csv')
# read review data
olist_reviews = pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_order_reviews_dataset.csv')
# read sellers data
olist_sellers = pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_sellers_dataset.csv')
# read geolocation data
olist_geolocation = pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_geolocation_dataset.csv')

# order information

print(olist_order.head())

print("# of records",olist_order.shape)
print("# of unique orders",olist_order.order_id.nunique())
print("# of unique customers",olist_order.customer_id.nunique())
# explore payment data 

print(olist_payment.head())

print("# records",olist_payment.shape)
print("# unique orders",olist_payment.order_id.nunique())

# get one order id where payment was made through more than one method
print(olist_payment[olist_payment['payment_sequential'] > 1].head())

# look through one such order id 
print(olist_payment[olist_payment['order_id'] == "5cfd514482e22bc992e7693f0e3e8df7"])

print(" different payemnt types",olist_payment.payment_type.value_counts())


# explore order item data

print(olist_item_order.head())

print("# records",olist_item_order.shape)
print("# unique orders",olist_item_order.order_id.nunique())
print("# unique products",olist_item_order.product_id.nunique())

# distribution of price
print(olist_item_order.price.describe())
print(olist_item_order.freight_value.describe())
# explore customer data

print(olist_customer.head())
print("# records",olist_customer.shape)
print("# unique cusotmer id",olist_customer.customer_id.nunique())
print("# unique cusotmer unique id",olist_customer.customer_unique_id.nunique())
# explore products data

print(olist_products.head())

print("# records",olist_products.shape)
print("# unique products",olist_products.product_id.nunique())

print("# unique category",olist_products.product_category_name.nunique())

# category name translation from spanish to english
olist_category.head()
# explore reviews data
print(olist_reviews.head())
print(olist_reviews.shape)

print(olist_reviews.order_id.nunique())

# group review by order id

review_grp = olist_reviews.groupby('order_id').agg({'review_score':'mean'}).reset_index()


# explore sellers data

olist_sellers.head()

# explore geolocation data

olist_geolocation.tail()
# merge order and customer demo data

df_process_v1 = olist_order.merge(olist_customer,on = 'customer_id', how = 'inner')

# merge order item information with order information, this will bring data at order_id - item_id level

df_process_v2 = olist_item_order.merge(df_process_v1,on = 'order_id',how = 'left')

# merge product data with above trasnaction data

#check shape
print(df_process_v2.shape)
df_process_v2 = df_process_v2.merge(olist_products,on = 'product_id',how = 'inner')

# check shape
print(df_process_v2.shape)

# merge english names of categories
df_transaction_v1 = df_process_v2.merge(olist_category, on = 'product_category_name',how = 'inner')
# check shape
print(df_transaction_v1.shape)

# merge review rating

df_transaction_v1 = df_transaction_v1.merge(review_grp, on = 'order_id', how = 'left')

# df_transaction_v1 = df_transaction_v1.merge(review_grp, on = 'order_id', how = 'inner')

# review_grp.order_id.nunique()
df_transaction_v1.shape
df_transaction_v1['product_height_cm'].fillna(df_transaction_v1['product_height_cm'].mean(),inplace = True)
df_transaction_v1['product_weight_g'].fillna(df_transaction_v1['product_weight_g'].mean(),inplace = True)
df_transaction_v1['product_width_cm'].fillna(df_transaction_v1['product_width_cm'].mean(),inplace = True)
df_transaction_v1['product_length_cm'].fillna(df_transaction_v1['product_length_cm'].mean(),inplace = True)
df_transaction_v1['product_height_cm'].fillna(df_transaction_v1['product_height_cm'].mean(),inplace = True)


df_transaction_v1.isna().sum()
#bring in the payment information 

#filtering out payment type as voucher separately and rest as others
olist_payment.loc[olist_payment.payment_type.isin(['voucher']),'payment_type_new'] = 'voucher'
olist_payment['payment_type_new'].fillna('other',inplace = True)

olist_payment_grp = olist_payment.groupby(['order_id','payment_type_new']).agg({'payment_value':'sum'}).reset_index()

# pivot up data to get voucher, non voucher revenue corresponding to each order_id
olist_payment_grp_pvt = olist_payment_grp.pivot(index = 'order_id',columns='payment_type_new',values = 'payment_value').reset_index()
olist_payment_grp_pvt.fillna(0,inplace = True)

#merge this information with transactional data
df_transaction_v2 = df_transaction_v1.merge(olist_payment_grp_pvt[['order_id','other','voucher']],on = 'order_id', how = 'left')
df_transaction_v2.head()
df_transaction_v1.order_purchase_timestamp.min()
df_transaction_v1.shape
df_transaction_v2.to_csv('olist_merged_transaction.csv',index=False)
#qc check if revenue mathces

df_v1 = df_process_v2.groupby(['order_id']).agg({'price':'sum','freight_value':'sum'}).reset_index()
df_v1['total_revenue_frm_item'] = df_v1['price'] + df_v1['freight_value']
df_v1['total_revenue_frm_item'] = df_v1['total_revenue_frm_item'].round(2)

# olist_payment_grp_pvt['total_revenue'] = olist_payment_grp_pvt['other'] + olist_payment_grp_pvt['voucher']
df_v2 = df_v1.merge(olist_payment_grp_pvt,on = 'order_id')
df_v2['total_revenue_frm_pymnt'] = df_v2['other'] + df_v2['voucher']
df_v2['total_revenue_frm_pymnt'] = df_v2['total_revenue_frm_pymnt'].round(2)
# there is mismatch in sum, probably due to floating point error

df_v2.total_revenue_frm_item
df_v2['total_revenue_frm_pymnt']
import datetime as dt
from datetime import timedelta
df_transaction_v2['order_purchase_timestamp'] = pd.to_datetime(df_transaction_v2['order_purchase_timestamp'])
df_transaction_v2['order_year'] = df_transaction_v2['order_purchase_timestamp'].dt.year
df_transaction_v2['order_month'] = df_transaction_v2['order_purchase_timestamp'].dt.month

# filter data for 3 months Aug'17 - Oct'17

filter_transaction_v1 = df_transaction_v2[(df_transaction_v2['order_month'].isin([8,9,10])) & (df_transaction_v2['order_year'].isin([2017]))]
# sample customers from transaction data from the filtered period - customers who were sent offers and reedemed those offers

#set seed
np.random.seed(seed = 42)

sample_cust = pd.Series(filter_transaction_v1.customer_unique_id.unique()).sample(frac = 0.25,random_state = 42)
offer_sent = np.random.randint(1,11,len(sample_cust))

offer_sent_redeem = pd.DataFrame({'customer_unique_id':sample_cust,
                                 'offer_id':offer_sent}).reset_index(drop = True)

offer_sent_redeem['offer_redeem'] = 1

# sample customers from full transaction data  - get customers who were sent offers but did not reedem


sample_cust = pd.Series(df_transaction_v2.customer_unique_id.unique()).sample(n = offer_sent_redeem.shape[0],random_state = 42)
offer_sent = np.random.randint(1,11,len(sample_cust))

offer_sent_no_redeem = pd.DataFrame({'customer_unique_id':sample_cust,
                                 'offer_id':offer_sent}).reset_index(drop=True)

offer_sent_no_redeem['offer_redeem'] = None
# concat offer data 

offer_sent = pd.concat([offer_sent_redeem,offer_sent_no_redeem],axis = 0)
offer_sent.drop_duplicates(['customer_unique_id'],keep = 'first',inplace = True)
offer_sent['offer_redeem'].fillna(0,inplace = True)

# add offer sent data - assume same date for all

offer_sent['offer_sent_date'] = "07/31/2017"
offer_sent['offer_sent_date'] = pd.to_datetime(offer_sent['offer_sent_date'], format='%m/%d/%Y')
offer_sent['validity_days'] = 90
offer_sent['offer_validity_date'] = offer_sent['offer_sent_date'] + timedelta(days = 90)

# prepare modeling data

#merge transaction and campaign data

model_data_v1 = df_transaction_v2.merge(offer_sent,on = 'customer_unique_id',how = 'inner')
model_data_v1 = model_data_v1[(model_data_v1['order_purchase_timestamp']<= model_data_v1['offer_validity_date']) & (model_data_v1['order_purchase_timestamp'] >= model_data_v1['offer_sent_date'])]

customer_revenue = model_data_v1.groupby(['customer_unique_id']).agg({'price':'sum','freight_value':'sum'}).reset_index()
customer_revenue['revenue'] = customer_revenue['price'] + customer_revenue['freight_value']

model_data_v1 = model_data_v1.merge(customer_revenue[['customer_unique_id','revenue']],on = 'customer_unique_id',how = 'inner')

model_data_v1.drop(['offer_sent_date','offer_validity_date','validity_days','other','voucher'],axis = 1,inplace = True)
model_data_v1.columns
# control group samples
model_data_v2 = filter_transaction_v1[~filter_transaction_v1.customer_unique_id.isin(model_data_v1.customer_unique_id)]
model_data_v2.loc[:,'offer_id'] = 0

customer_revenue = model_data_v2.groupby(['customer_unique_id']).agg({'price':'sum','freight_value':'sum'}).reset_index()
customer_revenue['revenue'] = customer_revenue['price'] + customer_revenue['freight_value']

model_data_v2 = model_data_v2.merge(customer_revenue[['customer_unique_id','revenue']],on = 'customer_unique_id',how = 'inner')
model_data_v2.drop(['other','voucher'],axis = 1,inplace = True)
model_data = pd.concat([model_data_v1,model_data_v2])
# remove columns from model data
model_data.drop(['order_id', 'order_item_id', 'product_id', 'seller_id',
       'shipping_limit_date', 'price', 'freight_value', 'customer_id','order_purchase_timestamp', 'order_approved_at',
       'order_delivered_carrier_date', 'order_delivered_customer_date',
       'order_estimated_delivery_date', 'customer_unique_id'],axis = 1,inplace = True)
model_data.head()

model_data.to_csv('offer_modeling_data_v1.csv',index = False)
product_cat = list(model_data.product_category_name_english.unique())
offer_ids = list(model_data.offer_id.unique())
offer_ids.remove(0)
offer_list = []
for offer_id in offer_ids:
    offer_dict = {"offer_id": int(offer_id),
                  "product_category": product_cat,
                  "budget" : int(np.random.randint(1000,5000)),
                 "reward": int(np.random.randint(4,11))}
    offer_list.append(offer_dict)
    
campaign_config_file = {}

campaign_config_file['offer_ids'] = offer_list
type(offer_id)
# write campaing config file to json
import json

with open('campaign_config.json','w') as fp:
    json.dump(campaign_config_file,fp)