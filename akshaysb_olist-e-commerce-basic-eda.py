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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
df_orders=pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_orders_dataset.csv')

df_items=pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_order_items_dataset.csv')

df_customers=pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_customers_dataset.csv')

df_payments=pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_order_payments_dataset.csv')

df_products=pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_products_dataset.csv')

df_category=pd.read_csv('/kaggle/input/brazilian-ecommerce/product_category_name_translation.csv')

df_sellers=pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_sellers_dataset.csv')

df_reviews=pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_order_reviews_dataset.csv')
df_eng=pd.read_csv('../input/brazilian-ecommerce/product_category_name_translation.csv')
df = pd.merge(df_orders,df_items,how='left',on='order_id') #

df = pd.merge(df,df_customers, how='outer', on='customer_id')

df = pd.merge(df,df_payments, how='outer' , on='order_id')

df = pd.merge(df,df_products, how='outer', on='product_id')

df = pd.merge(df,df_category, how='outer', on='product_category_name')

df = pd.merge(df,df_sellers, how='outer' , on='seller_id')

df = pd.merge(df,df_reviews, how='outer' , on='order_id')
df.drop(['product_name_lenght','product_length_cm',

         'product_weight_g','product_height_cm','product_width_cm',],axis=1,inplace=True)
olist=df[df['order_status'] != 'canceled']    
olist=df[df['order_status'] != 'unavailable']
pay=pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_order_payments_dataset.csv')
pay.head()
plt.figure(figsize=(10,10))

pay['payment_type'].value_counts().plot(kind='pie',autopct='%1.1f%%')

plt.title('Distribution of modes of payment')
pay.groupby('payment_type').payment_installments.max()
pay_cr=olist[olist['payment_type']=='credit_card']
pay_cr.head()
plt.figure(figsize=(15,15))

pay_cr.groupby('customer_state').payment_installments.median().plot(kind='bar')

plt.title('State-Wise Average Installments for Transaction through Credit Card')

plt.ylabel('Avg No. of installments')
plt.figure(figsize=(15,15))

pay_cr.groupby('customer_state').payment_value.median().plot(kind='bar')

plt.title('State-Wise Average Purchase of customers')

plt.ylabel('Avg Purchase through credit card')
orders=pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_orders_dataset.csv')
orders.head()
orders=orders.sort_values('order_id',axis=0)

orders.reset_index(inplace=True)
o_items=pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_order_items_dataset.csv')

o_items.head()
price=o_items.groupby('order_id')['price','freight_value'].sum()
price.reset_index(inplace=True)
price.drop('order_id',axis=1,inplace=True)
order_details=pd.concat([orders,price],axis=1)
order_details.drop('index',axis=1,inplace=True)
plt.figure(figsize=(15,15))

sns.countplot(order_details['order_status'])

plt.title('Order Status of orders')
delivered=order_details[order_details['order_status']=='delivered']
delivered.info()
delivered.dropna(subset=['order_delivered_customer_date'],inplace=True)
delivered.head()
delivery_date=[]

for i in delivered.order_delivered_customer_date:

    if i != np.NaN:

        i=str(i)

        a=i.split(' ')[0]

        delivery_date.append(a)

from datetime import datetime

del_date=[]

for i in delivery_date:

    if i != np.NaN:

        a=datetime.strptime(i,'%Y-%m-%d')

        del_date.append(a)

delivered['actual_delivery_date']=del_date

estimated_date=[]

for i in delivered.order_estimated_delivery_date:

    if i != np.NaN:

        i=str(i)

        a=i.split(' ')[0]

        estimated_date.append(a)

est_date=[]

for i in estimated_date:

    if i != np.NaN:

        a=datetime.strptime(i,'%Y-%m-%d')

        est_date.append(a)

delivered['estimated_delivery_date']=est_date

delivered['delay']=delivered['actual_delivery_date']-delivered['estimated_delivery_date']
delivered.head()
day_delay=[]

for i in delivered['delay']:

    i=str(i)

    a=i.split(' ')[0]

    a=int(a)

    day_delay.append(a)
delivered['delay']=day_delay
delivered.head()
delivered=pd.merge(delivered,df_customers,how='inner',on='customer_id')
delivered.drop(['customer_unique_id','customer_zip_code_prefix','customer_city'],axis=1,inplace=True)
plt.figure(figsize=(15,15))

delivered.groupby('customer_state').delay.mean().plot(kind='bar')

plt.ylabel('Avg Delay')

plt.title('State-Wise Delay of Order Delivery')
late_del=delivered[delivered['delay']>0]

late_del.head()
late_del.shape
plt.figure(figsize=(15,15))

late_del['customer_state'].value_counts().plot(kind='bar')

plt.xlabel('Customer_State')

plt.ylabel('Number of delayed deliveries')

plt.title('State-Wise Delayed Deliveries')
late_del=pd.merge(late_del,df_items,how='inner',on='order_id')

late_del=pd.merge(late_del,df_products,how='inner',on='product_id')

late_del=pd.merge(late_del,df_eng,how='inner',on='product_category_name')
plt.figure(figsize=(15,15))

late_del.groupby('product_category_name_english').delay.median().plot(kind='bar')

plt.xlabel('Product categories')

plt.ylabel('No. of delays')

plt.title('Product Category-Wise Delays')
cust=pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_customers_dataset.csv')
delivered=delivered.sort_values('customer_id',axis=0)

cust=cust.sort_values('customer_id',axis=0)
delivered.head()
cust_del=pd.merge(delivered,cust,how='inner',on='customer_id')

#cust_del=delivered.join(cust.set_index('customer_id'),on='customer_id')

#cust_del.reset_index(inplace=True)
cust_del.head()
cust_del.drop(['order_purchase_timestamp','order_approved_at','order_delivered_carrier_date','order_delivered_customer_date','order_estimated_delivery_date',

         'customer_unique_id','customer_zip_code_prefix'],axis=1,inplace=True)
plt.figure(figsize=(15,15))

cust_del.customer_state_y.value_counts().plot(kind='bar')

plt.title('State-Wise Distribution of Customers')

plt.xlabel('Customer_state')

plt.ylabel('No. of Customers')
plt.figure(figsize=(15,15))

cust_del.groupby('customer_state_y').price.sum().plot(kind='bar')

plt.xlabel('Customer_state')

plt.title('State-Wise Total Sales')

plt.ylabel('Total Sales')
plt.figure(figsize=(15,15))

cust_del.groupby('customer_state_y').freight_value.mean().plot(kind='bar')

plt.xlabel('Customer_state')

plt.ylabel('Average Frieght Value')

plt.title('State-Wise Average Freight Value')
df.drop(['product_category_name','review_id','review_comment_title','review_comment_message','review_creation_date','review_answer_timestamp'],

       axis=1,inplace=True)
plt.figure(figsize=(15,15))

df['product_category_name_english'].value_counts().plot(kind='bar')

plt.ylabel('No. of items sold')

plt.title('No. of Items Sold in Each Product Category')
pd.DataFrame(df.groupby('product_category_name_english').price.sum()).sort_values(by='price',ascending=False)
plt.figure(figsize=(15,15))

df.groupby('product_category_name_english').price.sum().plot(kind='bar')

plt.ylabel('Total Revenue')

plt.title('Product Category-Wise Revenue Generated')
pd.DataFrame(df.groupby('product_category_name_english').review_score.mean()).sort_values(by='review_score',ascending=False)
plt.figure(figsize=(15,15))

state_review=pd.DataFrame(df.groupby(['customer_state']).review_score.mean()).sort_values(by=['review_score'],ascending=False).plot(kind='bar')
state_review=pd.DataFrame(df.groupby(['customer_state','product_category_name_english']).agg({'review_score':'median','price':'sum'}))#['review_score','price'].mean())
state_review.reset_index(['customer_state','product_category_name_english'],inplace=True)
state_review.head()
state_review.columns
df_health=state_review[state_review['product_category_name_english']=='health_beauty']
df_health.head()
f , (ax1, ax2) = plt.subplots(1, 2 , figsize=(20,10))

ax1.bar(df_health['customer_state'],df_health['review_score'],data=df_health)

ax1.set_xlabel('Customer_state')

ax1.set_ylabel('Average Review')

ax1.set_title('State-Wise Review for Beauty_Health Products')

ax2.bar(df_health['customer_state'],df_health['price'],data=df_health)

ax2.set_xlabel('Customer_state')

ax2.set_ylabel('Total sales')

ax2.set_title('State-Wise Sales of Beauty_Health Products')

plt.show()
df_watches=state_review[state_review['product_category_name_english']=='watches_gifts']
df_watches.head()
f , (ax1, ax2) = plt.subplots(1, 2 , figsize=(20,10))

ax1.bar(df_watches['customer_state'],df_watches['review_score'],data=df_watches)

ax1.set_xlabel('Customer_state')

ax1.set_ylabel('Average Review')

ax1.set_title('State-Wise Review for Watches and Gifts')

ax2.bar(df_watches['customer_state'],df_watches['price'],data=df_watches)

ax2.set_xlabel('Customer_state')

ax2.set_ylabel('Total Sales')

ax2.set_title('State-Wise Sales of Watches and Gifts')

plt.show()
df_bed=state_review[state_review['product_category_name_english']=='bed_bath_table']
df_bed.head()
f , (ax1, ax2) = plt.subplots(1, 2 , figsize=(20,10))

ax1.bar(df_bed['customer_state'],df_bed['review_score'],data=df_bed)

ax1.set_xlabel('Customer_state')

ax1.set_ylabel('Average Review')

ax1.set_title('State-Wise Review for Bed Bath Table')

ax2.bar(df_bed['customer_state'],df_bed['price'],data=df_bed)

ax2.set_xlabel('Customer_state')

ax2.set_ylabel('Total Sales')

ax2.set_title('State-Wise Sales of Bed Bath Tables')

plt.show()
df_sports=state_review[state_review['product_category_name_english']=='sports_leisure']
df_sports.head()
f , (ax1, ax2) = plt.subplots(1, 2 , figsize=(20,10))

ax1.bar(df_sports['customer_state'],df_sports['review_score'],data=df_sports)

ax1.set_xlabel('Customer_state')

ax1.set_ylabel('Average Review')

ax1.set_title('State-Wise Review for Sports and Leisure Products')

ax2.bar(df_sports['customer_state'],df_sports['price'],data=df_sports)

ax2.set_xlabel('Customer_state')

ax2.set_ylabel('Total Sales')

ax2.set_title('State-Wise Sales of Sports and Leisure Products')

plt.show()
df_ca=state_review[state_review['product_category_name_english']=='computers_accessories']
df_ca.head()
f , (ax1, ax2) = plt.subplots(1, 2 , figsize=(20,10))

ax1.bar(df_ca['customer_state'],df_ca['review_score'],data=df_ca)

ax1.set_xlabel('Customer_state')

ax1.set_ylabel('Average Review')

ax1.set_title('State-Wise Review for Computer Accessories')

ax2.bar(df_ca['customer_state'],df_ca['price'],data=df_ca)

ax2.set_xlabel('Customer_state')

ax2.set_ylabel('Total Sales')

ax2.set_title('State-Wise Sales of Computer Accessories')

plt.show()
df_f=state_review[state_review['product_category_name_english']=='flowers']
df_f.head()
f , (ax1, ax2) = plt.subplots(1, 2 , figsize=(20,10))

ax1.bar(df_f['customer_state'],df_f['review_score'],data=df_f)

ax1.set_xlabel('Customer_state')

ax1.set_ylabel('Average Review')

ax1.set_title('State-Wise Review for Flowers')

ax2.bar(df_f['customer_state'],df_f['price'],data=df_f)

ax2.set_xlabel('Customer_state')

ax2.set_ylabel('Total Sales')

ax2.set_title('State-Wise Sales of Flowers')

plt.show()
df_h=state_review[state_review['product_category_name_english']=='home_comfort_2']
df_h.head()
f , (ax1, ax2) = plt.subplots(1, 2 , figsize=(20,10))

ax1.bar(df_h['customer_state'],df_h['review_score'],data=df_h)

ax1.set_xlabel('Customer_state')

ax1.set_ylabel('Average Review')

ax1.set_title('State-Wise Review for House hold products')

ax2.bar(df_h['customer_state'],df_h['price'],data=df_h)

ax2.set_xlabel('Customer_state')

ax2.set_ylabel('Total Sales')

ax2.set_title('State-Wise Sales of House Hold Products')

plt.show()
df_c=state_review[state_review['product_category_name_english']=='cds_dvds_musicals']
df_c
df_cl=state_review[state_review['product_category_name_english']=='fashion_childrens_clothes']
df_cl.head()
df_ss=state_review[state_review['product_category_name_english']=='security_and_services']
df_ss.head()