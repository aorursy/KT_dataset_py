import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
file_name = '/kaggle/input/kpmg-virtual-internship/KPMG_VI_New_raw_data_update_final.xlsx'

print(pd.ExcelFile(file_name).sheet_names)
cust_demo_df = pd.read_excel(file_name, header=1, sheet_name='CustomerDemographic')

cust_demo_df.head()
cust_demo_df.shape
cust_demo_df[cust_demo_df.duplicated()].sum()
print('customer_id blanks:', pd.isna(cust_demo_df['customer_id']).sum())
pd.notna(cust_demo_df['customer_id'].unique()).sum()
print('first_name blanks:', pd.isna(cust_demo_df['first_name']).sum())
print('gender:', cust_demo_df['gender'].unique())

print('blanks:', pd.isna(cust_demo_df['gender']).sum())

plt.hist(cust_demo_df['gender'][pd.notna(cust_demo_df['gender'])])
cust_demo_df['past_3_years_bike_related_purchases'].describe()
print('past_3_years_bike_related_purchases blanks:', pd.isna(cust_demo_df['past_3_years_bike_related_purchases']).sum())
cust_demo_df['DOB'].describe(datetime_is_numeric=True)
print('DOB blanks:', pd.isna(cust_demo_df['DOB']).sum())
plt.scatter([d.year for d in cust_demo_df['DOB']], [d.month for d in cust_demo_df['DOB']])
from datetime import datetime

cust_demo_df['age'] = (datetime.now() - cust_demo_df['DOB']) // 365
cust_demo_df['age'].describe()
plt.scatter([d.year for d in cust_demo_df['DOB']], cust_demo_df['age'].dt.days)
print('job_title:', cust_demo_df['job_title'].unique())

print('blanks:', pd.isna(cust_demo_df['job_title']).sum())
print('job_industry_category:', cust_demo_df['job_industry_category'].unique())

print('blanks:', pd.isna(cust_demo_df['job_industry_category']).sum())
print('wealth_segment:', cust_demo_df['wealth_segment'].unique())

print('blanks:', pd.isna(cust_demo_df['wealth_segment']).sum())

plt.hist(cust_demo_df['wealth_segment'][pd.notna(cust_demo_df['wealth_segment'])])
print('deceased_indicator:', cust_demo_df['deceased_indicator'].unique())

print('blanks:', pd.isna(cust_demo_df['deceased_indicator']).sum())

plt.hist(cust_demo_df['deceased_indicator'][pd.notna(cust_demo_df['deceased_indicator'])])
cust_demo_df['deceased_indicator'][cust_demo_df['deceased_indicator'] == 'Y'].count()
print('owns_car:', cust_demo_df['owns_car'].unique())

print('blanks:', pd.isna(cust_demo_df['owns_car']).sum())

plt.hist(cust_demo_df['owns_car'][pd.notna(cust_demo_df['owns_car'])])
cust_demo_df['tenure'].describe()
print('tenure blanks:', pd.isna(cust_demo_df['tenure']).sum())
cust_addr_df = pd.read_excel(file_name, header=1, sheet_name='CustomerAddress')

cust_addr_df.head()
cust_addr_df.shape
cust_addr_df[cust_addr_df.duplicated()].sum()
print('customer_id blanks:', pd.isna(cust_addr_df['customer_id']).sum())
pd.notna(cust_addr_df['customer_id'].unique()).sum()
print('customer_ids not in demographics dataset:', sum([(1 if (i not in cust_demo_df['customer_id']) else 0) for i in cust_addr_df['customer_id']]))
print('address blanks:', pd.isna(cust_addr_df['address']).sum())
print('postcode blanks:', pd.isna(cust_addr_df['postcode']).sum())
print('state:', cust_addr_df['state'].unique())

print('blanks:', pd.isna(cust_addr_df['state']).sum())

plt.hist(cust_addr_df['state'][pd.notna(cust_addr_df['state'])])
print('country:', cust_addr_df['country'].unique())

print('blanks:', pd.isna(cust_addr_df['country']).sum())
cust_addr_df['property_valuation'].describe()
print('property_valuation blanks:', pd.isna(cust_addr_df['property_valuation']).sum())

plt.hist(cust_addr_df['property_valuation'], bins=10)
txns_df = pd.read_excel(file_name, header=1, sheet_name='Transactions')

txns_df.head()
txns_df.shape
txns_df[txns_df.duplicated()].sum()
txns_df['list_price'].describe()
print('list_price blanks:', pd.isna(txns_df['list_price']).sum())
txns_df['standard_cost'].describe()
print('standard_cost blanks:', pd.isna(txns_df['standard_cost']).sum())
txns_df['profit'] = txns_df['list_price'] - txns_df['standard_cost']

txns_df['profit'].describe()
print('transaction_id blanks:', pd.isna(txns_df['transaction_id']).sum())
print('product_id blanks:', pd.isna(txns_df['product_id']).sum())
print('customer_id blanks:', pd.isna(txns_df['customer_id']).sum())
pd.notna(txns_df['customer_id'].unique()).sum()
print('customer_ids not in demographics dataset:', sum([(1 if (i not in cust_demo_df['customer_id']) else 0) for i in cust_addr_df['customer_id']]))

print('customer_ids not in addresses dataset:', sum([(1 if (i not in cust_demo_df['customer_id']) else 0) for i in txns_df['customer_id']]))
txns_df['transaction_date'].describe(datetime_is_numeric=True)
print('transaction_date blanks:', pd.isna(txns_df['transaction_date']).sum())
print('online_order:', txns_df['online_order'].unique())

print('blanks:', pd.isna(txns_df['online_order']).sum())

plt.hist(txns_df['online_order'][pd.notna(txns_df['online_order'])])
print('order_status:', txns_df['order_status'].unique())

print('blanks:', pd.isna(txns_df['order_status']).sum())

plt.hist(txns_df['order_status'][pd.notna(txns_df['order_status'])])
txns_df['order_status'][txns_df['order_status'] == 'Cancelled'].count()
print('brand:', txns_df['brand'].unique())

print('blanks:', pd.isna(txns_df['brand']).sum())

plt.hist(txns_df['brand'][pd.notna(txns_df['brand'])])
print('product_line:', txns_df['product_line'].unique())

print('blanks:', pd.isna(txns_df['product_line']).sum())

plt.hist(txns_df['product_line'][pd.notna(txns_df['product_line'])])
print('product_class:', txns_df['product_class'].unique())

print('blanks:', pd.isna(txns_df['product_class']).sum())

plt.hist(txns_df['product_class'][pd.notna(txns_df['product_class'])])
print('product_size:', txns_df['product_size'].unique())

print('blanks:', pd.isna(txns_df['product_size']).sum())

plt.hist(txns_df['product_size'][pd.notna(txns_df['product_size'])])
print('product_first_sold_date blanks:', pd.isna(txns_df['product_first_sold_date']).sum())
plt.scatter(txns_df['brand'][pd.notna(txns_df['brand'])], txns_df['profit'][pd.notna(txns_df['brand'])])