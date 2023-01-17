import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

from matplotlib import pyplot as plt

import pandas_profiling as pp

import folium as folium

import seaborn as sns

import pandas as pd 

import numpy as np 



print('Environment Setup Complete')
file_path = '../input/insidesherpa-anz-dataset-task1/ANZ synthesised transaction dataset.xlsx'

file = pd.read_excel(file_path)

print('Data Set Ready')
file.describe(include='all', datetime_is_numeric=True)
dataset_profile = pp.ProfileReport(file)

dataset_profile.to_file('profile_report_output.html')
file_null_values = file.isna().sum()

file_null_values
# Columns with large amounts of null values are removed from the data set

file[['extraction_date','extraction_time']] = file['extraction'].str.split(pat='T', expand=True)

file.drop(columns=['bpay_biller_code','merchant_code','extraction'], inplace=True)



file['extraction_date'] = pd.to_datetime(file['extraction_date'])

file['extraction_time'] = file['extraction_time'].astype('datetime64')

file['extraction_time'] = pd.to_datetime(file['extraction_time'], format= '%H:%M:%S').dt.time
avg_txn_amount = file['amount'].mean()

avg_txn_amount_per_month = pd.DataFrame(file.groupby(file['date'].dt.strftime('%b-%Y'))['amount'].mean())



avg_txn_amount_per_month.plot(kind='barh', legend=False, title='Average Transaction Amount per Month')

plt.xlabel('Average Transaction Amount')

plt.ylabel('Month')

plt.text(3,2,'182.045 AUD')

plt.text(3,1,'196.427 AUD')

plt.text(3,0,'185.121 AUD')

plt.savefig('Average Transaction Amount per Month.png')
txn_count_per_date = pd.DataFrame(file.groupby('date',as_index=False)['transaction_id'].nunique())

avg_txn_per_month = pd.DataFrame(txn_count_per_date.groupby(txn_count_per_date['date'].dt.strftime('%b-%Y'))['transaction_id'].mean())



avg_txn_per_month.plot(kind='barh', legend=False, title='Number of Average Transactions per Month')

plt.xlabel('Number of Transactions')

plt.ylabel('Month')

plt.text(3,2,'134 Transactions')

plt.text(3,1,'132 Transactions')

plt.text(3,0,'131 Transactions')

plt.savefig('Number of Average Transactions per Month.png')
txn_types = pd.DataFrame(file.groupby('txn_description')['transaction_id'].nunique())



txn_types.plot(kind='bar',legend=False, rot=45, figsize=(15,4), title='Number of Transaction Types\nFrom August 2018 to October 2018')

plt.xlabel('Transaction Types')

plt.ylabel('Number of Transactions')

plt.savefig('Number of Transaction Types.png')
txn_mvmt = pd.DataFrame(file.groupby('movement')['transaction_id'].nunique())



txn_mvmt.plot(kind='barh',legend=False, title='Credit Transactions VS Debit Transactions')

plt.xlabel('Number of Transactions')

plt.ylabel(' ')

plt.savefig('Credit Transactions VS Debit Transactions.png')
age_dist = pd.DataFrame(file.groupby('age')['transaction_id'].nunique())



age_dist.plot(kind='bar', figsize=(20,5), legend=False, rot=0, title='Number of Customer Transactions by Age')

plt.ylabel('Number of Transactions')

plt.xlabel('Customer Age')

plt.savefig('Number of Customer Transactions by Age.png')
state_dist = pd.DataFrame(file.groupby('merchant_state', as_index=False)['transaction_id'].nunique())

state_dist['merchant_state'] = state_dist['merchant_state'].replace({'ACT':'Australian Capital Territory',

                                      'NSW':'New South Wales',

                                      'NT':'Northern Territory',

                                      'QLD':'Queensland',

                                      'SA':'South Australia',

                                      'TAS':'Tasmania',

                                      'VIC':'Victoria',

                                      'WA':'Western Australia'})





aus_map = '../input/australia-state-map/australia state map.json'

state_dist_map = folium.Map(location=[-25.2744, 133.7751], zoom_start=5)



state_dist_map.choropleth(geo_data=aus_map, data=state_dist,

             columns=['merchant_state','transaction_id'],

             key_on='feature.properties.STATE_NAME',

             fill_color='GnBu', fill_opacity=0.7, line_opacity=0.2,

             legend_name='Number of Transactions')



print(state_dist)

state_dist_map
extraction_volume_day_data = pd.DataFrame(file.groupby('extraction_date', as_index=False)['extraction_time'].nunique())



sns.set_style('darkgrid')

plt.figure(figsize=(25,6))



ax = sns.lineplot(x=extraction_volume_day_data['extraction_date'], y=extraction_volume_day_data['extraction_time'], marker='o')

ax.set(xticks=extraction_volume_day_data.extraction_date.values)



plt.xticks(rotation=50)

plt.xlabel('Date')

plt.ylabel('Number of Transactions')

plt.title('Customer Transaction Volume per Day\nAugust 2018 to October 2018')

plt.savefig('Customer Transaction Volume per Day.png')
avg_spending_day_data = pd.DataFrame(file.groupby('extraction_date', as_index=False)['amount'].mean())



sns.set_style('darkgrid')

plt.figure(figsize=(25,6))



ax = sns.lineplot(x=avg_spending_day_data['extraction_date'], y=avg_spending_day_data['amount'], marker='o')

ax.set(xticks=avg_spending_day_data.extraction_date.values)



plt.xticks(rotation=50)

plt.xlabel('Date')

plt.ylabel('Average Transaction Value')

plt.title('Average Customer Transaction Value per Day\nAugust 2018 to October 2018')

plt.savefig('Average Customer Transaction Value per Day.png')
ext_time_data = pd.DataFrame(file.groupby('extraction_time', as_index=False)['transaction_id'].nunique())

ext_time_data['extraction_time'] = ext_time_data['extraction_time'].astype(str)

ext_time_data[['hour','minute','second']] = ext_time_data['extraction_time'].str.split(pat=':', expand=True)



txn_time_data = pd.DataFrame(ext_time_data.groupby('hour', as_index=False)['transaction_id'].sum())

txn_time_data.plot(kind='bar', legend=False, rot=0, figsize=(24,4))



plt.xlabel('Hour')

plt.ylabel('Number of Transactions')

plt.title('Number of Customer Transactions within a 24 Hour Period')

plt.savefig('Number of Customer Transactions within a 24 Hour Period.png')