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
import seaborn as sns

import matplotlib.pyplot as plt
customerAddress = pd.read_excel('../input/kpmg-data/KPMG_VI_New_raw_data_update_final.xlsx', sheet_name = 'CustomerAddress', skiprows = [0])
customerDemographic = pd.read_excel('../input/kpmg-data/KPMG_VI_New_raw_data_update_final.xlsx', sheet_name = 'CustomerDemographic', skiprows = [0])
newCustomerList = pd.read_excel('../input/kpmg-data/KPMG_VI_New_raw_data_update_final.xlsx', sheet_name = 'NewCustomerList', skiprows = [0])
transactions = pd.read_excel('../input/kpmg-data/KPMG_VI_New_raw_data_update_final.xlsx', sheet_name = 'Transactions', skiprows = [0])
customerAddress.customer_id.max()
newCustomerList['customer_id'] = 4004 + newCustomerList.index
newCustomerAddress = newCustomerList[['customer_id','address', 'postcode', 'state', 'country', 'property_valuation']]

newCustomerAddress
print('Shape of customerAddress before concat :', customerAddress.shape)

customerAddress = pd.concat([customerAddress, newCustomerAddress], axis=0)

print('Shape of customerAddress after concat :', customerAddress.shape)
newCustomerList.columns
newCustomerDemographic = newCustomerList[['customer_id','first_name', 'last_name', 'gender',

       'past_3_years_bike_related_purchases', 'DOB', 'job_title',

       'job_industry_category', 'wealth_segment', 'deceased_indicator', 'owns_car', 'tenure']]

newCustomerDemographic
print('Shape of customerDemographic before concat :', customerDemographic.shape)

customerDemographic = pd.concat([customerDemographic, newCustomerDemographic], axis=0)

print('Shape of customerDemographic after concat :', customerDemographic.shape)
customerAddress.info()
customerAddress.isnull().sum()
customerAddress.boxplot()
customerDemographic.isnull().sum()
customerDemographic = customerDemographic.drop(labels = ['default'],axis = 1)
customerDemographic.gender.value_counts()
customerDemographic.gender = customerDemographic.gender.replace(to_replace=['Female', 'Femal', 'F'], value= 'F')

customerDemographic.gender = customerDemographic.gender.replace(to_replace=['Male'], value= 'M')

customerDemographic.gender = customerDemographic.gender.astype('category')
customerDemographic.deceased_indicator = customerDemographic.deceased_indicator.replace(to_replace=['N'], value= 0)

customerDemographic.deceased_indicator = customerDemographic.deceased_indicator.replace(to_replace=['Y'], value= 1)



customerDemographic.deceased_indicator = customerDemographic.deceased_indicator.astype('bool')
customerDemographic.owns_car = customerDemographic.owns_car.replace(to_replace=['No'], value= 0)

customerDemographic.owns_car = customerDemographic.owns_car.replace(to_replace=['Yes'], value= 1)



customerDemographic.owns_car = customerDemographic.owns_car.astype('bool')
customerDemographic = customerDemographic.rename(columns={"DOB": "birth_date"})
customerDemographic.birth_date.min()
customerDemographic.loc[  customerDemographic.birth_date == customerDemographic.birth_date.min(), 'birth_date' ] = np.nan
customerDemographic.loc[  customerDemographic.birth_date == customerDemographic.birth_date.min()]
customerDemographic.info()
print( customerDemographic.skew() )
transactions.info()
transactions.product_first_sold_date = pd.to_datetime('1899-12-30') + pd.to_timedelta(transactions.product_first_sold_date,'D')
transactions.online_order = transactions.online_order.replace(to_replace=['No'], value= 0)

transactions.online_order = transactions.online_order.replace(to_replace=['Yes'], value= 1)



transactions.online_order = transactions.online_order.astype('bool')
transactions.info()
transactions.isnull().sum()
transactions.shape
197 / 20000 * 100
transactions = transactions.dropna( how = 'any')
print(transactions.skew())
transactions.standard_cost.plot.box()
transactions.loc[transactions.standard_cost == transactions.standard_cost.max(), ].shape
customers = customerDemographic.merge( customerAddress, on = 'customer_id')
transactions