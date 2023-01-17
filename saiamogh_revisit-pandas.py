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
# Reading data

df = pd.read_csv('/kaggle/input/retail-data-customer-summary-learn-pandas-basics/Retail_Data_Customers_Summary.csv')


df.head(5)
print(df.shape)



print(df.size)



# Finding the missing values in columns



df.isnull().sum(axis=0)
# filling missing values with 0



df.fillna(value=0,inplace = True)
df.head()
# Total  amount & Tranx & ATV

df['Total Customer Spend'] = df.iloc[:,1:6].sum(axis=1)

df['Total Customer Tranx'] = df.iloc[:,6:-3].sum(axis=1)

df['Customer ATV'] = np.round(df['Total Customer Spend'].div(df['Total Customer Tranx']),2)

df.head()



# Customer with highers tranx value



df[df['Total Customer Tranx']  == df['Total Customer Tranx'].max() ]



# slicing 

# Top 10 Customers in 2014 and 2015

Top_10_customers_tranx_2014_2015 = pd.DataFrame()

Top_10_customers_tranx_2014_2015[['customer_id_2014','tran_amount_2014']]= df[['customer_id','tran_amount_2014']].sort_values(by = ['tran_amount_2014'],ascending=False).head(10)

Top_10_customers_tranx_2014_2015[['customer_id_2015','tran_amount_2015']]=df[['customer_id','tran_amount_2015']].sort_values(by = ['tran_amount_2015'],ascending=False).head(10).reset_index().iloc[:,1:]

Top_10_customers_tranx_2014_2015
# customers who have done tranx in 2014 but not in 2015



customer_2015 =  df.sort_values(by = ['tran_amount_2014'],ascending=False).head()

customer_2015[['customer_id','tran_amount_2014','tran_amount_2015']].sort_values(by = ['tran_amount_2015'],ascending=True).head(10)
# Date Diff

df['days'] = (pd.to_datetime(df['Latest_Transaction']) - pd.to_datetime(df['First_Transaction']))



df[df['days'] < '365 days']
# Describe



df.iloc[:,1:-4].describe().T