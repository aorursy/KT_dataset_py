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
# importing libraries

import matplotlib.pyplot as plt
df=pd.read_csv("/kaggle/input/anz-synthesised-transaction-dataset/anz.csv")
df.head()
df.shape
df.info()
df.bpay_biller_code.value_counts()
df.drop(['bpay_biller_code','merchant_code'],axis=1,inplace=True)
df.isna().sum()
import missingno as msno
msno.matrix(df)
msno.matrix(df[['card_present_flag','merchant_id','merchant_suburb','merchant_state','merchant_long_lat']])
df.dropna(inplace=True)
msno.matrix(df)
print(f"we are left with {df.shape[0]} rows and {df.shape[1]} columns")
df.status.value_counts()
plt.plot(df.status)
df.drop(['status'],axis=1,inplace=True)
df['card_present_flag'].value_counts()
plt.hist(df['card_present_flag'])
df['account'].value_counts()
df.drop(['account'],axis=1,inplace=True)
df['currency'].value_counts()
df.drop(['currency'],axis=1,inplace=True)
df.long_lat.value_counts()
# new data frame with split value columns 

co_ordinates = df["long_lat"].str.split(" ", n = 1, expand = True) 

  

# making separate first name column from new data frame 

df["longitude"]= co_ordinates[0] 

  

# making separate last name column from new data frame 

df["latitude"]= co_ordinates[1] 
df['latitude'] = df['latitude'].astype(float)

df['longitude'] = df['longitude'].astype(float)
from mpl_toolkits.basemap import Basemap

fig = plt.figure(figsize=(12,9))

m = Basemap(projection='mill',

           llcrnrlat = -90,

           urcrnrlat = 90,

           llcrnrlon = -180,

           urcrnrlon = 180,

           resolution = 'c')

m.drawcoastlines()

m.drawparallels(np.arange(-90,90,10),labels=[True,False,False,False])

m.drawmeridians(np.arange(-180,180,30),labels=[0,0,0,1])

sites_lat_y = df['latitude'].tolist()

sites_lon_x = df['longitude'].tolist()

m.scatter(sites_lon_x,sites_lat_y,latlon=True)

plt.title('Basemap', fontsize=20)

plt.show()
df.drop(['long_lat'],axis=1,inplace=True)
df['txn_description'].value_counts()
plt.hist(df['txn_description'])
cleanup_txn = {"txn_description":{"POS": 1, "SALES-POS":0}}

df.replace(cleanup_txn, inplace=True)
df['merchant_id'].value_counts()
df.drop(['merchant_id'],axis=1,inplace=True)
df.first_name.value_counts()
df.balance.value_counts()
plt.hist(df.balance)
df=df[df['balance']<100000]
plt.hist(df.balance)
df.date.value_counts()
type(df.date[0])


df['date']= pd.to_datetime(df['date']) 

df['year'] = pd.DatetimeIndex(df['date']).year

df['month'] = pd.DatetimeIndex(df['date']).month
df.year.value_counts()
df.drop(['date','year'],axis=1,inplace=True)
df.month.value_counts()
df['gender'].value_counts()
cleanup_gender = {"gender":{"M": 1, "F":0}}

df.replace(cleanup_gender, inplace=True)
plt.hist(df.age)
df['merchant_suburb'].nunique()
df['merchant_state'].value_counts()
df.extraction.value_counts()
df.drop(['extraction'],axis=1,inplace=True)
plt.hist(df.amount)
df.amount.max()
perc =[.80,.90,.99] 

df.amount.describe(percentiles=perc)
df=df[df['amount']<378]
df['transaction_id'].value_counts()
df.drop(['transaction_id'],axis=1,inplace=True)
df['country'].value_counts()
df.drop(['country'],axis=1,inplace=True)
df['customer_id'].value_counts()
df.drop(['customer_id'],axis=1,inplace=True)
df.columns
df['merchant_long_lat'].value_counts()
df.drop(['merchant_long_lat'],axis=1,inplace=True)
df['movement'].value_counts()
df.drop(['movement'],axis=1,inplace=True)
df