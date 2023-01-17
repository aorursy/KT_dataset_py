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
# import required libraries
import pandas as pd
import numpy as np
from datetime import datetime as dt
import seaborn as sns # used for plot interactive graph. 
import matplotlib.pyplot as plt
# Importig dataset 

data = pd.read_csv('../input/online-retail-ii-uci/online_retail_II.csv')

#check first 5 rows 

data.head()
# No of records and columns 
data.shape
#check for any duplicate record first

data.duplicated().sum()
# Remove duplicate items 
data = data[~data.duplicated()]

# check no of records now 

data.shape
# check for the missing values 

total = data.isnull().sum().sort_values(ascending=False)

percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(10)

#Don't need records with Null customer id,so deleting them from the dataframe  

data.dropna(axis = 0, subset = ['Customer ID'], inplace = True)

data.shape  # so no of records are reduced now
# check the data types of variables 
data.info()
# Change the datatypes of InvoiceDate to datetime object and Customer ID to string

# Convert remaining Customer Ids to string type

data['Customer ID']= data['Customer ID'].astype(str)

data['InvoiceDate'] = pd.to_datetime(data ['InvoiceDate'])
# check the descriptive statstics of numerical variables 

data.describe()
# There are negative quantities and corresponding Invoice no is started with string C (Cancelled items)

display(data.sort_values('Quantity')[:5])
# Check and remove transactions with cancelled items.

data_new = data[~data.Invoice.str.contains('C', na=False)]

# check no of records, further reduced 

data_new.shape 
# Now there is no negative value of Quantity variable 

display(data_new.sort_values('Quantity')[:3])
# lets check which are countries with maximum no of transactions 

customer_country=data_new[['Country','Customer ID']].drop_duplicates()

customer_country.groupby(['Country'])['Customer ID'].aggregate('count').reset_index().sort_values('Customer ID', ascending=False)
# lets change the format of InvoiceDate 

import time
from datetime import datetime, date, time, timedelta

data_new['InvoiceDate'] = pd.to_datetime(data_new['InvoiceDate']).dt.date
# oldest and latest date 
print('Min:{}; Max:{}'.format(min(data_new.InvoiceDate),
max(data_new.InvoiceDate)))   
# Let's create a hypothetical snapshot_day data as if we're doing analysis recently.

snapshot_date = max(data_new.InvoiceDate) + timedelta(days=1)
print(snapshot_date)
#Create new columns called TotalSum column = Quantity x UnitPrice.

data_new['TotalSum'] = data_new['Quantity'] * data_new['Price']
data_new.head() # to check new column 
# Aggregate data on a customer level
datamart = data_new.groupby(['Customer ID']).agg({'InvoiceDate': lambda x: (snapshot_date - x.max()).days,'Invoice': 'count','TotalSum': 'sum'})
# Rename columns for easier interpretation # term used as recency, frequency and Monetary
datamart.rename(columns = {'InvoiceDate': 'Recency','Invoice': 'Frequency','TotalSum': 'MonetaryValue'}, inplace=True)
# check the first few rows
datamart.head()
 
#Recency with a decreasing range of 4 through 1

r_labels = range(4, 0, -1)  # create generators 

# Create a spend quartile with 4 groups and pass the previously created labels 
recency_quartiles = pd.qcut(datamart['Recency'], q=4, labels=r_labels)

# Assign the quartile values to the Recency_Quartile column in `data`
datamart['Recency_Quartile'] = recency_quartiles 

# Print `data` with sorted Recency_Days values
print(datamart.sort_values('Recency'))              
# Creating Frequency and Monetary quartiles  and Create labels for Frequency and monetray
f_labels = range(1,5)
m_labels = range(1,5)

#Assign these labels to 4 equal percentile groups based on Frequency.
f_quartiles = pd.qcut(datamart['Frequency'], 4, labels = f_labels)

## Assign these labels to 4 equal percentile groups
m_quartiles = pd.qcut(datamart['MonetaryValue'], 4, labels = m_labels)

#Create new columns F and M

datamart = datamart.assign(F = f_quartiles.values)
datamart = datamart.assign(M = m_quartiles.values)
# check first few rows with new columns   # 
datamart.head()
#Concatenate RFM quartile values to RFM_Segment and converted to string

def join_rfm(x): return str(x['Recency_Quartile']) + str(x['F']) + str(x['M'])
#group the customers into three separate groups based on Recency,Frequency and monetary values

datamart['RFM_Segment'] = datamart.apply(join_rfm, axis=1)

#Sum RFM quartiles values to RFM_Score

datamart['RFM_Score'] = datamart[['Recency_Quartile','F','M']].sum(axis=1)

datamart.head()
# Analyzing RFM segments : Largest 10 RFM segments

datamart.groupby('RFM_Segment').size().sort_values(ascending=False)[:10]
# Filtering on RFM segments 

# Select bottom RFM segment "111" and view top 5 rows

datamart[datamart['RFM_Segment']=='111'][:5]
#Summary metrics per RFM Score

datamart.groupby('RFM_Score').agg({'Recency': 'mean','Frequency': 'mean','MonetaryValue': ['mean', 'count'] }).round(1)

#Use RFM score to group customers into Gold, Silver and Bronze segments.

def segment_me(df):
    if df['RFM_Score'] >= 9:
       return 'Gold'
    elif (df['RFM_Score'] >= 5) and (df['RFM_Score'] < 9):
       return 'Silver'
    else:
       return 'Bronze'
# Create a new variable called Segment_Level

datamart['Segment_level'] = datamart.apply(segment_me, axis=1)
#Analyze average values of Recency, Frequency and MonetaryValue for the custom segments created.

datamart.groupby('Segment_level').agg({'Recency': 'mean','Frequency': 'mean','MonetaryValue': ['mean', 'count']}).round(1)
