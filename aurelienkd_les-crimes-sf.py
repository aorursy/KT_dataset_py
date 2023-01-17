import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
data = pd.read_csv('../input/train.csv', parse_dates=['Dates'])
print(data.shape)
data.head(3)

test = pd.read_csv('../input/test.csv', parse_dates=['Dates'])
print(test.shape)
test.head(3)
data.info()
data['Dates-year'] = data['Dates'].dt.year
data['Dates-month'] = data['Dates'].dt.month
data['Dates-day'] = data['Dates'].dt.day
data['Dates-hour'] = data['Dates'].dt.hour
data['Dates-minute'] = data['Dates'].dt.minute
data['Dates-second'] = data['Dates'].dt.second
fig, ((axis1,axis2,axis3),(axis4,axis5,axis6)) = plt.subplots(nrows=2, ncols=3)
fig.set_size_inches(18,6)

sns.countplot(data=data, x='Dates-year', ax=axis1)
sns.countplot(data=data, x='Dates-month', ax=axis2)
sns.countplot(data=data, x='Dates-day', ax=axis3)
sns.countplot(data=data, x='Dates-hour', ax=axis4)
sns.countplot(data=data, x='Dates-minute', ax=axis5)
sns.countplot(data=data, x='Dates-second', ax=axis6)
fig, (axis1,axis2) = plt.subplots(nrows=2, ncols=1, figsize=(18,4)) 
sns.countplot(data=data, x='Dates-hour', ax=axis1)
sns.countplot(data=data, x='Dates-minute', ax=axis2)
# Dates-hour exploration
data['Dates-hour'].value_counts()[-5:]
## 

def bin_data_minute(hour):
    if hour >=8 & hour ==0:
        return 'High_hour'
    else:
        return 'Low_hour'
    
data['bin_dates_hour'] = data['Dates-hour'].apply(bin_data_minute)
fig, axis1 = plt.subplots(figsize=(10,20))
sns.countplot(data=data, y='Category', hue='bin_dates_hour',ax=axis1)
data['Dates-minute'].value_counts()[:10]
# Nombre d'adresse contenant '/'
street_length = len(data[data['Address'].str.contains('/')])
print(street_length)

# Nombre d'adresse en block
print(len(data['Address'])- street_length)
def bin_address(address):
    if '/' in address and 'of' not in address:
        return 'Street'
    else:
        return 'Block'
data['Address_type'] = data['Address'].apply(bin_address)
data[['Address', 'Address_type']].head(5)
sns.countplot(data=data, x='Address_type')
fig, axis1 = plt.subplots(figsize=(10,20))
sns.countplot(data=data, y='Category', hue='Address_type', ax=axis1)
print(len(data[data['Address'] == 'OAK ST / LAGUNA ST']))
print(len(data[data['Address'] == 'LAGUNA ST / OAK ST']))
crossload = data[data['Address'].str.contains('/')]['Address'].unique()
print('Nombre unique d\'adresse : {0}'.format(len(crossload)))
topN_address_list = data['Address'].value_counts()
topN_address_list = topN_address_list[topN_address_list >=100]
topN_address_list = topN_address_list.index
print('topN criminal address count is',len(topN_address_list))
data['Address_clean'] = data['Address']
data.loc[~data['Address'].isin(topN_address_list), "Address_clean"] = 'Others'

data[['Address','Address_clean']].head(5)
crossload = data[data['Address_clean'].str.contains('/')]
print(crossload.shape)
crossload['Address_clean'].head(3)
crossload_list = crossload['Address_clean'].unique()
print('Before Adjustment ST_Address length is {0}' .format(len(crossload_list)))
from tqdm import tqdm
print(crossload_list[0].split('/')[1].strip() + " / " + crossload_list[0].split('/')[0].strip())
print(crossload_list[0])
for address in tqdm(crossload_list):
    reverse_address = address.split('/')[1].strip() + " / " + address.split('/')[0].strip()
    data.loc[data['Address_clean'] == reverse_address, 'Address_clean'] = address
crossload_list = data[data['Address_clean'].str.contains('/')]
crossload_list = crossload_list['Address_clean'].unique()
print('Final ST_Address length is {0}' .format(len(crossload_list)))
data[['Category','PdDistrict']].head(3)
data['PdDistrict'].value_counts()
sns.countplot(data=data,  x='PdDistrict')