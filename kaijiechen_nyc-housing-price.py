import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
# This is how to get data from 85 CSVs into 1 dataset

# 1600202 x 10

'''

years = range(2003, 2011)

years2 = range(2011, 2020)

counties = ['manhattan', 'bronx', 'brooklyn', 'queens', 'statenisland']

dataset = []



for year in years:

    for county in counties:

        filename = 'https://raw.githubusercontent.com/LosingMind/NYC-Housing-Price/master/dataset/' + str(

            year) + '_' + str(county) + '.csv'

        data = pd.read_csv(filename, sep=',', usecols=[1, 2, 3, 9, 11, 15, 16, 17, 20, 21], header=3)

        data.columns = data.columns.str.replace('\n', '')

        dataset.append(data)

        print(filename)



for year in years2:

    for county in counties:

        filename = 'https://raw.githubusercontent.com/LosingMind/NYC-Housing-Price/master/dataset/' + str(

            year) + '_' + str(county) + '.csv'

        data = pd.read_csv(filename, sep=',', usecols=[1, 2, 3, 9, 11, 15, 16, 17, 20, 21], header=4)

        data.columns = data.columns.str.replace('\n', '')

        dataset.append(data)

        print(filename)

        

datasets = pd.concat(dataset, axis=0, ignore_index=True)

datasets.to_csv('datasets1.csv', encoding='utf-8', index=False)

'''
# Load DataFrame from csv

data = pd.read_csv('../input/nyc-housing-data-20032019/data.csv')

data.shape
# Test na

data.isna().sum()
# Drop na

data.dropna(subset=['ZIP CODE', 'LAND SQUARE FEET', 'GROSS SQUARE FEET', 'YEAR BUILT'], inplace=True)

data.shape
# Converting types

data = data.astype({'BOROUGH': 'str', 'NEIGHBORHOOD': 'category', 'BUILDING CLASS CATEGORY': 'str', 'ADDRESS': 'str', 'ZIP CODE': 'int', 'LAND SQUARE FEET': 'int', 'GROSS SQUARE FEET': 'int', 'YEAR BUILT': 'int', 'SALE PRICE': 'int', 'SALE DATE': 'datetime64'})



boroughs=[('1','Manhattan'),('2','Bronx'),('3','Brooklyn'),('4','Queens'),('5','Staten Island')]

for index,borough in boroughs:

    data['BOROUGH'].replace(index,borough,inplace=True)

data.reset_index(drop=True, inplace=True)

data.shape
data
# Set to dates.year

dates = pd.DatetimeIndex(data['SALE DATE'])

data['SALE DATE'] = dates.year

data
# Check for types

data.dtypes
data.shape
# Filter 1

data.replace(0,np.nan,inplace=True)

columns=['ZIP CODE', 'YEAR BUILT','SALE PRICE']

data.dropna(subset=columns,inplace=True)

data.replace(np.nan,0,inplace=True)

data.shape
# Filter 2

data = data[data['YEAR BUILT'] >= 1800]

data = data[data['SALE PRICE'] >= 100000]

data = data[data['GROSS SQUARE FEET'] >= 100]

data
# Filter 3

data['BUILDING CLASS CATEGORY'][data['BUILDING CLASS CATEGORY'].str.contains('ONE FAMILY',regex=False)]='ONE FAMILY'

data['BUILDING CLASS CATEGORY'][data['BUILDING CLASS CATEGORY'].str.contains('TWO FAMILY',regex=False)]='TWO FAMILY'

data['BUILDING CLASS CATEGORY'][data['BUILDING CLASS CATEGORY'].str.contains('THREE FAMILY',regex=False)]='THREE FAMILY'

building_filter=~data['BUILDING CLASS CATEGORY'].isin(['ONE FAMILY','TWO FAMILY','THREE FAMILY'])

data.drop(data[building_filter].index,inplace=True)

data.reset_index(drop=True, inplace=True)

data
counties = ['Manhattan', 'Bronx', 'Brooklyn', 'Queens', 'Staten Island']

family = ['ONE FAMILY', 'TWO FAMILY', 'THREE FAMILY']

year = range(2003, 2020)

data_new = pd.DataFrame(columns=['BOROUGH', 'FAMILY SIZE', 'GROSS SQUARE FEET', 'SALE PRICE', 'PRICE PER SQUARE FOOT',

                              'SALE DATE', 'RECORDS'])
for i in year:

    for j in family:

        for k in counties:

            data_con = data[(data['BOROUGH'] == k) & (data['BUILDING CLASS CATEGORY'] == j) & (data['SALE DATE'] == i)]

            sale_avg = data_con['SALE PRICE'].mean()

            gross_avg = data_con['GROSS SQUARE FEET'].mean()

            records = data_con['SALE PRICE'].count()

            data_new = data_new.append({'BOROUGH': k, 'FAMILY SIZE': j, 'GROSS SQUARE FEET': int(gross_avg),

                                        'SALE PRICE': int(sale_avg), 'PRICE PER SQUARE FOOT': int(sale_avg / gross_avg),

                                        'SALE DATE': i, 'RECORDS': records}, ignore_index=True)



data_new
sns.regplot(data=data,

            y='PRICE PRICE',

            x="SALE DATE",

            marker='^',

            color='g',

            x_bins=5,

            order=2)
plt.figure(figsize=(60,18))

ax=sns.boxplot(x='BOROUGH',y='SALE PRICE',data=onefamily)

#ax.set(ylim=(0,2000000))

ax.set_yscale('log')

plt.show()
sns.boxplot(x='BOROUGH', y='SALE PRICE', data=onefamily)

plt.show()
#sns.distplot(twofamily['SALE PRICE'])

#plt.show()