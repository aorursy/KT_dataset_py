import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')

%matplotlib inline
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

data = data.astype({'BOROUGH': 'str', 'NEIGHBORHOOD': 'category', 'BUILDING CLASS CATEGORY': 'str', 

                    'ADDRESS': 'str', 'ZIP CODE': 'int', 'LAND SQUARE FEET': 'int', 'GROSS SQUARE FEET': 'int', 

                    'YEAR BUILT': 'int', 'SALE PRICE': 'int', 'SALE DATE': 'datetime64'})



boroughs=[('1','Manhattan'),('2','Bronx'),('3','Brooklyn'),('4','Queens'),('5','Staten Island')]

for index,borough in boroughs:

    data['BOROUGH'].replace(index,borough,inplace=True)

data.reset_index(drop=True, inplace=True)

data.shape
# Set to dates.year

dates = pd.DatetimeIndex(data['SALE DATE'])

data['SALE DATE'] = dates.year
# Check for types

#data.dtypes
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

data.shape
# Filter 3

data['BUILDING CLASS CATEGORY'][data['BUILDING CLASS CATEGORY'].str.contains('ONE FAMILY',regex=False)] = 'ONE FAMILY'

data['BUILDING CLASS CATEGORY'][data['BUILDING CLASS CATEGORY'].str.contains('TWO FAMILY',regex=False)] = 'TWO FAMILY'

data['BUILDING CLASS CATEGORY'][data['BUILDING CLASS CATEGORY'].str.contains('THREE FAMILY',regex=False)] = 'THREE FAMILY'

building_filter=~data['BUILDING CLASS CATEGORY'].isin(['ONE FAMILY','TWO FAMILY','THREE FAMILY'])

data.drop(data[building_filter].index,inplace=True)

data.reset_index(drop=True, inplace=True)

data.shape
counties = ['Manhattan', 'Bronx', 'Brooklyn', 'Queens', 'Staten Island']

family = ['ONE FAMILY', 'TWO FAMILY', 'THREE FAMILY']

year = range(2003, 2020)
data_new = pd.DataFrame(columns=['BOROUGH', 'FAMILY SIZE', 'GROSS SQUARE FEET', 'SALE PRICE', 'PRICE PER SQUARE FOOT',

                                 'SALE DATE', 'CHANGE', 'RECORDS'])

for i in year:

    for j in family:

        for k in counties:

            data_con = data[(data['BOROUGH'] == k) & (data['BUILDING CLASS CATEGORY'] == j) & (data['SALE DATE'] == i)]

            sale_avg = data_con['SALE PRICE'].mean()

            gross_avg = data_con['GROSS SQUARE FEET'].mean()

            records = data_con['SALE PRICE'].count()

            if i == 2003:

                change = 0

            else:

                change = (int(sale_avg / gross_avg) / data_new.iloc[-15]['PRICE PER SQUARE FOOT'] - 1) * 100            

            data_new = data_new.append({'BOROUGH': k, 'FAMILY SIZE': j, 'GROSS SQUARE FEET': int(gross_avg),

                                        'SALE PRICE': int(sale_avg), 'PRICE PER SQUARE FOOT': int(sale_avg / gross_avg),

                                        'SALE DATE': i, 'CHANGE': change, 'RECORDS': records}, ignore_index=True)

data_new = data_new.astype({'BOROUGH': 'category', 'FAMILY SIZE': 'category', 'GROSS SQUARE FEET': 'int',

                            'SALE PRICE': 'int', 'PRICE PER SQUARE FOOT': 'int', 'SALE DATE': 'int',

                            'CHANGE': 'float', 'RECORDS': 'int'})

data_new.head(n=20)
data_family = pd.DataFrame(columns=['FAMILY SIZE', 'GROSS SQUARE FEET', 'SALE PRICE', 

                                    'PRICE PER SQUARE FOOT','SALE DATE','CHANGE', 'RECORDS'])

for i in year:

    for j in family:

        data_con = data[(data['BUILDING CLASS CATEGORY'] == j) & (data['SALE DATE'] == i)]

        sale_avg = data_con['SALE PRICE'].mean()

        gross_avg = data_con['GROSS SQUARE FEET'].mean()

        records = data_con['SALE PRICE'].count()

        if i == 2003:

            change = 0

        else:

            change = (int(sale_avg / gross_avg) / data_family.iloc[-3]['PRICE PER SQUARE FOOT'] - 1) * 100

        data_family = data_family.append({'FAMILY SIZE': j, 'GROSS SQUARE FEET': int(gross_avg),

                                    'SALE PRICE': int(sale_avg), 'PRICE PER SQUARE FOOT': int(sale_avg / gross_avg),

                                    'SALE DATE': i, 'CHANGE': change, 'RECORDS': records}, ignore_index=True)

        

data_family = data_family.astype({'FAMILY SIZE': 'category', 'GROSS SQUARE FEET': 'int', 'SALE PRICE': 'int', 

                                  'PRICE PER SQUARE FOOT': 'int', 'SALE DATE': 'int', 'CHANGE': 'float', 'RECORDS': 'int'})
data_counties = pd.DataFrame(columns=['BOROUGH', 'GROSS SQUARE FEET', 'SALE PRICE', 'PRICE PER SQUARE FOOT',

                              'SALE DATE', 'CHANGE', 'RECORDS'])

for k in year:

    for l in counties:

        data_con = data[(data['BOROUGH'] == l) & (data['SALE DATE'] == k)]

        sale_avg = data_con['SALE PRICE'].mean()

        gross_avg = data_con['GROSS SQUARE FEET'].mean()

        records = data_con['SALE PRICE'].count()

        if k == 2003:

            change = 0

        else:

            change = (int(sale_avg / gross_avg) / data_counties.iloc[-5]['PRICE PER SQUARE FOOT'] - 1) * 100        

        data_counties = data_counties.append({'BOROUGH': l, 'GROSS SQUARE FEET': int(gross_avg),

                                    'SALE PRICE': int(sale_avg), 'PRICE PER SQUARE FOOT': int(sale_avg / gross_avg),

                                    'SALE DATE': k, 'CHANGE': change, 'RECORDS': records}, ignore_index=True)

        

data_counties = data_counties.astype({'BOROUGH': 'category', 'GROSS SQUARE FEET': 'int', 'SALE PRICE': 'int', 

                                      'PRICE PER SQUARE FOOT': 'int', 'SALE DATE': 'int',  'CHANGE': 'float','RECORDS': 'int'})
data_year = pd.DataFrame(columns=['GROSS SQUARE FEET', 'SALE PRICE', 'PRICE PER SQUARE FOOT',

                              'SALE DATE', 'RECORDS'])

for m in year:

    data_con = data[(data['SALE DATE'] == m)]

    sale_avg = data_con['SALE PRICE'].mean()

    gross_avg = data_con['GROSS SQUARE FEET'].mean()

    records = data_con['SALE PRICE'].count()

    data_year = data_year.append({'GROSS SQUARE FEET': int(gross_avg),

                                'SALE PRICE': int(sale_avg), 'PRICE PER SQUARE FOOT': int(sale_avg / gross_avg),

                                'SALE DATE': m, 'RECORDS': records}, ignore_index=True)

    

data_year = data_year.astype({'GROSS SQUARE FEET': 'int', 'SALE PRICE': 'int', 

                              'PRICE PER SQUARE FOOT': 'int', 'SALE DATE': 'int', 'RECORDS': 'int'})
sns.set(rc={'figure.figsize':(16,9)})

sns.barplot(data=data_year, x='SALE DATE', y='RECORDS', ci=None)

# Year 2011 / Year 2004

# Transactions dropped 60% after the housing bubble, and it is recovering slowly. 

print(f'{(data_year["RECORDS"][8] / data_year["RECORDS"][1] * 100):.2f}' + '%')
labels = counties

sizes = [data[data.BOROUGH=='Manhattan']['SALE DATE'].count(), data[data.BOROUGH=='Bronx']['SALE DATE'].count(), 

         data[data.BOROUGH=='Brooklyn']['SALE DATE'].count(), data[data.BOROUGH=='Queens']['SALE DATE'].count(), 

         data[data.BOROUGH=='Staten Island']['SALE DATE'].count()]

explode = (0.1, 0, 0, 0, 0)

fig1, ax1 = plt.subplots()

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)

ax1.axis('equal')
sns.lineplot(data=data_new, x='SALE DATE', y='PRICE PER SQUARE FOOT', hue='BOROUGH')

#The housing price of Manhattan has a significantly higher variance compared to others. 
sns.lmplot(data=data_counties, x='SALE DATE', y='PRICE PER SQUARE FOOT', col='BOROUGH')
sns.lineplot(data=data_counties, x='SALE DATE', y='CHANGE', hue='BOROUGH')
labels = family

sizes = [data_new[data_new['FAMILY SIZE']=='ONE FAMILY']['RECORDS'].sum(), 

         data_new[data_new['FAMILY SIZE']=='TWO FAMILY']['RECORDS'].sum(), 

         data_new[data_new['FAMILY SIZE']=='THREE FAMILY']['RECORDS'].sum()]

explode = (0.1, 0, 0)

fig1, ax1 = plt.subplots()

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)

ax1.axis('equal')

data_new[data_new['FAMILY SIZE']=='ONE FAMILY']['RECORDS'].sum()
sns.lineplot(data=data_new, x='SALE DATE', y='PRICE PER SQUARE FOOT', hue='FAMILY SIZE')
sns.lmplot(data=data_family, x='SALE DATE', y='PRICE PER SQUARE FOOT', hue='FAMILY SIZE', col='FAMILY SIZE')
sns.lineplot(data=data_family, x='SALE DATE', y='CHANGE', hue='FAMILY SIZE')
sns.set(rc={'figure.figsize':(16,9)})

sns.barplot(data=data_year, x='SALE DATE', y='PRICE PER SQUARE FOOT', ci=None)

# Year 2019 / Year 2003

# Return is 145% over 17 years

# Average return is 5.76% 

# SP500 average return is near 7.58%

print(f'{(data_year["PRICE PER SQUARE FOOT"][16] / data_year["PRICE PER SQUARE FOOT"][0] * 100):.2f}' + '%')

print(f'{(((data_year["PRICE PER SQUARE FOOT"][16] / data_year["PRICE PER SQUARE FOOT"][0]) ** (1 / 16) - 1) * 100):.2f}' + '%')
sns.catplot(data=data_new, kind='bar', x='SALE DATE', y='RECORDS', row='FAMILY SIZE', col='BOROUGH')
sns.catplot(data=data_new, kind='bar', x='SALE DATE', y='CHANGE', row='FAMILY SIZE', col='BOROUGH')
#Winner is Queens(42.9%) Two Family(40.7%)

#Stable return with high volume of transactions



#Was: 1600202 X 22

#Now: 439652 X 10



#Result 1: 42.9 % Queens, 28.7% Brooklyn, 17.0 % Staten Island, 10.5% Bronx, 0.9% Manhattan

#Result 2: Transactions dropped 60% after the housing bubble, and it is recovering slowly. 

#Result 3: The housing price of one family has a higher variance compared to two families & three families.

#Result 4: The housing price of Manhattan has a significantly higher variance compared to others. 

#Result 5: As of 2019, the price is nearly 2.5 times the price of the year 2008. 
