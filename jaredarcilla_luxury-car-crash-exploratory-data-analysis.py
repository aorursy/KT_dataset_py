% matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import missingno as msno
crashes_df = pd.read_csv('../Data/Motor_Vehicle_Crashes_-_Vehicle_Information__Three_Year_Window.csv')

crashes_df.head()
sales_df = pd.read_csv('car_sales.csv')

sales_df.head()
msno.matrix(crashes_df)
print('Tail of the Vehicle Make column: ' + str(crashes_df['Vehicle Make'].unique()[-30:]))

print('\n\nNumber of unique brand names: ' + str(len(crashes_df['Vehicle Make'].unique())))
vin_missing = crashes_df[(crashes_df['Vehicle Make'].notnull()) & crashes_df['Partial VIN'].isnull()]

vin_missing_unique = vin_missing['Vehicle Make'].unique()

len(vin_missing_unique)
vin_missing['Vehicle Make'].unique()[:20]
crashes_df = crashes_df.dropna(axis=0,subset=['Partial VIN'])

crashes_df = crashes_df.reset_index()
crashes_df = crashes_df.drop(labels=['index'],axis=1)
crashes_df.head()
vin_info = pd.read_csv('../vin_info.csv')
vin_info.head()
car_brands_df = vin_info.groupby(by='make').count()

car_brands_df = car_brands_df.sort_values(by='car_id',ascending=False)

car_brands_df.head()
car_brands_df = pd.DataFrame(car_brands_df.iloc[:,0])

car_brands_df = car_brands_df.rename(index=str,columns={'Unnamed: 0':'Num of Accidents'})

car_brands_df.head(10)
ax = car_brands_df.iloc[:10].plot(kind='bar')

ax.set_title('# of Car Accidents in NYS by Make from 2012-2015')

ax.set_ylabel('# of Accidents')

ax.set_xlabel('Car Make')
sales_df.head(20)
long_column_name = 'U.S. 2014 Calendar Year New Vehicle Sales Volume  \nAll Vehicles & All Automakers – Two Sheets \n© GoodCarBadCar.net'

sales_df = sales_df.rename(index=str,columns={long_column_name: 'Make',

                                              'Unnamed: 1': '2014',

                                              'Unnamed: 2': '2013',

                                              'Unnamed: 3': '% Change'})
# Checks if 'Brand Total' is in a string.

def brand_total(x):

    return ('Brand Total' in str(x)) 
sales_df = sales_df[sales_df['Make'].map(brand_total)]
# Formatting 'Make' column such that it is the same with our crashes dataset, then making it the index.

sales_df['Make'] = sales_df['Make'].map(lambda x: x.replace(" Brand Total","").upper())

sales_df = sales_df.set_index(sales_df['Make']).drop('Make',axis=1)

sales_df.head()
#Turning the strings to ints

sales_df.iloc[:,0] = sales_df.iloc[:,0].map(lambda x: int(x.replace(",","")))

sales_df.iloc[:,1] = sales_df.iloc[:,1].map(lambda x: int(x.replace(",","")))
sales_df['Mean Sales'] = (sales_df.iloc[:,0] + sales_df.iloc[:,1])/2

sales_df.head()
car_brands_df['Mean Sales'] = sales_df.ix[car_brands_df.index,'Mean Sales']
car_brands_df
car_brands_df['Accident Score'] = car_brands_df['Num of Accidents']/car_brands_df['Mean Sales']
car_brands_df = car_brands_df.sort_values(by='Accident Score',ascending=False)

car_brands_df.head(15)
car_brands_df = car_brands_df.drop('SUZUKI',axis=0)

car_brands_df.head(10)
ax = car_brands_df.iloc[1:15,2].plot(kind='bar')

ax.set_title('Accident Score of Different Car Brands')

ax.set_ylabel('Accident Score')

ax.set_xlabel('Car Make')
luxury = ['LINCOLN','VOLVO','ACURA','JAGUAR','INFINITI']

non_lux = ['SUZUKI','MITSUBISHI','DODGE','HONDA','CHRYSLER']

print('luxury car accident score: ' + str(car_brands_df.ix[luxury,2].sum()))

print('non-luxury car accident score: ' + str(car_brands_df.ix[non_lux,2].sum()))