import pandas as pd

import numpy as np



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

autos=pd.read_csv('/kaggle/input/used-cars-database-50000-data-points/autos.csv',encoding='Latin-1')
autos.head(5)
autos.info()
columns_names = autos.columns
columns_names
columns_names = autos.columns

columns_names_converted=[]



## helper function to conver columns names



import re

def convert_2_snakecase (name):

    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)

    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()



## applyting the function

for i in columns_names:

    columns_names_converted.append(convert_2_snakecase(i))



## applyting transformation

autos.columns=columns_names_converted
autos
autos.describe(include='all')
autos["nr_of_pictures"].value_counts()
autos = autos.drop(["nr_of_pictures", "seller", "offer_type"], axis=1)
autos['price'].unique()
autos['price']=autos['price'].str.replace("$","").str.replace(",","").astype(int)
autos['price'].head(10)
autos['odometer'].unique()
autos['odometer']=autos['odometer'].str.replace("km","").str.replace(",","").astype(int)
autos['odometer'].unique()
autos.rename(columns={'odometer':'odometer_km'},inplace=True)
autos
print(autos['price'].describe())

autos['price'].value_counts().sort_index(ascending=True)
autos = autos[autos["price"].between(1,351000)]

autos['price'].describe()

autos['odometer_km'].describe()

autos['odometer_km'].value_counts().sort_index(ascending=False)
autos[['date_crawled','date_created','last_seen']][0:5]
autos['date_crawled'].value_counts(normalize=True, dropna=False).head(10)
autos['year_of_registration'].describe()
print("year_of_registration BEFORE cleanup:")

print(autos['year_of_registration'].describe())

autos = autos[autos["year_of_registration"].between(1900,2016)]

print("year_of_registration AFTER cleanup:")

print(autos['year_of_registration'].describe())
autos['year_of_registration'].value_counts(normalize=True).head(10)
autos['brand'].unique()
autos['brand'].value_counts().index
brands_list=autos['brand'].value_counts().index

brands_mean_price={}



for i in brands_list:

    mean_price = autos['price'][autos['brand']==i].mean()

    brands_mean_price[i]=int(mean_price)

    

    



import operator

brands_mean_price_sorted = sorted(brands_mean_price.items(), key=operator.itemgetter(1),reverse=True)

    

brands_mean_price_sorted[:15]
brands_list=autos['brand'].value_counts().index

brands_mean_mileage={}





for i in brands_list:

    mean_mileage = autos['odometer_km'][autos['brand']==i].mean()

    brands_mean_mileage[i]=int(mean_mileage)

    

brands_mean_mileage    
bmp_series = pd.Series(brands_mean_price)



b_mileage_series = pd.Series(brands_mean_mileage)



#print(bmp_series)



df = pd.DataFrame(bmp_series, columns=['mean_price'])

df['mean_mileage']=b_mileage_series



df
