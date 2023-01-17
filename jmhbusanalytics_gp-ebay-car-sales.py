#import needed modules

import numpy as np

import pandas as pd
#Import dataset

autos = pd.read_csv('../input/used-cars-database-50000-data-points/autos.csv', encoding = 'Latin-1')



#Explore dataset

autos.info()

autos.head(3)
#Clean the column names from camelcase to snakecase

autos.columns = ['date_crawled', 'name', 'seller', 'offer_type', 'price', 'ab_test',

       'vehicle_type', 'registration_year', 'gear_box', 'power_ps', 'model',

       'odometer', 'registration_month', 'fuel_type', 'brand',

       'unrepaired_damage', 'ad_created', 'number_of_pictures', 'postal_code',

       'last_seen']



#Check column name update

autos.columns

autos.head(2)





#Explore dataset to determine what other cleaning tasks are needed

autos.describe(include = 'all')









#Investigating number of picture columns to see if it has different values

autos['number_of_pictures'].value_counts()
#Droping columns from the autos dataset

autos = autos.drop(['number_of_pictures', 'seller', 'offer_type'], axis = 1)

autos.info()
#Change the Price and odometer columns to numeric datatypes/ remove non-numeric characters

print(autos[['price','odometer']].dtypes)

print(autos[['price', 'odometer']].head(2))

autos['price'] = autos['price'].str.replace('$','').str.replace(',','').astype(int)

autos['odometer'] = autos['odometer'].str.replace(',','').str.replace('km','').astype(int)

print(autos[['price','odometer']].head(2))

print(autos[['price', 'odometer']].dtypes)



#Rename the odometer column to odomter_km to specify length

autos.rename({'odometer':'odometer_km'}, axis = 1, inplace = True)



#Check if name has been changed 

autos.columns
autos['odometer_km'].value_counts()
#Explore the price data to look for outliers

print(autos['price'].value_counts().shape)

print(autos['price'].describe())

print(autos['price'].value_counts().head(20))
print(autos['price'].value_counts().sort_index(ascending = True).head(20))
print(autos['price'].value_counts().sort_index(ascending = False).head(20))
#Keep prices that fall within 1 to 350,000 dollars

autos = autos[autos['price'].between(1,351000)]

autos['price'].describe()
autos['date_crawled'].str[:10].value_counts(normalize = True, dropna = False).sort_index()
autos['ad_created'].str[:10].value_counts(normalize = True, dropna = False).sort_index().tail(40)
autos['last_seen'].str[:10].value_counts(normalize = True, dropna = False).sort_index()
autos['registration_year'].describe()
#Find the percentage of registration dates that dont fall within 1900s to 2016

(~autos['registration_year'].between(1900,2016)).sum() / autos.shape[0]
#Find the percentage of cars being registered for each year

autos = autos[autos["registration_year"].between(1900, 2016)]

print(autos["registration_year"].value_counts(normalize = True).head(15))

print("\n")



#Compute the total percentage of cars registrations that fall within 1900s to 2016

print(autos["registration_year"].value_counts(normalize = True).sum())
#Find the average price of the top 20 brands on ebay

brand_count = autos['brand'].value_counts(normalize = True).head(20)

brand_count_index = brand_count.index



brand_mean_price = {}



for b in brand_count_index:

    selected_rows = autos[autos['brand'] == b]

    mean_price = selected_rows['price'].mean()

    brand_mean_price[b] = round(float(mean_price),2)

    

    

print(brand_count)    

print('\n')

print(brand_mean_price) 





    

#Take the brand_mean_price dictionary and translate it into a dataframe

bmp_series = pd.Series(brand_mean_price)

print(bmp_series)



df = pd.DataFrame(bmp_series, columns=['mean_price'])

df
#Find the average mileage of the top 20 brands on eBay

brand_mean_mileage = {}



for b in brand_count_index:

    selected_rows = autos[autos['brand'] == b]

    mean_mileage = selected_rows['odometer_km'].mean()

    brand_mean_mileage[b] = int(mean_mileage)

    

brand_mean_mileage



#Translate brand_mean_mileage dictionary into a dataframe/ change brand_mean_price into a series

mean_mileage = pd.Series(brand_mean_mileage).sort_values(ascending = False)

mean_prices = pd.Series(brand_mean_price).sort_values(ascending = False)

print(mean_mileage)

print('\n')

print(mean_prices)







brand_info = pd.DataFrame(mean_mileage, columns = ['mean_mileage'])

brand_info



#Create a dataframe that holds brands, mean_mileage and mean_price

brand_info['mean_price'] = mean_prices

brand_info