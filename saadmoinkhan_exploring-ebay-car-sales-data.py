import pandas as pd

import numpy as np



autos = pd.read_csv('../input/used-cars-database-50000-data-points/autos.csv', encoding  = 'Latin-1')
autos.head()
autos.info()
autos.columns
autos.columns = ['date_crawled', 'name', 'seller', 'offer_type', 'price', 'abtest',

       'vehicle_type', 'registration_year', 'gearbox', 'powerPS', 'model',

       'odometer', 'registration_month', 'fuel_type', 'brand',

       'unrepaired_damage', 'ad_created', 'nr_of_pictures', 'postal_code',

       'last_seen']
autos.describe(include = 'all')
autos.loc[:,'price'] = autos.loc[:,'price'].str.replace('$','').str.replace(',','').astype(float)

autos.loc[:,'odometer'] = autos.loc[:,'odometer'].str.replace('km','').str.replace('kms','').str.replace(',','').astype(float)

autos.rename({'odometer':'odometer_kms'}, inplace = True, axis = 1)



autos.describe(include = 'all')
autos.drop(columns = ['offer_type', 'seller', 'nr_of_pictures'], inplace = True)
autos.describe(include = 'all')
print(autos.loc[:,'price'].unique().shape,'\n')

print(autos.loc[:,'price'].describe(), '\n')

print(autos.loc[:,'price'].value_counts().sort_index(ascending = True).head(20))
print(autos.loc[:,'price'].unique().shape,'\n')

print(autos.loc[:,'price'].describe(), '\n')

print(autos.loc[:,'price'].value_counts().sort_index(ascending = False).head(20))
autos = autos.loc[autos.loc[:,'price'].between(1,350000)]



print(autos.loc[:,'price'].unique().shape,'\n')

print(autos.loc[:,'price'].describe(), '\n')

print(autos.loc[:,'price'].value_counts().sort_index(ascending = False).head(20))
print(autos.loc[:,'odometer_kms'].unique().shape,'\n')

print(autos.loc[:,'odometer_kms'].describe(), '\n')

print(autos.loc[:,'odometer_kms'].value_counts().sort_index(ascending = False).head(20))
print(autos.loc[:,'odometer_kms'].unique().shape,'\n')

print(autos.loc[:,'odometer_kms'].describe(), '\n')

print(autos.loc[:,'odometer_kms'].value_counts().sort_index(ascending = True).head(20))
autos.loc[:,['date_crawled', 'registration_month', 'registration_year', 'ad_created', 'last_seen']].info()
autos.loc[:,['date_crawled', 'ad_created', 'last_seen']]
autos.loc[:,['date_crawled', 'ad_created', 'last_seen']] = autos.loc[:,['date_crawled', 'ad_created', 

                                                                        'last_seen']].apply(pd.to_datetime, 

                                                                                            format = '%Y-%m-%d %H:%M:%S')
autos.info()
autos.loc[:,'registration_year'].unique()
autos.loc[:,'registration_year'].value_counts().sort_index(ascending = True)
print(autos.loc[:,'last_seen'].max())

print(autos.loc[:,'last_seen'].min())
# Calculating the percentages of entries which do not lie in the range of 1950-2016



((~autos.loc[:,'registration_year'].between(1950,2016)).sum()/autos.shape[0]) * 100
autos.loc[:,'registration_year'] = autos.loc[autos.loc[:,'registration_year'].between(1950,2016)]



autos.loc[:,'registration_year'].value_counts().sort_index(ascending = True)
#Converting to datetime



autos.loc[:,'registration_year'] = autos.loc[:,'registration_year'].apply(pd.to_datetime, format = '%Y')
autos.loc[:,'registration_month'].unique()
autos.loc[:,'registration_month'].value_counts().sort_index(ascending = True)
autos.loc[:,'registration_month'] = (autos.loc[:,'registration_month']).replace(12,0)



#Viewing the data again

autos.loc[:,'registration_month'].value_counts().sort_index(ascending = True)
#Converting to datetime



autos.loc[:,'registration_month'] = autos.loc[:,'registration_month'].apply(pd.to_datetime, format = '%M')
autos.info()
#Percentage of top Brands



print(autos.loc[:,'brand'].value_counts(normalize = True) * 100)
brand_counts = autos['brand'].value_counts(normalize=True)

top_brands = brand_counts[brand_counts > 0.05].index



print(top_brands)
brand_mean_prices = {}



for x in top_brands:

    temp_brand_df = autos.loc[autos.loc[:,'brand'] == x]

    temp_mean = temp_brand_df.loc[:,'price'].mean()

    brand_mean_prices[x] = int(temp_mean)

    

brand_mean_prices
brand_mean_mileage = {}



for x in top_brands:

    temp_brands_df = autos.loc[autos.loc[:,'brand'] == x]

    temp_kms = temp_brands_df.loc[:,'odometer_kms'].mean()

    brand_mean_mileage[x] = int(temp_kms)

    

brand_mean_mileage
mean_mileage = pd.Series(brand_mean_mileage).sort_values()

mean_price = pd.Series(brand_mean_prices).sort_values()
brand_agg = pd.DataFrame(mean_mileage, columns = ['mean_mileage'])

brand_agg['mean_price'] = mean_price



brand_agg
autos.head()
autos['vehicle_type'].unique()
autos['gearbox'].unique()
autos['fuel_type'].unique()
autos['unrepaired_damage'].unique()
words_translated = {

    'bus':'bus',

    'limousine':'limousine',

    'kleinwagen':'supermini',

    'kombi':'station_wagon',

    'coupe':'coupe',

    'suv':'suv',

    'cabrio':'cabrio',

    'andere' :'other',

    'manuell':'manual',

    'automatik':'automatic',

    'lpg':'lpg',

    'benzin':'petrol',

    'diesel':'diesel',

    'cng':'cng',

    'hybrid':'hybrid',

    'elektro':'electro',

    'nein':'no',

    'ja':'yes'

}
for x in ['vehicle_type','gearbox','fuel_type','unrepaired_damage']:

    autos[x] = autos[x].map(words_translated)
autos.head()
autos.groupby(['brand','model']).size().idxmax()
temp = autos.loc[(autos.loc[:,'brand'] == 'volkswagen') & (autos.loc[:,'model'] == 'golf')]

print('Total cars are :',temp['model'].value_counts()[0],'\n')

print('Average price is : $',temp['price'].mean(), sep = '')

print('Average kms are : ',int(temp['odometer_kms'].mean()),'kms', sep = ' ')
autos.loc[:,'odometer_range'] = pd.cut(autos.odometer_kms, bins = [

    0,30000,60000,90000,120000,150000], labels = ['0-30000', '30000-60000', '60000-90000','90000-120000', '120000-150000'])
ranges = autos.groupby('odometer_range').mean()

ranges.loc[:,'price']
damage = autos.groupby('unrepaired_damage').mean()

damage.loc[:,'price']