import pandas as pd
import numpy as np

#reading csv file
autos = pd.read_csv('../input/autos.csv',encoding='latin-1')
autos
autos.info()
autos.head()
#changing the column names

autos.columns = ['date_crawled', 'name', 'seller', 'offer_type', 'price', 
                 'ab_test','vehicle_type', 'registration_year', 'gearbox', 
                 'power_ps', 'model','odometer', 'registration_month', 
                 'fuel_type', 'brand','unrepaired_damage', 'ad_created', 
                 'num_photos', 'postal_code','last_seen']


autos.head()
autos.describe(include='all')
autos = autos.drop(columns=['seller','offer_type','num_photos'])
#converting column 'price' from object to numeric type
autos['price'] = (autos['price']
                 .str.replace('$','')
                 .str.replace(',','')
                  .astype(float)
                 )

#converting column 'odometer' from object to numeric type
autos['odometer'] = (autos['odometer']
                    .str.replace(',','')
                    .str.replace('km','')
                    .astype(float)
                    )
#renaming the column 'odometer' to 'odometer_km'
autos = autos.rename({'odometer':'odometer_km'}, axis=1)
print(autos['price'].unique().shape)
print(autos['price'].describe())
print(autos['price'].value_counts().head(20))
autos['price'].value_counts().sort_index(ascending=False).head(20)
autos['price'].value_counts().sort_index(ascending=True).head(20)
autos = autos[autos['price'].between(1,350000)]
autos['price'].describe()
autos['odometer_km'].value_counts().sort_index(ascending=False)
autos[['date_crawled','last_seen',
      'ad_created','registration_month',
      'registration_year']].info()
autos[['date_crawled','ad_created','last_seen']][0:5]
(autos['date_crawled']
 .str[:10]
 .value_counts(normalize=True,dropna=False)
 .sort_index()
)
(autos['date_crawled']
 .str[:10]
 .value_counts(normalize=True,dropna=False)
 .sort_values()
)
(autos['last_seen']
 .str[:10]
 .value_counts(normalize=True,dropna=False)
 .sort_index()
)
print(autos["ad_created"].str[:10].unique().shape)
(autos["ad_created"]
        .str[:10]
        .value_counts(normalize=True, dropna=False)
        .sort_index()
        )
autos['registration_year'].describe()
(~autos['registration_year'].between(1900,2016)).sum()/autos.shape[0]
autos = autos[autos['registration_year'].between(1900,2016)]
autos['registration_year'].value_counts(normalize=True).head(10)


autos['brand'].value_counts(normalize=True)
brand_counts = autos['brand'].value_counts(normalize=True)
common_brands = brand_counts[brand_counts > .05].index
print(common_brands)
brand_mean_prices = {}

for b in common_brands:
    b_only = autos[autos['brand'] == b]
    mean_price = b_only['price'].mean()
    brand_mean_prices[b] = int(mean_price)

brand_mean_prices
brand_mean_mileage = {}
for b in common_brands:
    b_only = autos[autos['brand'] == b]
    mean_mil = b_only['odometer_km'].mean()
    brand_mean_mileage[b] = int(mean_mil)
    

mean_mileage = pd.Series(brand_mean_mileage).sort_values(ascending=False)
mean_price = pd.Series(brand_mean_prices).sort_values(ascending=False)
brand_agg = pd.DataFrame(mean_mileage,columns=['mean_mileage'])
brand_agg
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
for each in ['vehicle_type','gearbox','fuel_type','unrepaired_damage']:
    autos[each] = autos[each].map(words_translated)
    
print(autos['vehicle_type'].unique())
print(autos['gearbox'].unique())
print(autos['fuel_type'].unique())
print(autos['unrepaired_damage'].unique())
autos.head()
date_cols = ['date_crawled','ad_created','last_seen']

for each in date_cols:
    autos[each] = (autos[each]
                  .str[:10]
                  .str.replace('-','')
                  .astype(int)
                  )
autos.head()

autos.info()