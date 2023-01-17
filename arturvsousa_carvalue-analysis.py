import pandas as pd

import numpy as np
autos = pd.read_csv('../input/used-cars-database-50000-data-points/autos.csv', encoding='Latin_1')
autos.info()

autos.head()
autos = autos.rename(index=str, columns={'yearOfRegistration': 'registration_year', 'monthOfRegistration': 'registration_month', 'notRepairedDamage': 'unrepaired_damage', 'dateCreated': 'ad_created', 'dateCrawled':'date_crawled', 'offerType':'offer_type', 'vehicleType':'vehicle_type', 'powerPS':'power_ps', 'odometer': 'odometer_km', 'fuelType':'fuel_type', 'nrOfPictures': 'pictures_num', 'postalCode':'postal_code', 'lastSeen':'last_seen'})
price_num = autos['price'].str.replace(r'[a-zA-Z]', '')

price_num = price_num.str.replace(',','')

price_num = price_num.str.replace('$', '')

price_num = price_num.astype(int)
autos['price'] = price_num
km = autos['odometer_km'].str.replace(',','')

km = km.str.replace('km', '').astype(int)

autos['odometer_km'] = km
valid_price = autos[autos['price'].between(100,500000)]
autos = valid_price

autos.info()
autos['odometer_km'].describe()
autos['date_crawled']
date_crawled = autos['date_crawled'].str[:10]
date_crawled.value_counts(normalize=True, dropna=False).sort_index(ascending=False).describe()
ad_created = autos['ad_created'].str[:10]

ad_created.value_counts(normalize=True, dropna=False).sort_index().describe()
last_seen = autos['last_seen'].str[:10]

last_seen.value_counts(normalize=True, dropna=False).sort_index().describe()
print ('Top days for date crawled, date of ad creation and last seen')

print (date_crawled.max())

print (ad_created.max())

print (last_seen.max())
print ('The worst day for the same')

print (date_crawled.min())

print (ad_created.min())

print (last_seen.min())
registration_year = autos['registration_year']

registration_year.describe()
safe_reg_year_bool = autos['registration_year'].between(1900, 2016)
safe_years_autos = autos[safe_reg_year_bool]

safe_years_autos
autos = safe_years_autos

autos['registration_year'].value_counts(normalize=True)
expensive_cars = {}

cheap_cars = {}

top_car = {}

top_mean_price = {}



unibrand = autos['brand'].unique()



for brand in unibrand:

    selected_rows = autos[autos['brand'] == brand]

        

    expensive = selected_rows.sort_values('price', ascending=False)

    cheap = selected_rows.sort_values('price', ascending=True)

        

    top = expensive[:1]

    top = top[['name', 'price', 'registration_year']]

    top_car[brand] = top

    

    expensive_mean = expensive['price'].mean()

    top_mean_price[brand] = expensive_mean

    

    cheapest = cheap[:1]

    cheapest = cheap[['name', 'price', 'registration_year']]

    
top_mean = sorted(top_mean_price.items(), key=lambda kv: kv[1], reverse=True)
top_mean
cheapest[:1]
top_car['porsche']
mean_kilometrage = {}



for brand in unibrand:

    selected_rows = autos[autos['brand'] == brand]

    meankm = selected_rows['odometer_km'].mean()

    mean_kilometrage[brand] = meankm

    
mean_km_apres = sorted(mean_kilometrage.items(), key=lambda kv: kv[1], reverse=True)
mean_km_apres
mean_prices = pd.Series(top_mean_price).astype(float)

mean_km = pd.Series(mean_kilometrage)
km_price_mean = pd.DataFrame()

km_price_mean['mean_price'] = mean_prices
km_price_mean['mean_km'] = mean_km
km_price_mean
index_km_value = km_price_mean['mean_km'] / km_price_mean['mean_price']
index_km_value.sort_values()
brand_ocurrences = {}

brand_total_vehicles = {}



for brand in unibrand:

    selected_rows = autos[autos['brand'] == brand]

    brand_count = selected_rows['model']

    brand_ocurrences[brand] = brand_count

    brand_total_vehicles[brand] = brand_ocurrences[brand].shape[0]
brand_total_vehicles_org = sorted(brand_total_vehicles.items(), key=lambda kv: kv[1], reverse=True)
brand_total_vehicles_org
volks_models = autos[autos['brand'] == 'volkswagen']
volks_models['model'].value_counts()
damage = autos[autos['unrepaired_damage'] == 'ja']
no_damage = autos[autos['unrepaired_damage'] != 'ja']
no_damage['price'].mean()
damage['price'].mean()