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
autos = pd.read_csv('/kaggle/input/autos.csv', encoding='Latin-1')
autos
autos.info()
autos.columns
autos.rename(columns={'dateCrawled':'date_crawled', 'offerType':'offer_type',

       'vehicleType':'vehicle_type', 'yearOfRegistration':'registration_year', 'powerPS':'power_ps',

              'monthOfRegistration':'registration_month', 'fuelType':'fuel_type',

       'notRepairedDamage':'unrepaired_damage', 'dateCreated':'ad_created', 'nrOfPictures':'num_photos', 'postalCode':'postal_code',

       'lastSeen':'last_seen'}, inplace=True)
autos.head()
autos.describe(include='all')
autos['num_photos'].value_counts()
autos['seller'].value_counts()
autos['offer_type'].value_counts()
autos = autos.drop(['num_photos','seller','offer_type'], axis=1);
autos['price']
autos['odometer']
autos['price'] = autos['price'].str.replace('$','')

autos['price'] = autos['price'].str.replace(',','').astype(int)
autos['odometer'] = autos['odometer'].str.replace('km','')

autos['odometer'] = autos['odometer'].str.replace(',','').astype(int)
autos['price']
autos['odometer']
autos.rename(columns={'odometer':'odometer_km'}, inplace=True)
print(autos['odometer_km'].unique().shape)

print('\n')

print(autos['odometer_km'].describe())

print('\n')

print(autos['odometer_km'].value_counts())

print(autos['price'].unique().shape)

print('\n')

print(autos['price'].describe())

print('\n')

print(autos['price'].value_counts())

autos['price'].value_counts().sort_index(ascending=False).head(20)
autos = autos[autos['price'].between(1,351000)]
autos['price'].describe()
autos[['date_crawled','ad_created','last_seen']][:5]
autos['date_crawled'].str[:10].value_counts(normalize=True, dropna=False).sort_index()
autos['last_seen'].str[:10].value_counts(normalize=True, dropna=False).sort_index()
print(autos['ad_created'].unique().shape)

autos['ad_created'].str[:10].value_counts(normalize=True, dropna=False).sort_index()
autos['registration_year'].describe()
autos = autos[autos['registration_year'].between(1900,2021)]

autos['registration_year'].value_counts().head(10)
autos['brand'].value_counts().head(10)
brand_freq = autos['brand'].value_counts(normalize=True)

common_brand = brand_freq[brand_freq > 0.05]
common_brand
mean_price = {}



for i in common_brand.index:

    srs = autos[autos['brand'] == i]

    price = sum(srs['price']) / srs.shape[0]

    mean_price[i] = price
mean_price
mean_mileage = {}



for i in common_brand.index:

    srs = autos[autos['brand'] == i]

    mileage = sum(srs['odometer_km']) / srs.shape[0]

    mean_mileage[i] = mileage
mean_mileage
meanprice_srs = pd.Series(mean_price)

meanmile_srs = pd.Series(mean_mileage)



df = pd.DataFrame(meanprice_srs, columns=['mean_price'])
df
df['mean_mileage'] = meanmile_srs
df
autos.columns
autos['vehicle_type'].value_counts(dropna=False)
vtype_map = {'bus':'bus', 'limousine':'limousine', 'kleinwagen':'small_car', 'kombi':'station_wagon', 'coupe':'coupe', 'suv':'suv',

       'cabrio':'convertible', 'andere':'others'}



gearbox_map = {'manuell':'manual', 'automatik':'automatic'}



fuel_map = {'lpg':'lpg', 'benzin':'gasoline', 'diesel':'diesel', 'cng':'cng', 'hybrid':'hybrid', 'elektro':'electric',

       'andere':'others'}



damage_map = {'nein':'no', 'ja':'yes'}



autos['vehicle_type'] = autos['vehicle_type'].map(vtype_map)

autos['gearbox'] = autos['gearbox'].map(gearbox_map)

autos['fuel_type'] = autos['fuel_type'].map(fuel_map)

autos['unrepaired_damage'] = autos['unrepaired_damage'].map(damage_map)
autos['vehicle_type'].value_counts()
autos['date_crawled_int'] = autos['date_crawled'].str[:10].str.replace('-','').astype(int)
brand_model = {}

brand = ['peugeot', 'bmw', 'volkswagen', 'smart', 'ford', 'chrysler',

       'seat', 'renault', 'mercedes_benz', 'audi',

       'opel', 'mazda', 'porsche', 'mini', 'toyota', 'dacia', 'nissan',

       'jeep', 'saab', 'volvo', 'mitsubishi', 'jaguar', 'fiat', 'skoda',

       'subaru', 'kia', 'citroen', 'chevrolet', 'hyundai', 'honda',

       'daewoo', 'suzuki', 'trabant', 'land_rover', 'alfa_romeo', 'lada',

       'rover', 'daihatsu', 'lancia']



for i in brand:

    srs = autos[autos['brand'] == i]

    top = srs['model'].value_counts().index[0]

    brand_model[i] = top
brand_model = {}

brand = ['peugeot', 'bmw', 'volkswagen', 'smart', 'ford', 'chrysler',

       'seat', 'renault', 'mercedes_benz', 'audi',

       'opel', 'mazda', 'porsche', 'mini', 'toyota', 'dacia', 'nissan',

       'jeep', 'saab', 'volvo', 'mitsubishi', 'jaguar', 'fiat', 'skoda',

       'subaru', 'kia', 'citroen', 'chevrolet', 'hyundai', 'honda',

       'daewoo', 'suzuki', 'trabant', 'land_rover', 'alfa_romeo', 'lada',

       'rover', 'daihatsu', 'lancia']



for i in brand:

    srs = autos[autos['brand'] == i]

    top = srs['model'].value_counts().index[0]

    brand_model[i] = top
brand_model
odo1 = autos[autos['odometer_km'].between(0,30000)]

odo2 = autos[autos['odometer_km'].between(30000,60000)]

odo3 = autos[autos['odometer_km'].between(60000,90000)]

odo4 = autos[autos['odometer_km'].between(90000,120000)]

odo5 = autos[autos['odometer_km'].between(120000,150000)]



odo1_price = sum(odo1['price']) / odo1.shape[0]

odo2_price = sum(odo2['price']) / odo2.shape[0]

odo3_price = sum(odo3['price']) / odo3.shape[0]

odo4_price = sum(odo4['price']) / odo4.shape[0]

odo5_price = sum(odo5['price']) / odo5.shape[0]
print(odo1_price)

print(odo2_price)

print(odo3_price)

print(odo4_price)

print(odo5_price)
