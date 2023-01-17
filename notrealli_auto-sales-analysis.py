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
autos = pd.read_csv('/kaggle/input/used-cars-database-50000-data-points/autos.csv', encoding='Latin-1')
autos.head()
autos.info()
autos.columns
autos.rename({'yearOfRegistration': 'registration_year',
             'monthOfRegistration': 'registration_month',
             'notRepairedDamage': 'unrepaired_damage',
             'dateCreated': 'date_created',
             'dateCrawled': 'date_crawled',
             'offerType': 'offer_type',
             'vehicleType': 'vehicle_type',
             'powerPS': 'power_ps',
             'fuelType': 'fuel_type',
             'nrOfPictures': 'num_pictures',
             'postalCode': 'postal_code',
             'lastSeen': 'last_seen'}, axis=1, inplace=True)
autos.columns
# numerical columns
autos.describe()
# string columns
autos.describe(include='object')
# or .unique()
autos['price'].value_counts().sort_index()
autos['price'] = autos['price'].str.replace('$', '').str.replace(',', '')
autos['price'] = autos['price'].astype(float)
# numerical columns
autos.describe()
autos['odometer'].value_counts().sort_index(key=lambda x: x.str.replace('km', '').str.replace(',', '').astype(int))
autos['odometer'] = autos['odometer'].str.replace('km', '').str.replace(',', '')
autos['odometer'] = autos['odometer'].astype(int)
# add unit to the name of odometer column
autos.rename(columns={'odometer': 'odometer_km'}, inplace=True)
autos.describe()
autos['odometer_km'].unique().shape
autos['odometer_km'].describe()
autos['odometer_km'].value_counts().sort_index()
autos['price'].unique().shape
autos['price'].describe()
autos['price'].value_counts()
mid_50 = autos['price'].quantile([.25, .75])
price_25, price_75 = mid_50.iloc[0], mid_50.iloc[1]

iqr = price_75 - price_25
low = price_25 - 1.5 * iqr
high = price_75 + 1.5 * iqr

low, iqr, high
autos = autos.loc[autos['price'] < high]
print('Count: ', autos.shape[0], '\nMin price: ', autos['price'].min(), '\nMax price:', autos['price'].max())
string_dates = autos[['date_crawled', 'last_seen', 'date_created']]
string_dates.head()
date_created = string_dates['date_created'].str[:10]
date_created.value_counts(normalize=True, dropna=False).sort_index()
date_created.describe()
date_created.str[:4].astype(int).describe()
last_seen = string_dates['last_seen'].str[:10]
last_seen.value_counts(normalize=True, dropna=False).sort_index()
last_seen.describe()
last_seen.str[:4].astype(int).describe()
date_crawled = string_dates['date_crawled'].str[:10]
date_crawled.value_counts(normalize=True, dropna=False).sort_index()
date_crawled.describe()
date_crawled.str[:4].astype(int).describe()
autos['registration_year'].describe()
autos['registration_year'].astype(str).describe()
print("Number of entries before: ", autos.shape[0])
# Firstly, the registration year can be later than 2016
year_high = 2016

# Secondly, let's assume the earliest year in which a car can be registered was the year 1885 (when the first car was invented)
year_low = 1885

autos = autos[autos['registration_year'].between(1885, 2016)]

# Lastly, let's say an ad can't be posted before the car was registered
autos = autos[autos['registration_year'] <= autos['date_created'].str[:4].astype(int)]

print("Number of entries after: ", autos.shape[0])
autos['registration_year'].describe()
autos['registration_year'].value_counts(normalize=True, dropna=False).sort_index()
top_5_brands_list = list(autos['brand'].value_counts(normalize=True).head(6).index)
top_5_brands_list
top_prices_mileage_info = {}


for brand in top_5_brands_list:
    mean_price = autos.loc[autos['brand'] == brand, 'price'].mean()
    min_price = autos.loc[autos['brand'] == brand, 'price'].min()
    max_price = autos.loc[autos['brand'] == brand, 'price'].max()
    mileage = autos.loc[autos['brand'] == brand, 'odometer_km'].mean()
    
    top_prices_mileage_info[brand] = ["$" + str(round(mean_price, 2)),
                                      "$" + str(round(min_price, 2)),
                                      "$" + str(round(max_price, 2)),
                                      round(mileage, 2)]
    

for key, val in top_prices_mileage_info.items():
    print(key, ":")
    print('Average price: ', val[0])
    print('Min price: ', val[1])
    print('Max price: ', val[2])
    print('Average mileage', val[3])
    print('\n')
price_mileage = pd.DataFrame(data=top_prices_mileage_info, index=['mean_price', 'min_price', 'max_price', 'mean_mileage'])
price_mileage = price_mileage.swapaxes('index', 'columns')
price_mileage