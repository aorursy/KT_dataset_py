import pandas as pd

import numpy as np



autos = pd.read_csv('../input/used-cars-database-50000-data-points/autos.csv', encoding = 'Latin-1')



autos.info()

autos.head()
autos.columns
autos.rename(columns = {'yearOfRegistration':'registration_year', 'monthOfRegistration':'registration_month', 

                       'notRepairedDamage':'unrepaired_damage', 'dateCreated':'ad_created'}, inplace = True)
import re



def underscore(word):

    """

    Make an underscored, lowercase form from the expression in the string.



    Example::



        >>> underscore("DeviceType")

        "device_type"



    As a rule of thumb you can think of :func:`underscore` as the inverse of

    :func:`camelize`, though there are cases where that does not hold::



        >>> camelize(underscore("IOError"))

        "IoError"



    """

    word = re.sub(r"([A-Z]+)([A-Z][a-z])", r'\1_\2', word)

    word = re.sub(r"([a-z\d])([A-Z])", r'\1_\2', word)

    word = word.replace("-", "_")

    return word.lower()



for item in autos.columns:

    col_name = item

    col_name = underscore(col_name)

    autos.rename(columns = {item : col_name}, inplace = True)

    

autos.columns
autos.describe(include = 'all')
autos['price'] = autos['price'].str.replace('$','')

autos['price'] = autos['price'].str.replace(',','')

autos['price'] = autos['price'].astype(float)

autos['odometer'] = autos['odometer'].str.replace('km','')

autos['odometer'] = autos['odometer'].str.replace(',','')

autos['odometer'] = autos['odometer'].astype(int)

autos.rename(columns = {'odometer' : 'odometer_km'}, inplace = True)

autos.head()
from scipy import stats



zp = np.abs(stats.zscore(autos['price']))

print(zp)
for item in np.where(zp > 3):

    print(autos['price'].iloc[item])
autos['price'].min()
autos = autos[autos['price'].between(0, 3890000, inclusive = False)]

autos.describe()
autos.shape
zo = np.abs(stats.zscore(autos['odometer_km']))

print(zo)
for item in np.where(zo > 3):

    print(autos['odometer_km'].iloc[item])
autos = autos[autos['odometer_km'].between(5000, 150001, inclusive = False)]

autos.shape
autos[['date_crawled', 'ad_created', 'last_seen']][0:5]
autos[['date_crawled', 'ad_created', 'last_seen']].isnull().sum()
autos['date_crawled'].str[:10].value_counts(normalize = True).sort_index()
autos['ad_created'].str[:10].value_counts(normalize = True).sort_index()
autos['last_seen'].str[:10].value_counts(normalize = True).sort_index()
autos['registration_year'].describe()
zr = np.abs(stats.zscore(autos['registration_year']))

for item in np.where(zr > 3):

    print(autos['registration_year'].iloc[item])
autos = autos[autos['registration_year'].between(1910, 2016)]

autos.shape
autos['registration_year'].value_counts(normalize = True).sort_index()
brand_count = autos['brand'].value_counts(normalize = True)

common_brand = brand_count[brand_count > 0.05].index

print(common_brand)
autos['brand'].value_counts(normalize = True)
mean_by_brand = {}



for brand in common_brand:

    brand_df = autos.loc[autos['brand'] == brand]

    mean_val = brand_df['price'].mean()

    mean_by_brand[brand] = int(mean_val)

    

print(mean_by_brand)
mileage_by_brand = {}



for brand in common_brand:

    brand_df = autos.loc[autos['brand'] == brand]

    avg_val = brand_df['odometer_km'].mean()

    mileage_by_brand[brand] = int(avg_val)



print(mileage_by_brand)
mean_series = pd.Series(mean_by_brand).sort_values(ascending = False)

print(mean_series)
mileage_series = pd.Series(mileage_by_brand).sort_values(ascending = False)

print(mileage_series)
mmb = pd.DataFrame(mean_series, columns = ['mean_price'])

mmb['average_mileage'] = mileage_series

mmb