import pandas as pd
import numpy as np
autos = pd.read_csv(r"../input/autos.csv", encoding="Windows-1252", nrows=50000)
autos.isnull().sum()
cols = ["date_crawled", "name", "seller", "offer_type", "dollar_price", "ab_test",
        "vehicle_type", "registration_year", "gearbox", "power_ps", "model",
       "kilometer", "registration_month", "fuel_type", "brand", "unrepaired_damage",
        "ad_created", "pictures_number", "postal_code",  "last_seen_online"]
autos.columns = cols
autos.head()
print(autos["pictures_number"].sum())
print(autos["ab_test"].unique())
print(autos["offer_type"].unique())
print(autos['seller'].unique())
autos.drop(["pictures_number", "ab_test", "offer_type", "seller"], axis=1, inplace=True)
autos.info()
autos.describe()
autos.dollar_price = autos.dollar_price.astype(np.uint32)
autos.kilometer = autos.kilometer.astype(np.uint32)
autos.power_ps = autos.power_ps.astype(np.uint16)
autos.registration_month = autos.registration_month.astype(np.uint8)
autos.registration_year = autos.registration_year.astype(np.uint16)
autos.postal_code = autos.postal_code.astype(np.uint32)
autos.info()
print(autos.dollar_price.value_counts().sort_index().head())
print(autos.dollar_price.unique().shape)
c_dollar_prices = autos.dollar_price.copy()
s_dollar_prices = c_dollar_prices.sort_values(ascending=False)
s_dollar_prices.index = autos.index
print(s_dollar_prices.head())
v_dollar_prices = autos[autos["dollar_price"] > 100000].copy()
v_dollar_prices.sort_values(by=['dollar_price'], ascending=False, inplace=True)
v_dollar_prices[['name', 'kilometer', 'dollar_price']].head(20)
v_dollar_prices._is_copy is None
autos.dollar_price.value_counts().sort_index().head(20)
print((~autos["dollar_price"].between(500,245000)).sum() / autos.shape[0])
autos = autos[autos["dollar_price"].between(500,245000)]
v_dollar_prices = autos[autos["dollar_price"] >= 50000].copy()
v_dollar_prices.sort_values(by=['dollar_price'], ascending=False, inplace=True)
v_dollar_prices.head(20)
autos.loc[(autos["vehicle_type"] == "kombi") & (autos["dollar_price"] > 50000), "dollar_price"]
autos.index
autos.index = range(45081)
print(autos['vehicle_type'].unique(), autos['fuel_type'].unique())
mapping_vehicle_type = {
    'coupe': 'coupé',
    'kleinwagen': 'small car',
    'kombi': 'station wagon',
    'cabrio': 'convertible',
    'andere': 'other',
    'hybrid':'hybrid',
    'limousine':'limousine',
    'bus':'bus'
                        }

mapping_fuel_type = {
    'benzin': 'gasoline',
    'diesel': 'diesel',
    'lpg': 'liquefied petroleum gas',
    'andere': 'other',
    'cng': 'compressed natural gas',
    'elekto':'electric'
                    }

autos['vehicle_type'] = autos['vehicle_type'].map(mapping_vehicle_type)
autos['fuel_type'] = autos['fuel_type'].map(mapping_fuel_type)
autos[['date_crawled','ad_created','last_seen_online']].head()
date_col = ['date_crawled', 'ad_created', 'last_seen_online']
for col in date_col:
    autos[col] = autos[col].str[:10]
autos['date_crawled'].value_counts(normalize=True).sort_index(ascending=True)
a = autos['ad_created'].value_counts(normalize=True).sort_index(ascending=False)
b = a[a > 0.02]
b
autos['last_seen_online'].value_counts(normalize=True).sort_index(ascending=False)
(autos['ad_created'] <= autos['last_seen_online']).all()
a = (autos['registration_year'] > 2016)
b = a[a == True]
b.count()
cars_by_year = autos['registration_year'].value_counts().sort_index(ascending=False)
autos.loc[autos["registration_year"].between(2017,2018), "registration_year"] = 2016
print(cars_by_year.head(12))
print(cars_by_year.tail(12))
(~autos["registration_year"].between(1945,2016)).sum() / autos.shape[0]
# Many ways to select rows in a dataframe that fall within a value range for a column.
# Using `Series.between()` is one way.
autos = autos[autos["registration_year"].between(1945,2016)]
autos["registration_year"].value_counts(normalize=True).head(10)
p_brands = autos['brand'].value_counts(normalize=True)
p_brands
brand_mean_price = {}
brands = p_brands[p_brands > 0.01].index
for b in brands:
    mean_price = int(autos.loc[autos.brand == b, "dollar_price"].mean())
    mean_mileage = int(autos.loc[autos.brand == b, "kilometer"].mean())
    mean_year = int(autos.loc[autos.brand == b, "registration_year"].mean())
    brand_mean_price[b] = [mean_price, mean_mileage, mean_year]
df = pd.DataFrame.from_dict(brand_mean_price, orient='index', columns=['mean_price', 'mean_mileage', 'mean_year'])
df.head()