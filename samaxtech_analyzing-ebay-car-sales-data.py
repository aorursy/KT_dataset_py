%autosave 2

import pandas as pd
import numpy as np
import csv
autos = pd.read_csv('../input/autos.csv', encoding='Latin-1')
print(autos.info())
autos.head()
autos.columns
autos.columns = ['date_crawled', 'name', 'seller', 'offer_type', 'price', 'abtest',
       'vehicle_type', 'registration_year', 'gearbox', 'power_PS', 'model',
       'odometer', 'registration_month', 'fuel_type', 'brand',
       'unrepaired_damage', 'ad_created', 'number_of_pictures', 'postal_code',
       'date_last_seen']
autos.head()
autos.describe(include='all')
#'price' column
autos["price"] = autos["price"].str.replace("$","").str.replace(",","")
autos["price"] = autos["price"].str.strip().astype(float)

#'odometer' column
autos["odometer"] = autos["odometer"].str.replace("km","").str.replace(",","")
autos["odometer"] = autos["odometer"].str.strip().astype(float)

#Rename both
autos = autos.rename(columns={"price": "price_dollars","odometer": "odometer_km"})

autos.head()
autos["price_dollars"].describe()
autos["odometer_km"].describe()
autos["price_dollars"].value_counts(ascending=False)
price_interval = [100,100000]

autos = autos.loc[autos["price_dollars"].between(price_interval[0],price_interval[1])]
autos["odometer_km"].value_counts(ascending=False)
autos.drop('number_of_pictures', axis=1, inplace=True)
date_crawled_dist = autos["date_crawled"].value_counts(normalize=True, dropna=False)
ad_created_dist = autos["ad_created"].value_counts(normalize=True, dropna=False)
last_seen_dist = autos["date_last_seen"].value_counts(normalize=True, dropna=False)

print(date_crawled_dist,ad_created_dist,last_seen_dist)
sorted_date_crawled = date_crawled_dist.sort_index()
sorted_ad_created = ad_created_dist.sort_index()
sorted_last_seen = last_seen_dist.sort_index()

print(sorted_date_crawled,sorted_ad_created,sorted_last_seen)
autos["registration_year"].describe()
autos.loc[autos["registration_year"].between(1900,2016)].shape
autos.drop(autos[~autos["registration_year"].between(1900,2016)].index, inplace=True)
autos["registration_year"].value_counts(normalize=True)
autos["brand"].value_counts(normalize=True)>0.05
aggregate_brands = ['volkswagen','bmw','opel','mercedes_benz','audi','ford']
brand_mean_prices = {}
brand_mean_mileage = {}

for brand in aggregate_brands:
    analyzed_brand = autos.loc[autos["brand"]==brand]
    mean_price = analyzed_brand["price_dollars"].mean()
    mean_mileage = (analyzed_brand["odometer_km"].mean())*0.621371 #Convert to miles
    brand_mean_prices[brand] = mean_price
    brand_mean_mileage[brand] = mean_mileage
    
print(brand_mean_prices,"\n\n",brand_mean_mileage)
brand_mean_prices = pd.Series(brand_mean_prices)
brand_mean_mileage = pd.Series(brand_mean_mileage)

new_df = pd.DataFrame(brand_mean_prices, columns=["mean_price"])
new_df["mean_mileage"] = brand_mean_mileage

new_df
autos.head()
#'seller' column
autos["seller"].unique()
autos.loc[autos["seller"]=='privat',"seller"] = 'private'
autos.loc[autos["seller"]=='gewerblich',"seller"] = 'commercial'
#'offer_type' column
autos["offer_type"].unique()
autos["offer_type"] = 'offer'
#'offer_type' column
autos["vehicle_type"].unique()
autos.loc[autos["vehicle_type"]=='kleinwagen',"vehicle_type"] = 'compact car'
autos.loc[autos["vehicle_type"]=='kombi',"vehicle_type"] = 'caravan'
autos.loc[autos["vehicle_type"]=='andere',"vehicle_type"] = 'other'
#'gearbox' column
autos["gearbox"].unique()
autos.loc[autos["gearbox"]=='manuell',"gearbox"] = 'manual'
autos.loc[autos["gearbox"]=='automatik',"gearbox"] = 'automatic'
#'model' column
autos.loc[autos["model"]=='andere',"model"] = 'other'
#'fuel_type' column
autos["fuel_type"].unique()
autos.loc[autos["fuel_type"]=='benzin',"fuel_type"] = 'gasoline'
autos.loc[autos["fuel_type"]=='elektro',"fuel_type"] = 'electric'
autos.loc[autos["fuel_type"]=='andere',"fuel_type"] = 'other'
#'unrepaired_damage' column
autos["unrepaired_damage"].unique()
autos.loc[autos["unrepaired_damage"]=='nein',"unrepaired_damage"] = 'no'
autos.loc[autos["unrepaired_damage"]=='ja',"unrepaired_damage"] = 'yes'
autos.head()
#Separate date and time into two different columns
autos["date_crawled"] = pd.Series(autos["date_crawled"]).astype(object).astype(str)
autos["ad_created"] = pd.Series(autos["ad_created"]).astype(object).astype(str)
autos["date_last_seen"] = pd.Series(autos["date_last_seen"]).astype(object).astype(str)

autos[["date_crawled","time_crawled"]] = autos["date_crawled"].str.split(expand=True)
autos[["ad_created","time_ad_created"]] = autos["ad_created"].str.split(expand=True)
autos[["date_last_seen","time_last_seen"]] = autos["date_last_seen"].str.split(expand=True)
autos.drop("time_ad_created", axis=1, inplace=True)
autos["date_crawled"] = autos["date_crawled"].str.replace("-","").astype(int)
autos["ad_created"] = autos["ad_created"].str.replace("-","").astype(int)
autos["date_last_seen"] = autos["date_last_seen"].str.replace("-","").astype(int)
autos
aux_df = pd.concat([autos["brand"],autos["model"]], axis=1)
aux_df = aux_df.groupby(['brand','model']).size().reset_index().rename(columns={0:'count'})
aux_df
aux_df.sort_values('count', ascending=False)
g1 = autos.loc[autos["odometer_km"]<50000,'odometer_km']
g2 = autos.loc[autos["odometer_km"].between(50000,100000),'odometer_km']
g3 = autos.loc[autos["odometer_km"].between(100000,125000),'odometer_km']
g4 = autos.loc[autos["odometer_km"].between(125000,150000),'odometer_km']
mileage_avg_prices = {}
groups = [g1,g2,g3,g4]
group_number = 1

for group in groups:
    av_price = autos.loc[group.index, "price_dollars"].mean()
    mileage_avg_prices['g'+str(group_number)] = av_price
    group_number+=1
    
mileage_avg_prices
    
#Find what models are/are not damaged
damaged_models = autos.loc[autos['unrepaired_damage']=='yes',"model"]
non_damaged_models = autos.loc[autos['unrepaired_damage']=='no',"model"]

#Get the price for those models
price_damaged = autos.loc[damaged_models.index,"price_dollars"]
non_damaged_price = autos.loc[non_damaged_models.index,"price_dollars"]

print("\n\nDAMAGED:\n\n",price_damaged.describe(),"\n\n\nNON-DAMAGED:\n\n",non_damaged_price.describe())
