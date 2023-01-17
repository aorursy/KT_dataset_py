import numpy as np

import pandas as pd
auto_filepath = "../input/used-cars-database-50000-data-points/autos.csv"
autos = pd.read_csv(auto_filepath)
# Attempt to use Latin-1 encoding to resolve UnicodeDecodeError

autos = pd.read_csv(auto_filepath, encoding = "Latin-1")
autos.head()
autos.info()
auto_columns = autos.columns.to_series()

columns_dict = {"yearOfRegistration": "registration_year", "monthOfRegistration": "registration_month",

               "notRepairedDamage": "unrepaired_damage", "dateCreated": "ad_created"}

# First replace clunky column names

auto_columns.replace(columns_dict, inplace = True)

# Use a regular expression to insert underscores in between camelcase letters

auto_columns = auto_columns.str.replace(r"([a-z])([A-Z])", lambda m: "_".join(m.group(1, 2)))

# Make all column names lowercase

auto_columns = auto_columns.str.lower()

auto_columns
autos.columns = auto_columns

autos.head()
autos.describe(include = "all")
autos["seller"].value_counts()
autos["offer_type"].value_counts()
autos["nr_of_pictures"].value_counts()
# Calculate the most recent date an observation was collected.

pd.to_datetime(autos["date_crawled"]).max()
# Count the number of cars with registration years prior to 1906 or after 2016

autos.loc[~autos["registration_year"].between(1906, 2016), "registration_year"].count()
autos.loc[~autos["registration_year"].between(1906, 2016), "registration_year"].value_counts()
# Count the number of cars with power rating less than 1ps or greater than 2000ps

autos.loc[~autos["power_ps"].between(1, 2000), "power_ps"].count()
autos.loc[~autos["power_ps"].between(1, 2000), "power_ps"].value_counts()
autos["registration_month"].value_counts()
# First strip leading $

cleaned_price = autos["price"].str.strip("$")

# Remove commas

cleaned_price = cleaned_price.str.replace(",", "")

# Convert to float64 dtype

cleaned_price = cleaned_price.astype(float)

autos["price"] = cleaned_price
# First strip trailing "km"

cleaned_odometer = autos["odometer"].str.strip("km")

# Remove commas

cleaned_odometer = cleaned_odometer.str.replace(",", "")

# Convert to float64

cleaned_odometer = cleaned_odometer.astype(float)

autos["odometer"] = cleaned_odometer
autos.rename(columns = {"odometer": "odometer_km"}, inplace = True)
autos["odometer_km"].nunique()
autos["odometer_km"].describe()
autos["odometer_km"].value_counts()
(autos["odometer_km"] >= 100000).sum()
autos["price"].nunique()
np.round(autos["price"].describe(), 2)
# Get value counts for the prices that are less than 0.01 and greater than 4000000

autos.loc[~autos["price"].between(0.01, 4000000), "price"].value_counts().sort_index()
# Count the number of cars with prices less than 0.01 and greater than 4000000

autos.loc[~autos["price"].between(0.01, 4000000), "price"].count()
np.round(autos.loc[autos["price"].between(0.01, 4000000), "price"].describe(), 2)
autos.loc[autos["price"].between(0.01, 4000000), "price"].value_counts().sort_index().head(25)
autos.loc[autos["price"].between(0.01, 4000000), "price"].value_counts().sort_index().tail(25)
autos.loc[autos["price"] > 350000, ["name", "model", "brand", "registration_year", "price"]]
autos[["date_crawled", "ad_created", "last_seen"]].head()
autos[["date_crawled", "ad_created", "last_seen"]] = autos[["date_crawled", "ad_created", "last_seen"]].apply(pd.to_datetime)
autos[["date_crawled", "ad_created", "last_seen"]].info()
autos["date_crawled"].dt.floor("D").value_counts(normalize = True).sort_index()
autos["date_crawled"].dt.hour.value_counts(normalize = True).sort_index()
autos["ad_created"].dt.floor("D").value_counts(normalize = True).sort_index().head()
autos["ad_created"].dt.floor("D").value_counts(normalize = True).sort_index().tail()
autos["ad_created"].dt.year.value_counts(normalize = True)
autos.loc[autos["ad_created"].dt.year == 2016, "ad_created"].dt.month.value_counts(normalize = True).sort_index()
autos["ad_created"].dt.hour.value_counts(normalize = True).sort_index()
autos["ad_created"].dt.time.value_counts(normalize = True).sort_index()
autos["last_seen"].dt.floor("D").value_counts(normalize = True).sort_index()
autos["last_seen"].dt.floor("D").value_counts(normalize = True).sort_index().tail().sum()
autos["registration_month"].value_counts(normalize = True).sort_index()
autos["registration_year"].describe()
# Calculate the most recent date an observation was collected.

autos["date_crawled"].max()
# Count the number of cars with registration years prior to 1906 or after 2016

autos.loc[~autos["registration_year"].between(1906, 2016), "registration_year"].count()
autos.loc[~autos["registration_year"].between(1906, 2016), "registration_year"].value_counts()
autos.loc[autos["registration_year"].between(1906, 2016), "registration_year"].describe()
autos.loc[autos["registration_year"].between(1906, 2016), "registration_year"].quantile(0.1)
autos.loc[autos["registration_year"].between(1995, 2016), "registration_year"].value_counts(normalize = True).sort_index()
autos.loc[autos["registration_year"].between(1906, 1994), "registration_year"].value_counts(normalize = True).sort_index()
autos.loc[autos["registration_year"].between(1906, 1994), "registration_year"].value_counts(normalize = True).sort_index().loc[1970:].sum()
autos.loc[autos["registration_year"].between(1906, 2016), "registration_year"].value_counts(normalize = True).sort_index().loc[1970:].sum()
# Drop seller, offer_type, nr_of_pictures columns

autos.drop(columns = ["seller", "offer_type", "nr_of_pictures"], inplace = True)
# Filter out rows with impossible registration year 

# and impossible/unrealistic price

price_year_filter = autos["registration_year"].between(1906, 2016) & autos["price"].between(0.01, 4000000)

autos = autos[price_year_filter]
autos["vehicle_type"].value_counts()
vehicle_types = {"andere": "other", "bus": "van", "cabrio": "convertible",

                "coupe": "coupe", "kleinwagen": "subcompact", "kombi": "station wagon",

                "limousine": "sedan", "suv": "suv"}

autos["vehicle_type"] = autos["vehicle_type"].map(vehicle_types)
autos["gearbox"].value_counts()
autos["gearbox"] = autos["gearbox"].map({"manuell": "manual", "automatik": "automatic"})
autos["fuel_type"].value_counts()
fuel_types = {"andere": "other", "benzin": "gasoline", "cng": "cng", "diesel": "diesel",

             "elektro": "electric", "hybrid": "hybrid", "lpg": "lpg"}

autos["fuel_type"] = autos["fuel_type"].map(fuel_types)
autos["unrepaired_damage"].value_counts()
autos["unrepaired_damage"] = autos["unrepaired_damage"].map({"ja": "yes", "nein": "no"})
autos.head()
autos["brand"].nunique()
autos["brand"].value_counts(normalize = True)
autos["brand"].value_counts(normalize = True).head(10).sum()
top_10_brands = autos["brand"].value_counts().head(10).index
# Compute mean of odometer_km and price columns

# Round to the nearest whole number for easier interpretation

# Restrict focus to the top ten most common brands

# Sort from highest mean price to lowest

autos.groupby("brand")[["odometer_km", "price"]].mean().loc[top_10_brands].sort_values("price", ascending = False).round(0)
# Compute median of price column

# Round to the nearest whole number for easier interpretation

# Restrict focus to the top ten most common brands

# Sort from highest median price to lowest

autos.groupby("brand")["price"].agg(["min", "median", "max"]).loc[top_10_brands].sort_values("median", ascending = False).round(0)
autos.groupby("odometer_km")["price"].describe().sort_index().round(2)
autos.loc[autos["odometer_km"] < 50000, "price"].mean()
autos.fillna({"unrepaired_damage": "unknown"}).groupby("unrepaired_damage")["price"].describe().round(2)
# Group by brand and then get the value counts for the models in each brand

# Include null values in the value counts

# Then sort by values in descending order

autos.groupby("brand")["model"].value_counts(dropna = False).sort_values(ascending = False).head(10)
# Flatten multi-indexed series

flat_brand_model = autos.groupby("brand")["model"].value_counts(dropna = False).reset_index(name = "count")

flat_brand_model["proportion"] = flat_brand_model["count"]/autos.shape[0]

flat_brand_model.sort_values(["proportion"], ascending = False).head(10)
flat_brand_model["proportion"].sort_values(ascending = False).head(10).sum()