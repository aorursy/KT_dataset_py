import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("../input/price-of-agricultural-raw-materials/agric_raw_material.csv")
df.head()
df.shape
df.describe()
df.info()
df.tail()
# Convert month to date time format
df.iloc[:, 0] = pd.to_datetime(df.iloc[:,0], format='%b-%y')

# Convert percent change to float
# Converting price to float32 probably saves memory too
for x in range(1, 25):
        df.iloc[:, x] = pd.to_numeric(df.iloc[:, x], errors='coerce', downcast='float')

df.info()
df.describe()
df = df.set_index('Month')
df.index
# Price chart
df.iloc[:, 0:12:2].plot(kind='line', figsize=(14,6))
# Price chart continued
df.iloc[:, 12::2].plot(kind='line', figsize=(14,6))
# Month range and format
df.index
# Check out the list of columns 
df.columns
# Change this
plant = "Plywood"        # Don't include " Price" and " price % Change". It is added below in "# Don't change this" section
start_date = "2010-03"   # year[-month[-day]]
stop_date = "2019-08"


# Don't change this
price = df.loc[start_date:stop_date, plant+' Price']
price_change = df.loc[start_date:stop_date, plant+' price % Change']
price.describe()
# Price chart
title = "Price chart for {} between {} and {}".format(plant, start_date, stop_date)
price.plot(kind='line', figsize=(14,6), title=title)
price_change.describe()
# % change chart
title = "Price % change for {} between {} and {}".format(plant, start_date, stop_date)
price_change.plot(kind='line', figsize=(14,6), title=title)
# maximum and minimum price
max_price_date = price[price == price.max()]
min_price_date = price[price == price.min()]
print(max_price_date, '\n\n', min_price_date)
# maximum and minimum price % change
max_change_date = price_change[price_change == price_change.max()]
min_change_date = price_change[price_change == price_change.min()]
print(max_change_date, '\n\n', min_change_date)
# Check out the list of columns 
df.columns
plant = "Plywood"        # Don't include " Price" and " price % Change". It is added below in "# Don't change this" section

# Year in range(1991-2020). 1990 and 2020 is incomplete
start_year = "2010"  
stop_year = "2019"


# Don't change this
price = df.loc[start_year:stop_year, plant+' Price']
change = df.loc[start_year:stop_year, plant+' price % Change']
annual_price = price.groupby(price.index.year).mean().round(2)
annual_price
annual_change = change.groupby(price.index.year).mean().round(2)
annual_change
title = "Annual Price of {} from {} to {}".format(plant, start_year, stop_year)
annual_price.plot(kind='line', figsize=(14,6), title=title)
title = "Annual Price % change of {} from {} to {}".format(plant, start_year, stop_year)
annual_change.plot(kind='line', figsize=(14,6), title=title)
cases = df.loc["2017-09":"2020-04"].dropna(axis=1, thresh=30)
cases.iloc[:,::2].plot(figsize=(14,6), title="Prices of raw materials from 2017-09 to 2020-04")
