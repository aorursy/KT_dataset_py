import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # nice plots

import matplotlib.pyplot as plt # basic plotting library

import re # regular expressions



from mpl_toolkits.basemap import Basemap # library for plotting 2D maps in Python



import warnings

warnings.simplefilter('ignore')



import os

print(os.listdir("../input"))
data = pd.read_csv("../input/craigslist-carstrucks-data/craigslistVehicles.csv")
data.head()
data.columns
data.size
cars = data.drop(["url","city_url","title_status","VIN","desc","image_url","size"],axis=1)

cars.size
cars.head()
cars.shape
nans = cars.isnull().sum().sort_values(ascending=False).div(len(cars))

plt.figure(figsize=(16,8))

sns.barplot(x=nans.index, y=nans.values)

plt.title("Percent of missing data")

plt.show()
def cylinders(row):

    if type(row["cylinders"]) is str:

        cyl = re.findall(r"(\d) cylinders", row["cylinders"])

        if len(cyl) != 0:

            return int(cyl[0])

        else:

            return -1

    else:

        return -1

        

data["cylinders"] = data.apply(cylinders,axis=1)
cars["year"].fillna(0, inplace=True)

cars["year"] = cars["year"].astype("int32")
plt.figure(figsize=(20,10))



cars_sub = cars.sample(20000)



m = Basemap(projection='merc', # mercator projection

            llcrnrlat = 20,

            llcrnrlon = -170,

            urcrnrlat = 70,

            urcrnrlon = -60,

            resolution='l')



m.shadedrelief()

m.drawcoastlines() # drawing coaslines

m.drawcountries(linewidth=2) # drawing countries boundaries

m.drawstates(color='b') # drawing states boundaries

#m.fillcontinents(color='grey',lake_color='aqua')



for index, row in cars_sub.iterrows():

    latitude = row['lat']

    longitude = row['long']

    x_coor, y_coor = m(longitude, latitude)

    m.plot(x_coor,y_coor,'.',markersize=0.2,c="red")
#identifying outliers:

price_over_98pct = cars["price"].quantile(.98)



price_yr_cleaned = cars[(1 < cars["price"]) & (cars["price"] < price_over_98pct) & (cars["year"] != 0) & (cars["year"] != 2020)]



plt.figure(figsize=(16,9))

sns.boxplot(x="year", y="price", data = price_yr_cleaned)

plt.title("Price of cars vs. manufacturing year")

plt.ylabel("Price [USD]")



max_year = price_yr_cleaned["year"].max()

min_year = price_yr_cleaned["year"].min()

steps = 2

lab = np.sort(price_yr_cleaned["year"].unique())[::2]

pos = np.arange(0,111,2)



plt.xticks(ticks=pos, labels=lab, rotation=90)

plt.show()
no_of_offers = price_yr_cleaned.groupby("year")["price"].count()



plt.figure(figsize=(16,8))

sns.barplot(x = no_of_offers.index, y = no_of_offers.values)

plt.title("Numbers of offers for each manufacturing year")

plt.ylabel("Count of offers")

plt.xticks(rotation=90)

plt.show()
manufacturers = cars["manufacturer"].value_counts().div(len(cars)).mul(100)

manufactuters_TOP20 = manufacturers[:20]



plt.figure(figsize=(16,8))

sns.barplot(x=manufactuters_TOP20.index, y=manufactuters_TOP20.values)

plt.title("20 most popular manufactureres in the USA")

plt.ylabel("Popularity in %")

plt.xticks(rotation=90)

plt.show()
cars_odo_clear = cars[(cars["odometer"]<cars["odometer"].quantile(.99)) & ((cars["price"]<cars["price"].quantile(.99)))& ((cars["price"]>1))]
plt.figure(figsize=(16,8))

sns.scatterplot(x = 'odometer', y = 'price', data = cars_odo_clear)

plt.show()