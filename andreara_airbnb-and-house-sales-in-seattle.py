# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import geopandas as gpd

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#loading the dataset and making sure that the zip codes are the same data type for merging

air = pd.read_csv('../input/seattle/listings.csv')

sales = pd.read_csv("../input/housesalesprediction/kc_house_data.csv")

air.zipcode = air.zipcode.astype("str")

sales.zipcode = sales.zipcode.astype("str")
air.info()
sales.info()
#count of airbnb by zip

airZipCount = air.groupby(['zipcode']).id.count()

airZipCount = airZipCount.to_frame(name="count")

airZipCount.head(5)
#count of sales by zip code

salesZipCount = sales.groupby(['zipcode']).id.count()

salesZipCount = salesZipCount.to_frame(name="count")

salesZipCount.head(5)
#this steps gets all the zip codes from both sets, and merge them.  These are all the zipcodes we know of

#the step might not be strictly necessary and it could be cleaned up further.

zips = pd.concat([sales.zipcode, air.zipcode])

zips = zips.unique()

zips
#makes the array into a dataframe

zipsDF = pd.DataFrame(columns = ['zipcode'])

zipsDF.zipcode=zips

zipsDF.head(5)
#join the 3 datasets together

joined = zipsDF.join(salesZipCount, lsuffix="Zip", rsuffix='sales', on='zipcode', how="left")

joined = joined.join(airZipCount, lsuffix="Sales", rsuffix='Air', on='zipcode', how="left")



#joined = airZipCount.join(salesZipCount, lsuffix="Zip", rsuffix='sales', on='zipcode')

joined.info()
joined.head(5)
#histogram of airbnb and sales for curiosity

f, (ax, bx) = plt.subplots(2,figsize=(10,5), sharex=True)

ax.set_title("Histo of AirBnb counts")

bx.set_title("Histo of Sales counts")

ax.hist(joined.countAir, bins=30)

bx.hist(joined.countSales, bins=30)

plt.show()
#remove the na to just see the zip that have both sales and airbnb

joined = joined.dropna()
#map the count of sales with the count of airbnb

f, ax = plt.subplots(figsize=(30,15))

ax.set_title("AirBnb X Sales")

sns.set_context("talk", font_scale=1, rc={"font.size":12,"axes.labelsize":16})

sns.scatterplot(x="countAir", y="countSales", data=joined, ax=ax)



#add annotations to see the zip codes

for txt in joined.zipcode:

    ax.annotate(txt, (joined[joined.zipcode==txt].countAir+3, joined[joined.zipcode==txt].countSales+3),size=14);
#calculat airbnb to sales ratio

joined["ratio"] = joined.countAir/joined.countSales

joined.head(5)
#lets map

#the file comes from http://data-seattlecitygis.opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2

sf = "../input/seattle-zip-shapefile/Zip_Codes.shp"

map_df = gpd.read_file(sf)
#all of king county

map_df.plot();
#how does the data look like

map_df.info()
#merge the map data with our data

tomap = map_df.set_index('ZIPCODE').join(joined.set_index('zipcode'))

tomap.head(5)
#create two maps with our data, one for the whole county and one for only the zip code where we have data

fig, ax = plt.subplots(1,2, figsize=(20, 10))

ax[0].axis('off') 

ax[1].axis('off') 

ax[0].set_title('Kind County') 

ax[1].set_title('Zips with data') 

tomap.plot(column='ratio', cmap='Blues', linewidth=0.8, ax=ax[0], edgecolor='0.8')

tomap.dropna().plot(column='ratio', cmap='Blues', linewidth=0.8, ax=ax[1], edgecolor='0.8');