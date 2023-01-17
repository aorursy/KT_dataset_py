import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

#print(os.listdir("../input"))

import warnings

warnings.filterwarnings("ignore")
hsales = pd.read_csv('../input/nyc-property-sales/nyc-rolling-sales.csv') 
hsales.shape
# let's check what we have 

hsales.head()
hsales.drop(['Unnamed: 0', 'EASE-MENT'],1, inplace=True)
hsales.info()
#First, let's check which columns should be categorical

print('Column name')

for col in hsales.columns:

    if hsales[col].dtype=='object':

        print(col, hsales[col].nunique())
# LAND SQUARE FEET,GROSS SQUARE FEET, SALE PRICE, BOROUGH should be numeric. 

# SALE DATE datetime format.

# categorical: NEIGHBORHOOD, BUILDING CLASS CATEGORY, TAX CLASS AT PRESENT, BUILDING CLASS AT PRESENT,

# BUILDING CLASS AT TIME OF SALE, TAX CLASS AT TIME OF SALE,BOROUGH 



numer = ['LAND SQUARE FEET','GROSS SQUARE FEET', 'SALE PRICE', 'BOROUGH']

for col in numer: # coerce for missing values

    hsales[col] = pd.to_numeric(hsales[col], errors='coerce')



categ = ['NEIGHBORHOOD', 'BUILDING CLASS CATEGORY', 'TAX CLASS AT PRESENT', 'BUILDING CLASS AT PRESENT', 'BUILDING CLASS AT TIME OF SALE', 'TAX CLASS AT TIME OF SALE']

for col in categ:

    hsales[col] = hsales[col].astype('category')



hsales['SALE DATE'] = pd.to_datetime(hsales['SALE DATE'], errors='coerce')
missing = hsales.isnull().sum()/len(hsales)*100



print(pd.DataFrame([missing[missing>0],pd.Series(hsales.isnull().sum()[hsales.isnull().sum()>1000])], index=['percent missing','how many missing']))

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

plt.figure(figsize=(8,10))

sns.heatmap(hsales.isnull(),cmap='viridis')
# let us check for outliers first

hsales[['LAND SQUARE FEET','GROSS SQUARE FEET']].describe()
sns.jointplot(x='LAND SQUARE FEET', y='GROSS SQUARE FEET', data=hsales[(hsales['LAND SQUARE FEET']<=3500)& (hsales['GROSS SQUARE FEET']<=2560)], kind='scatter')
hsales[(hsales['LAND SQUARE FEET']<=3500)& (hsales['GROSS SQUARE FEET']<=2560)][['LAND SQUARE FEET','GROSS SQUARE FEET']].corr()
print(hsales[(hsales['LAND SQUARE FEET'].isnull()) & (hsales['GROSS SQUARE FEET'].notnull())].shape)

print(hsales[(hsales['LAND SQUARE FEET'].notnull()) & (hsales['GROSS SQUARE FEET'].isnull())].shape)
hsales['LAND SQUARE FEET'] = hsales['LAND SQUARE FEET'].mask((hsales['LAND SQUARE FEET'].isnull()) & (hsales['GROSS SQUARE FEET'].notnull()), hsales['GROSS SQUARE FEET'])

hsales['GROSS SQUARE FEET'] = hsales['GROSS SQUARE FEET'].mask((hsales['LAND SQUARE FEET'].notnull()) & (hsales['GROSS SQUARE FEET'].isnull()), hsales['LAND SQUARE FEET'])
#  Check for duplicates before

print(sum(hsales.duplicated()))

hsales[hsales.duplicated(keep=False)].sort_values(['NEIGHBORHOOD', 'ADDRESS']).head(10)

# df.duplicated() automatically excludes duplicates, to keep duplicates in df we use keep=False

# in df.duplicated(df.columns) we can specify column names to look for duplicates only in those mentioned columns.
hsales.drop_duplicates(inplace=True)

print(sum(hsales.duplicated()))
missing = hsales.isnull().sum()/len(hsales)*100

print(pd.DataFrame([missing[missing>0],pd.Series(hsales.isnull().sum()[hsales.isnull().sum()>1000])], index=['percent missing','how many missing']))
print("The number of non-null prices for missing square feet observations:\n",((hsales['LAND SQUARE FEET'].isnull()) & (hsales['SALE PRICE'].notnull())).sum())
print("non-overlapping observations that cannot be imputed:",((hsales['LAND SQUARE FEET'].isnull()) & (hsales['SALE PRICE'].isnull())).sum())
hsales[hsales['COMMERCIAL UNITS']==0].describe()
# for visualization purposes, we replace borough numbering with their string names

hsales['BOROUGH'] = hsales['BOROUGH'].astype(str)

hsales['BOROUGH'] = hsales['BOROUGH'].str.replace("1", "Manhattan")

hsales['BOROUGH'] = hsales['BOROUGH'].str.replace("2", "Bronx")

hsales['BOROUGH'] = hsales['BOROUGH'].str.replace("3", "Brooklyn")

hsales['BOROUGH'] = hsales['BOROUGH'].str.replace("4", "Queens")

hsales['BOROUGH'] = hsales['BOROUGH'].str.replace("5", "Staten Island")

hsales['BOROUGH'].value_counts()
# house prices greater than 5 mln probably represents outliers.

import matplotlib.ticker as ticker



sns.set_style("whitegrid")

plt.figure(figsize=(10,5))

plotd = sns.distplot(hsales[(hsales['SALE PRICE']>100) & (hsales['SALE PRICE'] < 5000000)]['SALE PRICE'], kde=True, bins=100)



tick_spacing=250000 # set spacing for each tick

plotd.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

plotd.set_xlim([-100000, 5000000]) # do not show negative values 

plt.xticks(rotation=30) # rotate x ticks by 30 degrees

plt.axvline(hsales[(hsales['SALE PRICE']>100) & (hsales['SALE PRICE'] < 5000000)]['SALE PRICE'].mean(), c='red')

plt.axvline(hsales[(hsales['SALE PRICE']>100) & (hsales['SALE PRICE'] < 5000000)]['SALE PRICE'].median(), c='blue')

plt.text(250000,0.0000012, "median")

plt.text(850000,0.0000010, "mean")

plt.show()
# The dataset seem to have lots of outliers, mainly due to commercial property sales

sns.boxplot(x='RESIDENTIAL UNITS',data=hsales)

plt.title('Average units per property')

plt.show()

#print('not included:', hsales[hsales['RESIDENTIAL UNITS']>10].shape[0], 'properties')
sns.boxplot(x='COMMERCIAL UNITS',data=hsales)

plt.title('Commercial units at property')

plt.show()

#print('not included:', hsales[hsales['COMMERCIAL UNITS']>20].shape[0], 'properties')
sns.boxplot(x='TOTAL UNITS',data=hsales)

plt.title('total units at property')

plt.show()

#print('not included:', hsales[hsales['TOTAL UNITS']>10].shape[0], 'properties')
sns.boxplot(x='GROSS SQUARE FEET',data=hsales)

plt.title('GROSS SQUARE FEET per property')

plt.show()

#print('not included:', hsales[hsales['GROSS SQUARE FEET']>20000].shape[0], 'properties')
print("Uneqaul values for total units:", (hsales["TOTAL UNITS"] != hsales['COMMERCIAL UNITS'] + hsales['RESIDENTIAL UNITS']).sum())
hsales[hsales["TOTAL UNITS"] != hsales['COMMERCIAL UNITS'] + hsales['RESIDENTIAL UNITS']]['TOTAL UNITS'].value_counts()
hsales[(hsales["TOTAL UNITS"] != hsales['COMMERCIAL UNITS'] + hsales['RESIDENTIAL UNITS']) & (hsales["TOTAL UNITS"]==1)]['BUILDING CLASS CATEGORY'].value_counts()[:5]
dataset = hsales[(hsales['COMMERCIAL UNITS']<20) & (hsales['TOTAL UNITS']<50) & (hsales['SALE PRICE']<5000000) & (hsales['SALE PRICE']>100000) & (hsales['GROSS SQUARE FEET']>0)]
plt.figure(figsize=(10,6))

sns.boxplot(x='COMMERCIAL UNITS', y="SALE PRICE", data=dataset)

plt.title('Commercial Units vs Sale Price')
plt.figure(figsize=(10,6))

sns.boxplot(x='RESIDENTIAL UNITS', y='SALE PRICE', data=dataset)

plt.title('Residential Units vs Sale Price')

plt.show()
dataset[dataset['YEAR BUILT']<1800]['YEAR BUILT'].value_counts()
dataset[dataset['YEAR BUILT']<1800]['BUILDING CLASS CATEGORY'].value_counts()[:15]
plt.figure(figsize=(10,6))

plotd=sns.countplot(x=dataset[dataset['YEAR BUILT']>1900]['YEAR BUILT'])

#tick_spacing=1 # set spacing for each tick

#plotd.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

#plotd.set_xlim([1900, 2020])

plt.tick_params(labelbottom=False)

plt.xticks(rotation=30) 

plt.title("Quantity of properties sold by year built")

plt.show()
sns.regplot(x='YEAR BUILT', y='SALE PRICE', data=dataset[dataset['YEAR BUILT']>1900][dataset['RESIDENTIAL UNITS']<=5], fit_reg=False, scatter_kws={'alpha':0.1})
dataset[dataset['YEAR BUILT']>1900][dataset['RESIDENTIAL UNITS']<=5].plot.scatter(x='YEAR BUILT', y='SALE PRICE', c='RESIDENTIAL UNITS', cmap='coolwarm',figsize=(12,8),s=dataset[dataset['YEAR BUILT']>1900][dataset['RESIDENTIAL UNITS']<=5]['RESIDENTIAL UNITS']*10)

plt.title('Sales Price vs year. bubble size for units')

plt.show()
dataset[dataset['YEAR BUILT']>1900][dataset['RESIDENTIAL UNITS']<=5].plot.scatter(x='YEAR BUILT', y='SALE PRICE', c='GROSS SQUARE FEET', cmap='coolwarm',figsize=(12,8),s=dataset[dataset['YEAR BUILT']>1900][dataset['RESIDENTIAL UNITS']<=5]['GROSS SQUARE FEET']*.008)

plt.title('Sales Price vs year. bubble size for gross square feet')

plt.show()
plt.figure(figsize=(10,6))

order = sorted(dataset['BUILDING CLASS CATEGORY'].unique())

sns.boxplot(x='BUILDING CLASS CATEGORY', y='SALE PRICE', data=dataset, order=order)

plt.xticks(rotation=90)

plt.title('Sale Price Distribution by Bulding Class Category')

plt.show()
# Sales prices by borough

plt.figure(figsize=(10,6))

sns.boxplot(x='BOROUGH', y='SALE PRICE', data=dataset)

plt.title('Sale Price Distribution by Borough')

plt.show()
import folium # library for interactive map drawing
# from geopy.geocoders import Nominatim # get longitude and latitude based on the address

# def get_lonlat(str_):

#     geolocator = Nominatim()

#     location = geolocator.geocode(str_, country_codes="US")

#     try:

#         return location.latitude, location.longitude

#     except:

#         return np.nan, np.nan



# import requests

# response = requests.get('https://maps.googleapis.com/maps/api/geocode/json?address=1600+Amphitheatre+Parkway,+Mountain+View,+CA')

# resp_json_payload = response.json()

# print(resp_json_payload)



# too many requests

# lonlat = []

# for val in addresses['ADDRESS']:

#     locatn = get_lonlat(val)

#     #print(val, locatn)

#     lonlat.append(locatn)

# lonlat=pd.DataFrame(lonlat, columns=["lon","lat"])

# lonlat.to_csv(path_or_buf="/kaggle/working/lonlat.csv",index=False)

# print("saved")
zipcodes = dataset[hsales["ZIP CODE"]>0]

zipcodes['ZIP']=zipcodes['ZIP CODE'].astype(str) # zipcodes should be str type because geojson file zipcodes are read as str 
boroughs = zipcodes[['ZIP','BOROUGH']]

boroughs.drop_duplicates('ZIP', inplace=True)
us_zipcodes = pd.read_csv("../input/nyc-zipcode-geodata/uszipcodes_geodata.txt", delimiter=',', dtype=str)

zipcodes_agg=pd.merge(zipcodes.groupby('ZIP').agg(np.mean), us_zipcodes, how='left', on='ZIP')

zipcodes_agg = pd.merge(zipcodes_agg, boroughs, how='left', on='ZIP')

zipcodes_agg.loc[116,'LAT']="40.6933"

zipcodes_agg.loc[116,'LNG']="-73.9925"

#zipcodes_agg
from folium.plugins import MarkerCluster # for clustering the markers

map = folium.Map(location=[40.693943, -73.985880], default_zoom_start=12)

map.choropleth(geo_data="../input/nyc-zipcode-geodata/nyc-zip-code-tabulation-areas-polygons.geojson", # I found this NYC zipcode boundaries by googling 

             data=zipcodes_agg, # my dataset

             columns=['ZIP', 'SALE PRICE'], # zip code is here for matching the geojson zipcode, sales price is the column that changes the color of zipcode areas

             key_on='feature.properties.postalCode', # this path contains zipcodes in str type, this zipcodes should match with our ZIP CODE column

             fill_color='BuPu', fill_opacity=0.7, line_opacity=0.3,

             legend_name='SALE PRICE')



# add a marker for every record in the filtered data, use a clustered view

marker_cluster = MarkerCluster().add_to(map) # create marker clusters

for i in range(zipcodes_agg.shape[0]):

    location = [zipcodes_agg['LAT'][i],zipcodes_agg['LNG'][i]]

    tooltip = "Zipcode:{}<br> Borough: {}<br> Click for more".format(zipcodes_agg["ZIP"][i], zipcodes_agg['BOROUGH'][i])

    folium.Marker(location, 

                  popup="""<i>Mean sales price: </i> <br> <b>${}</b> <br>

                  <i>mean total units: </i><b><br>{}</b><br>

                  <i>mean square feet: </i><b><br>{}</b><br>""".format(round(zipcodes_agg['SALE PRICE'][i],2), round(zipcodes_agg['TOTAL UNITS'][i],2), round(zipcodes_agg['GROSS SQUARE FEET'][i],2)), 

                  tooltip=tooltip).add_to(marker_cluster)

map
map = folium.Map(location=[40.693943, -73.985880], default_zoom_start=12)

map.choropleth(geo_data="../input/nyc-zipcode-geodata/nyc-zip-code-tabulation-areas-polygons.geojson", # I found this NYC zipcode boundaries by googling 

             data=zipcodes, # my dataset

             columns=['ZIP', 'SALE PRICE'], # zip code is here for matching the geojson zipcode, sales price is the column that changes the color of zipcode areas

             key_on='feature.properties.postalCode', # this path contains zipcodes in str type, this zipcodes should match with our ZIP CODE column

             fill_color='BuPu', fill_opacity=0.7, line_opacity=0.2,

             legend_name='SALE PRICE')



# add a marker for every record in the filtered data, use a clustered view

# marker_cluster = MarkerCluster().add_to(map) # create marker clusters

# for i in range(zipcodes_agg.shape[0]):

#     location = [zipcodes_agg['LAT'][i],zipcodes_agg['LNG'][i]]

#     tooltip = "Zipcode:{}<br> Borough: {}<br> Click for more".format(zipcodes_agg["ZIP"][i], zipcodes_agg['BOROUGH'][i])

#     folium.Marker(location, 

#                   popup="""<i>Mean sales price: </i> <br> <b>${}</b> <br>

#                   <i>mean total units: </i><b><br>{}</b><br>

#                   <i>mean square feet: </i><b><br>{}</b><br>""".format(round(zipcodes_agg['SALE PRICE'][i],2), round(zipcodes_agg['TOTAL UNITS'][i],2), round(zipcodes_agg['GROSS SQUARE FEET'][i],2)), 

#                   tooltip=tooltip).add_to(marker_cluster)

map
map.save('mymap.html')