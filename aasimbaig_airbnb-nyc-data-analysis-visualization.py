import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import folium

from folium.plugins import HeatMap
## loading the data

data = pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")
## lets examine at first 3 rows

data.head(3)
## how many rows are present or what size of data we are dealing with

len(data)
## how many columns in our data and their types 

data.dtypes
##

#data.isnull().sum()

## can be use in different datasets too !!

## we will use 'total' to save the sorted value in descending order

total = data.isnull().sum().sort_values(ascending=False)

## now we need to calculate percentage ie. (null_value of column * 100) / (total_null values)

percent = ((data.isnull().sum())*100)/data.isnull().count().sort_values(ascending=False)

## now we add these two variables using concat function

missing_data = pd.concat([total, percent], axis=1, keys=['Total','Percent'], sort=False).sort_values('Total', ascending=False)

## print the missing values

missing_data.head()
## lets look at last_review column

data.last_review.head()
data.drop(['name','host_name','last_review'],axis=1,inplace=True)

### axis=1 -> remove column

### inplace=True -> immediate update to our data frame
# Rechecking the data

data.head()
## lets add 0 to our reviews_per_month instead of NAN values

data.reviews_per_month.fillna(0,inplace=True)

data.reviews_per_month.isnull().sum()
# Thankyou to Michal Bogacz for commenting : 

# There is no sense to count correlation with ID. 

# This is only ordinary number, not variable.

columns = ['id','host_id']

new_data = data.drop(axis=1,labels=columns)

new_data.corr().style.background_gradient(cmap="coolwarm")
data.describe().T
## lets look at host listings on airbnb



## we will take the top 10 host and look at their listings on airbnb



## lets take value_counts() function and add head(10) so we get the top 10 host in data

hosts = data.host_id.value_counts().head(10)

hosts
plt.figure(figsize=(10,8))

h = hosts.plot(kind = 'bar')

h.set_title("Top hosts on AIRBNB in NYC")

h.set_xlabel("Host ID")

h.set_ylabel("Listings")
## we will plot 2 different subplots

## 1. Pie plot visualization 2. bar graph visualization



f,ax = plt.subplots(1,2,figsize=(15,5))

## explode -> adds space between each pie wedge

## autopct -> adds percentage value of each pie wedge

data.neighbourhood_group.value_counts().plot.pie(explode=[0.1,0.1,0.1,0.1,0.1],autopct='%1.1f%%',ax=ax[0])

## easy countplot or bar graph of neighbourhood_group

sns.countplot(data['neighbourhood_group'])



## plots the graph

plt.show()
## creates the figure of size 10 widht and 10 height

plt.figure(figsize=(10,10))

## simple scatterplot of different neighbourhood_groups in nyc

## later we will plot these stuff on actual nyc map

sns.scatterplot(x='longitude', y='latitude', hue='neighbourhood_group',s=20, data=data, palette="muted")



## we can see different borough in nyc
## setting style for our plots

sns.set(style="white", palette="muted", color_codes=True)

## ignore -- f, axes = plt.subplots(3, 2, figsize=(10, 10), sharex=True) -- ignore ##



## figure size with 10 width and 5 height

plt.figure(figsize=(10, 5))

## create dataframe "df1" with all the neighbourhood of Brooklyn and their price

df1 = data[data.neighbourhood_group == "Brooklyn"][["neighbourhood","price"]]

## lets take mean of all the prices of neighbouhood

d = df1.groupby("neighbourhood").mean()

## distplot -> distribution plot

## axlabel == xlabel

## kde_kws -> kernel density estimate keyword arguments -> color="black"

## hist_kws -> histogram keyword arguments -> histogram type = step

sns.distplot(d,color='r',axlabel ="Price Distribution in Brooklyn",kde_kws={"color": "k"},

             hist_kws={"histtype":"step","linewidth": 3})

plt.plot()
plt.figure(figsize=(10, 5))

df1 = data[data.neighbourhood_group == "Manhattan"][["neighbourhood","price"]]

d = df1.groupby("neighbourhood").mean()

sns.distplot(d,color='r',axlabel ="Price Distribution in Manhattan",kde_kws={"color": "k"},

             hist_kws={"histtype":"step","linewidth": 3})

plt.plot()
plt.figure(figsize=(10, 5))

df1 = data[data.neighbourhood_group == "Queens"][["neighbourhood","price"]]

d = df1.groupby("neighbourhood").mean()

sns.distplot(d,color='r',axlabel ="Price Distribution in Queens",kde_kws={"color": "k"},

             hist_kws={"histtype":"step","linewidth": 3})

plt.plot()
plt.figure(figsize=(10, 5))

df1 = data[data.neighbourhood_group == "Staten Island"][["neighbourhood","price"]]

d = df1.groupby("neighbourhood").mean()

sns.distplot(d,color='r',axlabel ="Price Distribution in Staten Island",kde_kws={"color": "k"},

             hist_kws={"histtype":"step","linewidth": 3})

plt.plot()
plt.figure(figsize=(10, 5))

df1 = data[data.neighbourhood_group == "Bronx"][["neighbourhood","price"]]

d = df1.groupby("neighbourhood").mean()

sns.distplot(d,color='r',axlabel ="Price Distribution in Bronx",kde_kws={"color": "k"},

             hist_kws={"histtype":"step","linewidth": 3})

plt.plot()
plt.figure(figsize=(10,5))

df=data[data.price<500]

tt=sns.violinplot(data=df, x='neighbourhood_group', y='price')

tt.set_title('Distribution of price in neighbourhood')
data['price'].groupby(data["neighbourhood_group"]).describe()
f,ax = plt.subplots(1,2,figsize=(15,5))

data.room_type.value_counts().plot.pie(explode=[0.1,0.1,0.1],autopct='%1.1f%%',ax=ax[0],colors = ['#66b3ff','#ff9999','#99ff99'])

ax = sns.countplot(data.room_type,palette="Pastel1")

plt.show()
plt.figure(figsize=(10,5))

df1 = data[data.room_type == "Private room"][["neighbourhood_group","price"]]

d2 = df1.groupby("neighbourhood_group").mean()

sns.distplot(d2,color='b',axlabel ="Price Distribution for Private room",kde_kws={"color": "k"},

             hist_kws={"histtype":"step","linewidth": 3})

plt.show()
plt.figure(figsize=(10,5))

df1 = data[data.room_type == "Shared room"][["neighbourhood_group","price"]]

d2 = df1.groupby("neighbourhood_group").mean()

sns.distplot(d2,color='b',axlabel ="Price Distribution for Shared room",kde_kws={"color": "k"},

             hist_kws={"histtype":"step","linewidth": 3})

plt.show()
plt.figure(figsize=(10,5))

df1 = data[data.room_type == "Entire home/apt"][["neighbourhood_group","price"]]

d2 = df1.groupby("neighbourhood_group").mean()

sns.distplot(d2,color='b',axlabel ="Price Distribution for Entire home/apt",kde_kws={"color": "k"},

             hist_kws={"histtype":"step","linewidth": 3})

plt.show()
plt.figure(figsize=(10,5))

df=data[data.price<500]

tt=sns.violinplot(data=df, x='room_type', y='price')

tt.set_title('Distribution of price for different rooms type')
plt.figure(figsize=(10,10))

ax = sns.countplot(data['room_type'],hue=data['neighbourhood_group'], palette='muted')
data['price'].groupby(data["room_type"]).describe()
## create a plot of 10x10

plt.figure(figsize=(10,10))



## reads the nyc image in nyc_img 

nyc_img = plt.imread("../input/new-york-city-airbnb-open-data/New_York_City_.png",0)



## plot the nyc_image using "imshow()"

## zorder -> how close our objects to foreground

## eg - zorder = 0 -> closer to background ,

##      zorder = 5 -> closer to front

## we use "extent" to aling our latitude and longitude to our image so

## extent = [latitude_start, latitude_end, longitude_start, longitude_end]



## plt.gca() -> gives current axes of the plot



plt.imshow(nyc_img,zorder=0,extent=[-74.258, -73.7, 40.49,40.92])

ax=plt.gca()



## we take price between 0$ and 300$ so we can remove the outliers

st = data[data.price<300]



## plotting scatter plot of price on the nyc image

vt = st.plot(kind="scatter",

        x="longitude", # data positions x

        y="latitude",  # data positions y

        c = "price", # sequence

        cmap = plt.get_cmap('jet'),

        colorbar=True, # shows the colorbar at right size of image

        alpha = 0.4, # opacity of the scatter points

        zorder=5, # zorder = 5 in above the image or closer to front

        label="availability_365", 

        ax = ax) # axis that we got from plt.gca()

## shows the legend

vt.legend()

# plot the scatter plot

plt.show()
## plotting avaliability_365

f,ax = plt.subplots(figsize=(10,10))

plt.imshow(nyc_img,zorder=0,extent=[-74.258, -73.7, 40.49,40.92])

ax=plt.gca()



ax = sns.scatterplot(

        x ="longitude",

        y ="latitude",

        hue="availability_365",

        palette = 'jet',

        data = data,

    alpha = 0.4

)

plt.show()
m=folium.Map([40.7128,-74.0060],zoom_start=10)

location = ['latitude','longitude']

df = data[location]

HeatMap(df.dropna(),radius=8,gradient={.4: 'blue', .65: 'lime', 1: 'red'}).add_to(m)

display(m)