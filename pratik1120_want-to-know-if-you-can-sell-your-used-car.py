import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import plotly.express as px
data = pd.read_csv('../input/craigslist-carstrucks-data/vehicles.csv')

data['year'] = data['year'].astype(float)
#dropping unnecessary columns

data.drop(['url','region_url','image_url','description','county'], axis=1, inplace=True)
sns.kdeplot(data=data['price'],label="Price" ,shade=True)
price = data.groupby('manufacturer')['price'].max().reset_index()

price  = price.sort_values("price")

price = price.tail(10)

fig = px.pie(price,

             values="price",

             names="manufacturer",

             template="seaborn")

fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")

fig.show()
price = data.groupby('manufacturer')['price'].min().reset_index()

price  = price.sort_values("price")

price = price[price['price']!=0] #this is necessary so that zero price values are not considered

price = price.head(10)

fig = px.pie(price,

             values="price",

             names="manufacturer",

             template="seaborn")

fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")

fig.show()
price = data.groupby('condition')['price'].max().reset_index()

price  = price.sort_values("price")

price = price.tail(10)

fig = px.pie(price,

             values="price",

             names="condition",

             template="seaborn")

fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")

fig.show()
sns.kdeplot(data=data['odometer'], label="Distance", shade=True)
price = data.groupby('fuel')['price'].max().reset_index()

price  = price.sort_values("price")

price = price.tail()

fig = px.pie(price,

             values="price",

             names="fuel",

             template="seaborn")

fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")

fig.show()
color = data.loc[:,['paint_color']]

color['count'] = color.groupby(color['paint_color'])['paint_color'].transform('count')

color = color.drop_duplicates()

color = color.sort_values("count",ascending = False)

color = color.head(10)

fig = px.pie(color,

             values="count",

             names="paint_color",

             template="seaborn")

fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")

fig.show()
data.drop(['id','vin'], axis=1, inplace=True)
cyl = data.loc[:,['cylinders']]

cyl['count'] = cyl.groupby(cyl['cylinders'])['cylinders'].transform('count')

cyl = cyl.drop_duplicates()

cyl = cyl.sort_values("count",ascending = False)

cyl = cyl.head(10)

fig = px.pie(cyl,

             values="count",

             names="cylinders",

             template="seaborn")

fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")

fig.show()
title = data.loc[:,['title_status']]

title['count'] = title.groupby(title['title_status'])['title_status'].transform('count')

title = title.drop_duplicates()

title = title.sort_values("count",ascending = False)

title = title.head(10)

fig = px.pie(title,

             values="count",

             names="title_status",

             template="seaborn")

fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")

fig.show()
sns.scatterplot(data['price'], data['odometer'])
perc = data.loc[:,["year","manufacturer",'price']]

perc['mean_price'] = perc.groupby([perc.manufacturer,perc.year])['price'].transform('mean')

perc.drop('price', axis=1, inplace=True)

perc = perc.drop_duplicates()

perc = perc[perc['year'].astype('float')>=2000.0]

perc = perc.sort_values("year",ascending = False)

top_brand = ['toyota','ford','lexus','hyundai',"honda"] 

perc = perc.loc[perc['manufacturer'].isin(top_brand)]

perc = perc.sort_values("year")

perc = perc.fillna(100)

fig=px.bar(perc,x='manufacturer', y="mean_price", animation_frame="year", 

           animation_group="manufacturer", color="manufacturer", hover_name="manufacturer")

fig.show()
trans = data.loc[:,['price','transmission']]

trans = trans.groupby('transmission')['price'].max().reset_index()

trans  = trans.sort_values("price")

fig = px.pie(trans,

             values="price",

             names="transmission",

             template="seaborn")

fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")

fig.show()
top_regions = data['region'].value_counts().head().reset_index()

sns.barplot(top_regions['index'], top_regions['region'])
model = data.loc[:,['price','manufacturer','model']]

model = model.groupby(['manufacturer','model'])['price'].max().reset_index()

model = model.sort_values('price')

model = model.tail(10)

fig = px.pie(model, values='price', names='model', template='seaborn')

fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")

fig.show()
model = data.loc[:,['price','manufacturer','model']]

model = model.groupby(['manufacturer','model'])['price'].min().reset_index()

model = model.sort_values('price')

model = model[model['price']!=0]

model = model.head(10)

fig = px.pie(model, values='price', names='model', template='seaborn')

fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")

fig.show()
drives = ['4wd','fwd','rwd']



for drive in drives:

    drive_df = data.loc[data['drive']==drive, ['year','drive']]

    drive_df = drive_df[drive_df['year']>=2010]

    drive_df['counts'] = drive_df.groupby('year')['drive'].transform('count')

    drive_df = drive_df.sort_values('year')

    sns.lineplot(x='year', y='counts', data=drive_df, label=drive)
types = data.loc[:,['price','type']]

types = types.groupby('type')['price'].max().reset_index()

types = types.sort_values('price')

types = types.tail(10)

fig = px.pie(types, values='price', names='type', template='seaborn')

fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")

fig.show()
top_states = data['state'].value_counts().head().reset_index()

sns.barplot(top_states['index'], top_states['state'])
import warnings

warnings.filterwarnings('ignore')

import folium



m = folium.Map(location=[35.7636, -78.7443])



data_map = data.loc[:,['region','lat','long']]

data_map = data_map.dropna().reset_index(drop=True)

new_map = data_map.groupby('region')['lat','long'].mean().reset_index()

for i in range(new_map.shape[0]):

    folium.Marker([new_map.loc[i]['lat'], new_map.loc[i]['long']], tooltip=new_map.loc[i]['region']).add_to(m)

m