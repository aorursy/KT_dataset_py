import numpy as np

import pandas as pd

import os

import seaborn as sns

sns.set(style="ticks")



import datetime

import matplotlib as mpl

import matplotlib.pyplot as plt
avo=pd.read_csv("../input/notebook-datasets/avocado.csv",encoding="utf-8")
#get general info about the dataset

avo.info()
#check for missing values

avo.isnull().sum()
for col in avo.columns:

    if type(avo[col][1])!=str:

        print(col, ' min: ', avo[col].min(), ' max: ', avo[col].max())
#all columns with numbers are floats and all columns with words are strings

#transform data column to datetime format

avo["Date"]=pd.to_datetime(pd.Series(avo["Date"]), format="%m/%d/%Y")

len(avo.Date.unique()) # it includes 169 dates
#let's see the string columns to make sure that values there are in line

len(avo['region'].unique())
for i in avo['region'].unique():

    print(i)
not_cities=['TotalUS','West','California','Midsouth','Northeast','SouthCarolina','SouthCentral','Southeast','GreatLakes','NothernNewEngland']

cities=avo[avo['region'].isin(not_cities)==False]
cities['month']=[x.month for x in cities.Date]

cities['year']=[x.year for x in cities.Date]

pivot_cities=pd.pivot_table(cities,index=['Date','region'],values=['AveragePrice','Total Volume'])

pivot_cities=pivot_cities.reset_index()

pivot_cities=pivot_cities.rename(columns={'region':'city'})
plt.figure(figsize=(20,15))  

g=sns.lineplot(x='Date', y='AveragePrice', hue='city',data=pivot_cities)

g=g.set_xlim(pivot_cities['Date'].min(), pivot_cities['Date'].max())

plt.legend(bbox_to_anchor=(1, 1), loc=2)

plt.title("Avocado Average Price 2015 - 2018",pad=20, fontsize=30)

plt.ylabel('Average Price', fontsize=20)

plt.xlabel('Date', fontsize=20)

plt.show(g)
plt.figure(figsize=(20,15))  

g=sns.lineplot(x='Date', y='Total Volume', hue='city',data=pivot_cities)

g=g.set_xlim(pivot_cities['Date'].min(), pivot_cities['Date'].max())

plt.legend(bbox_to_anchor=(1, 1), loc=2)

plt.title("Avocade Sales Total Volume 2015 - 2018",pad=20, fontsize=30)

plt.ylabel('Total Volume', fontsize=20)

plt.xlabel('Date', fontsize=20)

plt.show(g)
# or another approach add column that will scale Total Volume to be able to show in one chart

pivot_cities['TV']=pivot_cities['Total Volume']/1000000

g=sns.FacetGrid(pivot_cities, col='city', col_wrap=5,height=1.5, aspect=1.5)

g=g.map(plt.plot, 'Date','AveragePrice').set_xticklabels([str(x)[:10] for x in pivot_cities['Date']],rotation=90)

g.map(plt.plot, 'Date', 'TV',color='r').set_xticklabels([str(x)[:10] for x in pivot_cities['Date']],rotation=90).set_ylabels('Volume & Price')

g.add_legend()

g.set_titles('{col_name}')
#create new pivot table to be able to see the avocado type difference

pivot_cities1=pd.pivot_table(cities,index=['Date','region','type'],values=['AveragePrice','Total Volume'])

pivot_cities1=pivot_cities1.reset_index()



g = sns.FacetGrid(pivot_cities1, col="type",height=4, aspect=2)

g.map(plt.hist, "AveragePrice")



g = sns.FacetGrid(pivot_cities1, col="type",height=4, aspect=2)

g.map(plt.hist, "Total Volume")
#calculate total volume per city for the whole period



sum_tot={}

for c in pivot_cities.city:

    a=pivot_cities[pivot_cities['city']==c]

    tot=a['Total Volume'].sum()

    sum_tot[c]=round(tot,2)



cities_list=[i for i in sum_tot.keys()]

val=[i for i in sum_tot.values()]



plt.figure(figsize=(3,10))

sns.barplot(val,cities_list, palette="ch:.25")

plt.title("Total Avocado Consumption per City", pad=20, fontsize=20)


coord = pd.read_csv("../input//notebook-datasets/coordinates.csv")
coord.info()
coord=coord[(coord['name']!='GreatLakes') & (coord['name']!='NorthernNewEngland')].reset_index()

c_dic=dict(zip(cities_list, val))

coord['total']=coord['name'].map(c_dic)

coord['total']=[round(i,2) for i in coord['total']]

#plot map



plt.figure(figsize=(50,50))

#style = dict(size=10, color='gray')

#plt.imshow(usa_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5)

ax=coord.plot(kind="scatter", x="longitude", y="latitude",

     c=coord['total'],s=coord['total']/100000, cmap=plt.get_cmap("jet"),

    colorbar=True, alpha=0.2, figsize=(15,10),

)

for i, txt in enumerate(coord['name']):

    ax.annotate(txt, (coord['longitude'][i]+0.3, coord['latitude'][i]+0.3))

    #plt.text(x+.03, y+.03, txt, fontsize=9)



plt.title("Total Avocado Consumption Bubble Map", pad=20, fontsize=20)

plt.ylabel("Latitude", fontsize=14)

plt.legend()

plt.show(ax)
import plotly.express as px
coord_lat_dic=dict(zip(coord['name'],coord['latitude']))

coord_lon_dic=dict(zip(coord['name'],coord['longitude']))

pivot_cities['lat']=pivot_cities['city'].map(coord_lat_dic)

pivot_cities['lon']=pivot_cities['city'].map(coord_lon_dic)
pivot_cities['date']=[str(x)[:10] for x in pivot_cities['Date']]



fig = px.scatter_geo(pivot_cities, color="Total Volume",

                     hover_name="city", size="Total Volume",

                     lon='lon',

                     lat='lat',

                     animation_frame="date",

                     scope='usa',

                     title ={'text':"Avocado Consumption Growth 2015-2018",

                            'xanchor': 'center',

                            'y':0.9,

                            'x':0.5}

                    )

fig.show()