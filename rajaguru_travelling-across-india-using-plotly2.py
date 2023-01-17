import pandas as pd

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.figure_factory as ff

from IPython.display import HTML, Image

import matplotlib.pyplot as plt

import numpy as np

import os

import sys

plt.style.use("seaborn-white")
DIR="../input/"

indiancities=os.path.join(DIR,'cities_r2.csv')
indianCities=pd.read_csv(indiancities)

indianCities.head()
trace1 = go.Scatter(

                    x = indianCities.name_of_city,

                    y = indianCities.population_male,

                    mode = "lines",

                    name = "citations",

                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),

                    text= indianCities.name_of_city) 

# Creating trace2

trace2 = go.Scatter(

                    x = indianCities.name_of_city,

                    y = indianCities.population_female,

                    mode = "lines+markers",

                    name = "teaching",

                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),

                    )

data = [trace1, trace2]

layout = dict(title = 'Citation and Teaching vs World Rank of Top 100 Universities',

              xaxis= dict(title= 'World Rank',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)

indianCities.columns.tolist() 
lat=[]

lon=[]

for x in indianCities['location']:

  a=x.split(',')

  lat.append(float(a[0]))

  lon.append(float(a[1]))

  
mapbox_access_token="pk.eyJ1IjoiZ3VydW5hdGgwNSIsImEiOiJjanNuYmpyeDIwYWx2NGFsanZoejJsMzRwIn0.LhyRdspSIhxn-1s5R_9KLQ"
data = [

    go.Scattermapbox(

        lat=lat,

        lon=lon,

        mode='markers',

        marker=go.scattermapbox.Marker(

            size=indianCities['population_total']//170000,

            color=indianCities['state_code'],

        ),

        

        text=indianCities['state_name']+" :- "+indianCities['name_of_city'] +" Total population "+indianCities['population_total'].astype(str)

                                    +" Male population "+indianCities['population_male'].astype(str)+" Female population "+

                                    indianCities['population_female'].astype(str),

    )

]



layout = go.Layout(

    autosize=True,

    width=1000,

    height=1000,

    hovermode='closest',

    mapbox=go.layout.Mapbox(

        accesstoken=mapbox_access_token,

        bearing=0,

        center=go.layout.mapbox.Center(

            lat=min(lat),

            lon=min(lon),

        ),

        pitch=0,

        zoom=3,

    ),

)



fig = go.Figure(data=data, layout=layout)

iplot(fig)

indianCities['state_name'].value_counts()
indianCities.columns[4:].tolist()

indianCities[indianCities['name_of_city'].str.contains('Chennai')].values[:,4:].ravel().tolist()
data=[go.Scatterpolar(

      name = "Tambaram",

      r = indianCities[indianCities['name_of_city'].str.contains('Tambaram')].values[:,4:].ravel().tolist(),

      theta =indianCities.columns[4:].tolist(),

      fill = "toself",

    ),

      

    go.Scatterpolar(

      name = "Pallavaram",

      r = indianCities[indianCities['name_of_city'].str.contains('Pallavaram')].values[:,4:].ravel().tolist(),

      theta =indianCities.columns[4:].tolist(),

      fill = "toself",

    ),

     ]

layout = go.Layout(

    polar = dict(

        radialaxis = dict(

            visible = True,

        )

    ),

    showlegend = True

)



fig = go.Figure(data=data, layout=layout)

iplot(fig)
numerical_columns=indianCities.select_dtypes('int64').columns.tolist()+indianCities.select_dtypes('float64').columns.tolist()
indianStates=indianCities.groupby('state_name').sum()[numerical_columns].reset_index()

indianStates.head()
indianStates.columns[3:].tolist()
data=[go.Scatterpolar(

      name = state,

      r = indianStates[indianStates['state_name'].str.contains(state)].values[:,3:].ravel().tolist(),

      theta =indianStates.columns[3:].tolist(),

      fill = "toself",

    ) for state in indianStates['state_name']]

layout = go.Layout(

    title="All states comparison",

    polar = dict(

        radialaxis = dict(

            visible = False,

        )

    ),

    showlegend = True

)



fig = go.Figure(data=data, layout=layout)

iplot(fig)
plt.style.available
plt.style.use('seaborn-talk')
plt.bar(indianStates['state_name'],indianStates['total_graduates'])

plt.xticks(rotation='vertical')

plt.show()
mapper_list=['total_graduates','male_graduates','female_graduates']



fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(8,20))

plt.xticks(rotation='vertical')

ax[0].bar(indianStates['state_name'],indianStates[mapper_list[0]])

plt.setp( ax[0].xaxis.get_majorticklabels(), rotation=90 )

ax[1].bar(indianStates['state_name'],indianStates[mapper_list[1]])

ax[2].bar(indianStates['state_name'],indianStates[mapper_list[2]])

ax[0].set_title(mapper_list[0])

ax[1].set_title(mapper_list[1])

ax[2].set_title(mapper_list[2])

plt.setp( ax[1].xaxis.get_majorticklabels(), rotation=90 )

plt.setp( ax[2].xaxis.get_majorticklabels(), rotation=90 )



plt.tight_layout()
plt.style.use("seaborn-bright")
mapper_list=['total_graduates','male_graduates','female_graduates']



fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15,10))

plt.xticks(rotation='vertical')

ax[0][0].bar(indianStates['state_name'],indianStates[mapper_list[0]])

plt.setp( ax[0][0].xaxis.get_majorticklabels(), rotation=90 )

ax[0][1].bar(indianStates['state_name'],indianStates[mapper_list[1]],color='green')

ax[1][0].bar(indianStates['state_name'],indianStates[mapper_list[2]],color='red')

ax[1][1].plot(indianStates['state_name'],indianStates[mapper_list[0]],ls='--')

ax[1][1].plot(indianStates['state_name'],indianStates[mapper_list[1]],ls='--',color='green')

ax[1][1].plot(indianStates['state_name'],indianStates[mapper_list[2]],ls='--',color='red')



ax[0][0].set_title(mapper_list[0])

ax[0][1].set_title(mapper_list[1])

ax[1][0].set_title(mapper_list[2])

plt.setp( ax[0][0].xaxis.get_majorticklabels(), rotation=90 )

plt.setp( ax[0][1].xaxis.get_majorticklabels(), rotation=90 )

plt.setp( ax[1][0].xaxis.get_majorticklabels(), rotation=90 )



plt.tight_layout()
mapper_list=['population_total','population_male','population_female']



fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15,10))

plt.xticks(rotation='vertical')

ax[0][0].bar(indianStates['state_name'],indianStates[mapper_list[0]])

plt.setp( ax[0][0].xaxis.get_majorticklabels(), rotation=90 )

ax[0][1].bar(indianStates['state_name'],indianStates[mapper_list[1]],color='green')

ax[1][0].bar(indianStates['state_name'],indianStates[mapper_list[2]],color='red')

ax[1][1].plot(indianStates['state_name'],indianStates[mapper_list[0]],ls='--')

ax[1][1].plot(indianStates['state_name'],indianStates[mapper_list[1]],ls='--',color='green')

ax[1][1].plot(indianStates['state_name'],indianStates[mapper_list[2]],ls='--',color='red')



ax[0][0].set_title(mapper_list[0])

ax[0][1].set_title(mapper_list[1])

ax[1][0].set_title(mapper_list[2])

plt.setp( ax[0][0].xaxis.get_majorticklabels(), rotation=90 )

plt.setp( ax[0][1].xaxis.get_majorticklabels(), rotation=90 )

plt.setp( ax[1][0].xaxis.get_majorticklabels(), rotation=90 )

plt.title("population overview")

plt.tight_layout()


mapper_list=['literates_total','literates_male','literates_female']

mapper_list=['0-6_population_total','0-6_population_male','0-6_population_female']

mapper_list=['effective_literacy_rate_total','effective_literacy_rate_male','effective_literacy_rate_female']

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15,10))

plt.xticks(rotation='vertical')

ax[0][0].bar(indianStates['state_name'],indianStates[mapper_list[0]])

plt.setp( ax[0][0].xaxis.get_majorticklabels(), rotation=90 )

ax[0][1].bar(indianStates['state_name'],indianStates[mapper_list[1]],color='green')

ax[1][0].bar(indianStates['state_name'],indianStates[mapper_list[2]],color='red')

ax[1][1].plot(indianStates['state_name'],indianStates[mapper_list[0]],ls='--')

ax[1][1].plot(indianStates['state_name'],indianStates[mapper_list[1]],ls='--',color='green')

ax[1][1].plot(indianStates['state_name'],indianStates[mapper_list[2]],ls='--',color='red')



ax[0][0].set_title(mapper_list[0])

ax[0][1].set_title(mapper_list[1])

ax[1][0].set_title(mapper_list[2])

plt.setp( ax[0][0].xaxis.get_majorticklabels(), rotation=90 )

plt.setp( ax[0][1].xaxis.get_majorticklabels(), rotation=90 )

plt.setp( ax[1][0].xaxis.get_majorticklabels(), rotation=90 )

plt.title("population overview")

plt.tight_layout()
data = [

    go.Scattermapbox(

        lat=lat,

        lon=lon,

        mode='markers',

        marker=go.scattermapbox.Marker(

            size=indianCities['population_total']//170000,

            color=indianCities['state_code'],

        ),

        

        text=indianCities['state_name']+" :- "+indianCities['name_of_city'] +" Total population "+indianCities['population_total'].astype(str)

                                    +" Male population "+indianCities['population_male'].astype(str)+" Female population "+

                                    indianCities['population_female'].astype(str),

    )

]



layout = go.Layout(

    autosize=True,

    width=1000,

    height=1000,

    hovermode='closest',

    mapbox=go.layout.Mapbox(

        accesstoken=mapbox_access_token,

        bearing=0,

        center=go.layout.mapbox.Center(

            lat=min(lat),

            lon=min(lon),

        ),

        pitch=0,

        zoom=3,

    ),

)



fig = go.Figure(data=data, layout=layout)

iplot(fig)

tamilnaduCities=indianCities[indianCities['state_name']=='TAMIL NADU']

tamilnaduCities.head()
data=[go.Scatterpolargl(

      name = city,

      r = tamilnaduCities[tamilnaduCities['name_of_city'].str.contains(city)].values[:,3:].ravel().tolist(),

      theta =tamilnaduCities.columns[3:].tolist(),

      mode='markers',

      fill = "toself",

    ) for city in tamilnaduCities['name_of_city']]

layout = go.Layout(

    title="Tamil Nadu cities comparison",

    polar = dict(

        radialaxis = dict(

            visible = False,

        )

    ),

    showlegend = True

)





fig = go.Figure(data=data, layout=layout)

iplot(fig)
reduced_columns=['population_total','literates_total','sex_ratio','effective_literacy_rate_total','total_graduates']

data=[go.Scatterpolar(

      name = city,

      r = tamilnaduCities[tamilnaduCities['name_of_city'].str.contains(city)][reduced_columns].values.ravel().tolist(),

      theta =reduced_columns,

      fill = "toself",

    ) for city in tamilnaduCities['name_of_city']]

layout = go.Layout(

    title="Tamil Nadu cities comparison",

    polar = dict(

        radialaxis = dict(

            visible = False,

        )

    ),

    showlegend = True

)



fig = go.Figure(data=data, layout=layout)

iplot(fig)
tamilnaduCities['location']
tamilnaduCities['name_of_city'].astype('category').cat.codes.tolist()
import matplotlib.pyplot as plt

from matplotlib import cm

from matplotlib.dates import date2num

from mpl_toolkits.basemap import Basemap



indianCities['longitude']=lon

indianCities['latitude']=lat
print("The Top 10 Cities sorted according to the Total Population (Descending Order)")

top_pop_cities = indianCities.sort_values(by='population_total',ascending=False)

top10_pop_cities=top_pop_cities.head(10)

top10_pop_cities

from numpy import array

plt.subplots(figsize=(20, 15))

map = Basemap(width=1200000,height=900000,projection='lcc',resolution='l',

                    llcrnrlon=67,llcrnrlat=5,urcrnrlon=99,urcrnrlat=37,lat_0=28,lon_0=77)



map.drawmapboundary ()

map.drawcountries ()

map.drawcoastlines ()



lg=array(top10_pop_cities['longitude'])

lt=array(top10_pop_cities['latitude'])

pt=array(top10_pop_cities['population_total'])

nc=array(top10_pop_cities['name_of_city'])



x, y = map(lg, lt)

population_sizes = top10_pop_cities["population_total"].apply(lambda x: int(x / 5000))

plt.scatter(x, y, s=population_sizes, marker="o", c=population_sizes, cmap=cm.Dark2, alpha=0.7)





for ncs, xpt, ypt in zip(nc, x, y):

    plt.text(xpt+60000, ypt+30000, ncs, fontsize=10, fontweight='bold')



plt.title('Top 10 Populated Cities in India',fontsize=20)

import plotly_express as px

px.scatter(tamilnaduCities, x="population_male", y="population_female", color="name_of_city",size='population_total')

indianCities.columns.tolist()
indianStates=indianCities.groupby('state_name').sum()[numerical_columns].reset_index()

indianStates.head()
indianCities['total_graduates']==indianCities['male_graduates']+indianCities['female_graduates']
px.scatter(indianStates, x="male_graduates", y="female_graduates", animation_frame="state_name",

           color="state_name", hover_name="state_name", log_x=True)

state=indianStates[indianStates['state_name']=='TAMIL NADU']
trace = go.Pie(labels=['m','f'], values=[state['population_male'].values[0],state['population_female'].values[0]],hole=.4)



iplot([trace])

px.scatter_matrix(indianStates, dimensions=['population_male','literates_male','0-6_population_male','effective_literacy_rate_male','male_graduates']

)

px.scatter_matrix(indianStates, dimensions=['population_male','literates_male','0-6_population_male','effective_literacy_rate_male','total_graduates']

)

px.scatter_matrix(indianStates, dimensions=['population_female','literates_female','0-6_population_female','female_graduates']

)
