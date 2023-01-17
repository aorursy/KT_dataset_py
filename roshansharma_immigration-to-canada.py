#installing plotly express for advanced visualizations



!pip install plotly_express
# for some basic operatios

import numpy as np 

import pandas as pd 



# for visualizations

import matplotlib.pyplot as plt

import seaborn as sns



import folium

import plotly_express as px



# for networks

import networkx as nx



# for providing path

import os

print(os.listdir("../input"))

# there are three sheets in this dataset

# the first sheet is Regions by Citizenship, 



# the second sheet is Canada by Citizenship, and 

# the third sheet is Canada by Citizenship2



# In this Study we will focus on Canada by Citizenship, 

# we have skip first 20 rows as it contains some poster of an organization 



# and also the last row as it contains total.



data = pd.read_excel('../input/Canada.xlsx',

                     sheet_name='Canada by Citizenship',

                     skiprows = range(20),

                     skipfooter = 2)



# getting the shape of the data

data.shape



# checking the head of the data



data.sample(15)
# let's check the columns in the data set



data.columns
plt.rcParams['figure.figsize'] = (15, 15)

plt.style.use('fivethirtyeight')



sns.heatmap(data.corr(), cmap = 'viridis')

plt.title('Correlation Map', fontsize = 20)

plt.show()


fig = px.bar_polar(data, r="REG", theta="AreaName",color="RegName", template="plotly_dark",

            color_discrete_sequence= px.colors.sequential.Plasma[-2::-1])

fig.show()



import plotly.io as pio



pio.templates.default = "ggplot2"



d = data[['Coverage','AreaName','DevName',1980]]



fig = px.parallel_categories(d, color=1980, color_continuous_scale=px.colors.sequential.Viridis)

fig.show()
plt.rcParams['figure.figsize'] = (18, 9)

plt.style.use('fivethirtyeight')



sns.countplot(data['AreaName'], palette = 'hsv')

plt.title('Area from where People migrate to Canada', fontsize = 20, fontweight = 100)

plt.xticks(rotation = 90)

plt.show()
from wordcloud import WordCloud



wc = WordCloud(background_color = 'lightgray',

              width = 2000,

              height = 2000,

              colormap = 'magma',

              max_words = 70).generate(str(data['OdName']))



plt.rcParams['figure.figsize'] = (15, 15)

plt.title('Countries from Where People Migrate to Canada', fontsize = 30, fontweight = 10)



plt.imshow(wc)

plt.axis('off')



plt.show()



import plotly.express as px

gapminder = px.data.gapminder()





fig = px.scatter_geo(gapminder, locations="iso_alpha", color="continent", hover_name="country", size="pop",

               animation_frame="year", projection="natural earth")

fig.show()


plt.style.use('_classic_test')



colors = plt.cm.cool(np.linspace(0, 50, 100))

data['DevName'].value_counts().plot.pie(colors = colors,

                                       figsize = (10, 10))



plt.title('Types of Countries', fontsize = 20, fontweight = 30)

plt.axis('off')



plt.legend()

plt.show()

import plotly.io as pio



pio.templates.default = "seaborn"



data.rename(index = int, columns = {1980:"Ninety-Eighty", 2013:"Twenty-Thirteen"}, inplace = True)

fig = px.scatter(data, x='Ninety-Eighty', y='Twenty-Thirteen', facet_col="DevName",

                 width=800, height=700)



fig.update_layout(

    margin=dict(l=100, r=100, t=100, b=100),

    paper_bgcolor="LightBlue",

)



fig.update_traces(marker=dict(size=12,

                              line=dict(width=2,

                                        color='DarkSlateGrey')),

                  selector=dict(mode='markers'))



fig.show()

# data cleaning



# let's remove the columns which are not required

data = data.drop(['AREA','REG','DEV','Type','Coverage'], axis = 1)



# adding a Total column to add more information

data['Total'] = data.sum(axis = 1)



# let's check the head of the cleaned data

data.head()
# download countries geojson file



!wget --quiet https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DV0101EN/labs/Data_Files/world_countries.json -O world_countries.json

    

print('GeoJSON file downloaded!')
world_geo = r'world_countries.json' # geojson file



# create a plain world map

world_map = folium.Map(location=[0, 0], zoom_start=2, tiles='Mapbox Bright')



import warnings

warnings.filterwarnings('ignore')



# generate choropleth map using the total immigration of each country to Canada from 1980 to 2013

world_map.choropleth(

    geo_data=world_geo,

    data=data,

    columns=['OdName', 'Total'],

    key_on='feature.properties.name',

    fill_color='Greens', 

    fill_opacity=0.7, 

    line_opacity=0.2,

    legend_name='Immigration to Canada'

)



# display map

world_map


















































































