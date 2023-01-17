!pip install squarify
# for some basic operations

import numpy as np 

import pandas as pd 



# for visualizations

import matplotlib.pyplot as plt

import seaborn as sns

import folium

import squarify



# for providing path

import os

print(os.listdir("../input"))

# reading the dataset



data = pd.read_csv('../input/Police_Department_Incidents_-_Previous_Year__2016_.csv')



# check the shape of the data

data.shape







# checking the head of the data



data.head()







# describing the data



data.describe()







# checking if there are any null values



data.isnull().sum()







# filling the missing value in PdDistrict using the mode values



data['PdDistrict'].fillna(data['PdDistrict'].mode()[0], inplace = True)



data.isnull().any().any()









# plotting a tree map



y = data['Category'].value_counts().head(25)

    

plt.rcParams['figure.figsize'] = (15, 15)

plt.style.use('fivethirtyeight')



color = plt.cm.magma(np.linspace(0, 1, 15))

squarify.plot(sizes = y.values, label = y.index, alpha=.8, color = color)

plt.title('Tree Map for Top 25 Crimes', fontsize = 20)



plt.axis('off')

plt.show()
# description of the crime



from wordcloud import WordCloud



plt.rcParams['figure.figsize'] = (15, 15)

plt.style.use('fast')



wc = WordCloud(background_color = 'orange', width = 1500, height = 1500).generate(str(data['Descript']))

plt.title('Description of the Crime', fontsize = 20)



plt.imshow(wc)

plt.axis('off')

plt.show()
# Regions with count of crimes



plt.rcParams['figure.figsize'] = (20, 9)

plt.style.use('seaborn')



color = plt.cm.spring(np.linspace(0, 1, 15))

data['PdDistrict'].value_counts().plot.bar(color = color, figsize = (15, 10))



plt.title('District with Most Crime',fontsize = 30)



plt.xticks(rotation = 90)

plt.show()
# Regions with count of crimes



plt.rcParams['figure.figsize'] = (20, 9)

plt.style.use('seaborn')



color = plt.cm.ocean(np.linspace(0, 1, 15))

data['Address'].value_counts().head(10).plot.bar(color = color, figsize = (15, 10))



plt.title('Top 10 Addresses with the most Crime',fontsize = 20)



plt.xticks(rotation = 90)

plt.show()
# Regions with count of crimes



plt.style.use('seaborn')





data['DayOfWeek'].value_counts().head(15).plot.pie(figsize = (15, 8), explode = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1))



plt.title('Crime count on each day',fontsize = 20)



plt.xticks(rotation = 90)

plt.show()
# Regions with count of crimes



plt.style.use('seaborn')



color = plt.cm.winter(np.linspace(0, 10, 20))

data['Resolution'].value_counts().plot.bar(color = color, figsize = (15, 8))



plt.title('Resolutions for Crime',fontsize = 20)

plt.xticks(rotation = 90)

plt.show()
data['Date'] = pd.to_datetime(data['Date'])



data['Month'] = data['Date'].dt.month



plt.style.use('fivethirtyeight')

plt.rcParams['figure.figsize'] = (15, 8)



sns.countplot(data['Month'], palette = 'winter',)

plt.title('Monthly Crime Rate', fontsize = 20)



plt.show()
# checking the time at which crime occurs mostly



import warnings

warnings.filterwarnings('ignore')



color = plt.cm.twilight(np.linspace(0, 5, 100))

data['Time'].value_counts().head(5).plot.bar(color = color, figsize = (15, 9))



plt.title('Crime Occurrence Throughout The Day', fontsize = 20)

plt.show()


df = pd.crosstab(data['Category'], data['PdDistrict'])

color = plt.cm.Greys(np.linspace(0, 1, 10))



df.div(df.sum(1).astype(float), axis = 0).plot.bar(stacked = True, color = color, figsize = (18, 12))

plt.title('District vs Category of Crime', fontweight = 30, fontsize = 20)



plt.xticks(rotation = 90)

plt.show()
t = data.PdDistrict.value_counts()



table = pd.DataFrame(data=t.values, index=t.index, columns=['Count'])

#table = table.reindex(["CENTRAL", "NORTHERN", "PARK", "SOUTHERN", "MISSION", "TENDERLOIN", "RICHMOND", "TARAVAL", "INGLESIDE", "BAYVIEW"])



table = table.reset_index()

table.rename({'index': 'Neighborhood'}, axis='columns', inplace=True)



table

gjson = r'https://cocl.us/sanfran_geojson'

sf_map = folium.Map(location = [37.77, -122.42], zoom_start = 12)


#generate map

sf_map.choropleth(

    geo_data=gjson,

    data=table,

    columns=['Neighborhood', 'Count'],

    key_on='feature.properties.DISTRICT',

    fill_color='YlOrRd', 

    fill_opacity=0.7, 

    line_opacity=0.2,

    legend_name='Crime Rate in San Francisco'

)



sf_map
import gmplot

# For improved table display in the notebook

from IPython.display import display
