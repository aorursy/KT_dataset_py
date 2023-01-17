#importing python packages

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sb

import folium as fm

import plotly.express as px

import plotly.graph_objects as go

import cufflinks as cf

import json 



%matplotlib inline



print('packages imported')
#to is the name of the dataframe for Toronto.

to = pd.read_csv('../input/covid19-toronto/COVID19 cases.csv')



#printing the first five rows of the dataset.

to.head()
#Information on dataset

to.info()
# Converted date columns to date datatypes. They were previously objects. 

to['Reported Date']= pd.to_datetime(to['Reported Date'])

to['Episode Date'] = pd.to_datetime(to['Episode Date'])

to['FSA'] = to['FSA'].astype('str')

to.dtypes.reset_index()
# Added a column for month and day. 

to['Month'] = to['Reported Date'].apply(lambda time: time.month)

to['Day'] = to['Reported Date'].apply(lambda time: time.dayofweek)

to
#Checking the quantity of null values.

to.isnull().sum()
# the two columns are missning values and both rely on postal codes.

# filling null values with 'mpc' which means missing postal codes.

to.fillna({'Neighbourhood Name':'mpc', 'FSA': 'mpc'})
# Number of rows and columns. 

to.shape
to['Source of Infection'].value_counts()
to['Age Group'].value_counts()
# Generated bar graph based on the outcome of patient. 

plt.style.use('seaborn')



color = plt.cm.winter(np.linspace(0, 10, 20))

to['Outcome'].value_counts().plot.bar(color = color, figsize = (15, 8))



plt.title('Outcome of each patient',fontsize = 20)

plt.xticks(rotation = 90)

plt.show()
plt.figure(figsize = (15, 10))

ax = sb.countplot(x="Source of Infection", data=to, palette="viridis")
plt.figure(figsize = (15, 10))

sb.countplot(y= 'Neighbourhood Name', data = to, order = to['Neighbourhood Name'].value_counts().iloc[:10].index)
Month_count = []



for i in to.Month.unique():

    Month_count.append(len(to[to['Month']==i]))



plt.figure(figsize=(10,5))

sb.pointplot(x=to.Month.unique(),y=Month_count,color='red',alpha=0.8)

plt.xlabel('Month',fontsize = 15,color='blue')

plt.xticks(rotation=45)

plt.ylabel('Values',fontsize = 15,color='blue')

plt.title('Coronavirus Progress',fontsize = 15,color='blue')

plt.grid()

plt.show()
#plotly chart which is interactive. 

to1 = to.groupby(["Age Group"]).count().reset_index()



fig = px.bar(to1,

             y=to.groupby(["Age Group"]).size(),

             x="Age Group",

             color='Age Group')

fig.show()
to2 = to.groupby(["Currently Hospitalized"]).count().reset_index()



fig = px.bar(to2,

             y=to.groupby(["Currently Hospitalized"]).size(),

             x="Currently Hospitalized",

             color="Currently Hospitalized")

fig.show()
to3 = to.groupby(["Ever in ICU"]).count().reset_index()



fig = px.bar(to3,

             y=to.groupby(["Ever in ICU"]).size(),

             x="Ever in ICU",

             color="Ever in ICU")

fig.show()
# Interactive pie chart showing the percentages of genders who are infected. 

labels=to['Client Gender'].unique()

values=to['Client Gender'].value_counts()

fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

fig.show()


t = to.FSA.value_counts()



table = pd.DataFrame(data=t.values, index=t.index, columns=['Count'])

table = table.reindex(['M2N','M3A','M4W','M2R','M1V','M2J','M4R',

'M2L','M5A','M6G','M4G','M2P','M1B','M5N','M6P','M5V','M8Y','M4J','M9B','M6R',

'M8W','M6H','M4S','M6E','M6K','M5R','M4N','M4P','M4K','M9C','M5M','M1C','M1S','M6B',

'M8V','M9W','M5P','M6S','M3L','M5T','M6J','M1K','M4A','M4E','M4C','M2M','M1W',

'M3B','M9R','M4V','M2K','M4B','M5S','M5H','M3K','M3C','M4L','M1E','M1P','M6A',

'M9A','M5J','M4Y','M6C','M5G','M1L','M1M','M3H','M3N','M1H','M1J','M4M','M5E',

'M8Z','M9V','M1R','M5B','M4X','M9L','M1N','M1T','M6N','M8X','M3J','M2H','M6L',

'M9M','M6M','M9N','M4H','M3M','M9P','M1G','M1X','M4T','M5C'])



table = table.reset_index()

table.rename({'index': 'FSA'}, axis='columns', inplace=True)



table

gjson = r'../input/toronto-map/torontomap.geojson'

to_map = fm.Map(location = [43.6532, -79.3832], zoom_start = 10)



to_map.choropleth(

    geo_data=gjson,

    data=table,

    columns=['FSA','Count'],

    key_on='feature.properties.CFSAUID',

    fill_color='YlOrRd', 

    fill_opacity=0.7, 

    line_opacity=0.2,

    legend_name='Rate of virus'

)





to_map