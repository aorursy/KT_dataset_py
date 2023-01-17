import pandas as pd

import numpy as np



import plotly.express as px

from wordcloud import WordCloud,STOPWORDS

from collections import Counter





import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv("../input/indian-food-101/indian_food.csv")
data.head(10)
data.tail(10)
data.shape
data.info()
data.describe(include= 'object')
for x in data.columns:

    data[x] = data[x].apply(lambda x: np.nan if(x=='-1' or x==-1)else x)
for x in data.columns:

    print('Null Value in {0} = {1} \npercent = {2:.2f}%\n'.format(x,data[x].isna().sum(),data[x].isna().sum()/len(data)))
# we can chose not to drop the nan values 

# there are lot of method avaliable to fill the nan value 

# but we can't use that because fill region and state require the proper research and documentation
for x in data.columns:

    print('Unique Values in  {0} = {1}'.format(x,data[x].nunique()))
data['diet'].value_counts()
fig = px.histogram(data,x='diet',color='diet',title='Unique Values in Diet Column')



fig.show()
data['flavor_profile'].value_counts()
fig = px.histogram(data.dropna(),x='flavor_profile',color= 'flavor_profile',title = "Unique Values in flavor_profile Column")



fig.show()
data['course'].value_counts()
fig = px.histogram(data,x='course',color='course',title = 'Unique Value in Course Column')



fig.show()
sweets = data['state'].value_counts()

sweets.values

sweets = data['state'].value_counts()

fig = px.choropleth(

    sweets,

    geojson="https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson",

    featureidkey='properties.ST_NM',

    locations=sweets.index,

    color=sweets.index,

    color_continuous_scale='Reds',

    height = 700,

    hover_name = sweets.values,

    title = 'States and Total Numbers of Cuisine'

)



fig.update_geos(fitbounds="locations", visible=False)



fig.show()
fig = px.histogram(data.dropna(),x='region',color = 'region',title = 'Region Contribution in Indian Cuisine')

fig.show()
sweet_state = data[data['flavor_profile'] == 'sweet']

sweets = sweet_state['state'].value_counts()

fig = px.choropleth(

    sweets,

    geojson="https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson",

    featureidkey='properties.ST_NM',

    locations=sweets.index,

    color=sweets.index,

    color_continuous_scale='Reds',

    height = 700,

    hover_name = sweets.values,

    title = 'States and Total Numbers of Sweets'

)



fig.update_geos(fitbounds="locations", visible=False)



fig.show()
stopwords = set(STOPWORDS) 

def WordCloudSW(values):

    wordcloud = WordCloud(width = 500, height = 300, 

                background_color ='white', 

                stopwords = stopwords, 

                min_font_size = 10).generate(values) 

    

    plt.figure(figsize=(19,9))



    plt.axis('off')

    plt.title("Ingredients")

    plt.imshow(wordcloud)

    plt.show()





ing = ''

for x in list(sweet_state['ingredients']):

    ing+=''.join(x.split(','))

WordCloudSW(ing)

veg = data[data['diet']=='vegetarian']



#lets see Which state have high number of Veg Cuisine

sweets = veg['state'].value_counts()

fig = px.choropleth(

    sweets,

    geojson="https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson",

    featureidkey='properties.ST_NM',

    locations=sweets.index,

    color=sweets.index,

    color_continuous_scale='Reds',

    height = 700,

    hover_name = sweets.values,

    title = 'States and Total Numbers of Veg Cuisine'

)



fig.update_geos(fitbounds="locations", visible=False)



fig.show()
fig = px.histogram(veg.dropna(),x='flavor_profile',color= 'flavor_profile',title='Vegetarian Cuisine Flavor Profile')



fig.show()
fig = px.histogram(data , x='prep_time',title = 'Total Estimated  Time for Cuisine Preperation')



fig.show()
fig = px.histogram(data , x='cook_time',title = 'Estimated Cooking Time for Cuisine')



fig.show()
ing = ''

for x in list(data['ingredients']):

    ing+=''.join(x.split(','))

WordCloudSW(ing)