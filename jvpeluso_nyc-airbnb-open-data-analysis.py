import os

import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import folium

from folium.plugins import MarkerCluster

import geopandas as gpd

from PIL import Image

import requests

from io import BytesIO

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



# ignore deprecation warnings in sklearn

import warnings

warnings.filterwarnings('ignore')



# Default configurations and constants

plt.style.use('seaborn-pastel')

pd.set_option('display.max_columns', 500)

pd.set_option('display.max_rows', 500)

colPlots = ['skyblue','lightgreen','lightcoral','gainsboro','cadetblue', 'khaki']

colGeoJson = [['#ffffcc','#fecc5c','#fd8d3c','#f03b20','#bd0026'],

              ['#ffffcc','#cceeff','#4da6ff','#1a1aff','#000080'],

              ['#ffffcc','#bbff99','#78c679','#31a354','#006837'],

              ['#d73027','#f46d43','#fdae61','#fee08b','#ffffbf','#d9ef8b','#a6d96a','#66bd63','#1a9850'],

              ['#f7fcf5','#e5f5e0','#c7e9c0','#a1d99b','#74c476','#41ab5d','#238b45','#006d2c','#00441b'],

              ['#80ff80','#ff9980','#80b3ff','#ff99ff', '#ffff66']]



print('Libraries imported and default configuration set!')
def cardinalN(i):

    

    """ With a number given, return it's cardinal suffix"""

    

    if i == 1:

        return '1st'

    elif i == 2:

        return '2nd'

    elif i == 3:

        return '3rd'

    else:

        return str(i) + 'th'

    
def foliumGJ(mMap, df, key, countV, lstColor,  aliases, vmin, vmax, name, step = 10):

    

    """ Function that create a choropleth map. 

    

    Arguments

    ---------

    map:        Folium map

    df:         Dataframe

    key:        Key to use with the GeoJson

    countV:     Varible with the amount to be used with the colormap

    lstColor:   Color list to create the colormap

    aliases:    Aliases for the popup

    vmin, vmax: Min and Max for the colormap

    step:       Gradient steps for the colormap



    """

    colormap = folium.LinearColormap(colors = lstColor,

                                     vmin = vmin, vmax = vmax).to_step(step)

    folium.GeoJson(df[['geometry',key ,countV]],

                   name = name,

                   style_function = lambda x: {"weight" : 0.25, 'color':'black','fillColor':colormap(x['properties'][countV]), 'fillOpacity' : 0.55},

                   highlight_function = lambda x: {'weight': 0.75 , 'color':'black', 'fillOpacity' : 0.85},

                   smooth_factor=2.0,

                   tooltip=folium.features.GeoJsonTooltip(fields=[key,countV],

                                                          aliases=aliases, 

                                                          labels=True)).add_to(mMap)



    return mMap
def plotBarV(X, y, fig, title, width, factor, nType = '', symbol = '', fontS = 10, bColor = 'skyblue'):



    """ Function that draw a tuned bar plot 

    

    Arguments

    ---------

    X, y:   Axis of the plot

    fig:    Figure to tune

    title:  Plot Title

    width:  Bar width

    factor: amount to add to bar height text 

    nType:  if is double (percentage or currency)

    symbol: if is double (percentage '%' or currency '$'/'€'/etc)

    fontS:  Font size of the lables that will be printed above the bars

    bColor: Color of the bars



    """

    plt.title(title, fontsize = 15)

    _ = plt.bar(X, y, width = 0.65, color = bColor, alpha = 0.65)

    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')

    plt.xticks(fontsize = 13)

    plt.grid(b=False)

    plt.rcParams['axes.facecolor'] = 'white'

    b,t = plt.ylim()

    plt.ylim(top=(t*1.15))



    for spine in plt.gca().spines.values():

        spine.set_visible(False)



    for bar in _:

        height = bar.get_height()

        if nType == 'double':

            txtHeight = str(np.around(height,decimals=2))+symbol

        else:

            txtHeight = str(height)

            

        plt.gca().text(bar.get_x() + bar.get_width()/2, (bar.get_height()+factor), txtHeight,

                       ha='center', color='black', fontsize=fontS)



    return fig
def autolabel(AX, rects, symbol='', fontS = 10):

    

    """Attach a text label above each bar in *rects*, displaying its height."""

    

    for rect in rects:

        height = round(rect.get_height(),2)

        AX.annotate('{:.2f}{}'.format(height, symbol),

                    xy = (rect.get_x() + rect.get_width() / 2, height),

                    xytext=(0, 3),  # 3 points vertical offset

                    textcoords="offset points", fontsize = fontS,

                    ha='center', va='bottom')
def autolabel2(AX, x ,y, text, fSize = 12, color = 'black'):

    

    """Display text in an AX on given coordinates"""

    

    AX.text(x,y, text, fontsize = fSize, color = color)
# Search files in the folder



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Import dataset of NYC Airbnb data in pandas dataframe

fullDF = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')



# Import NYC geoJsons

NYC_NGJ = gpd.read_file('/kaggle/input/nyc-borough-geojson/NYC_neighborhoods.geojson')

NYC_BGJ = gpd.read_file('/kaggle/input/nyc-borough-geojson/NYC_borough.geojson')



print('Data imported!')
fullDF.head()
# Check shape and statistics of the dataset

print(fullDF.shape)

fullDF.describe()
# Check if there're NaN values

fullDF.info()
# As we see, there're NaN values in several features, let's see the list

fullDF.isna().sum()
#Create figure

fig1 = plt.figure(figsize=(22, 7))



# Create the data

tmpDF = fullDF.groupby('neighbourhood_group').count()['id']

strTitle = 'Distribution of rooms per Borough'



# Plot figure

plotBarV(tmpDF.index, tmpDF, fig1.add_subplot(1,2,1), strTitle,

         0.65, 200, 'int', fontS = 12)



# Create the data

tmpDF = fullDF.groupby('neighbourhood_group').agg({'price' : np.mean}).reset_index()

strTitle = 'Average price per Borough'



# Plot figure

plotBarV(tmpDF['neighbourhood_group'], tmpDF['price'],  fig1.add_subplot(1,2,2), strTitle,

         0.65, 2, nType = 'double', symbol = '$', bColor = colPlots[1], fontS = 12)



plt.show()
# Create the map



mMap = folium.Map(location=[40.683356, -73.911270], zoom_start = 10, width = 600, height = 600, tiles='cartodbpositron')



# Create the data (rooms)

NYC_rooms = NYC_BGJ.merge(fullDF.groupby('neighbourhood_group').count()['id'].reset_index(), 

                          left_on='borough', right_on='neighbourhood_group', how='inner').fillna(0)



# Add geoJson density layer

mMap = foliumGJ(mMap, df = NYC_rooms, key = 'borough', 

                countV = 'id', lstColor = colGeoJson[0], 

                aliases = ['Neighborhood:','Airbnbs:'], 

                vmin = 0, vmax = int(NYC_rooms['id'].max()),

                name = 'Airbnb places in NYC')



# Create the data (prices)

tmpDF = fullDF.groupby('neighbourhood_group').agg({'price' : np.mean}).reset_index()

vMin = round(tmpDF['price'].min(),0)

vMax = round(tmpDF['price'].max(),0)

tmpDF['price'] = tmpDF['price'].map('{:,.2f}'.format)

tmpDF['price'] = tmpDF['price'].astype(float)



NYC_pricesB = NYC_BGJ.merge(tmpDF, left_on='borough',

                            right_on='neighbourhood_group', how='inner').fillna(0)



# Add geoJson density layer

mMap = foliumGJ(mMap, df = NYC_pricesB, key = 'borough', 

                countV = 'price', lstColor = colGeoJson[1], 

                aliases = ['Neighborhood:','Average Price:'], 

                vmin = vMin, vmax = vMax, step = 5,

                name = 'Airbnb average price in NYC')



# Add title and layer control

title_html = '''

             <h3 align="left" style="font-size:16px"><b>NYC boroughs (Airbnb places and average price)</b></h3>

             '''

mMap.get_root().html.add_child(folium.Element(title_html))

mMap.add_child(folium.map.LayerControl(collapsed=False))



# Show map

mMap
# Create figure

fig1 = plt.figure(figsize=(12, 7))



# Create the data

tmpDF = fullDF.groupby('room_type').count()['id']

strTitle = 'Room type distribution'



# Plot figure

plotBarV(tmpDF.index, tmpDF, fig1.add_subplot(1,2,1), strTitle,

         0.65, 200, fontS = 12)



# Create the data

tmpDF = fullDF.groupby('room_type').agg({'price' : np.mean}).reset_index()

strTitle = 'Average price per Airbnb room type'



# Plot figure

plotBarV(tmpDF['room_type'], tmpDF['price'], fig1.add_subplot(1,2,2), strTitle,

         0.65, 2, nType = 'double', symbol = '$', bColor = colPlots[1], fontS = 12)



plt.show()
# Create the data

tmpDF = fullDF.groupby(['neighbourhood_group', 'room_type']).count()['id'].reset_index()

tmpDF = tmpDF.pivot(index='neighbourhood_group', columns='room_type', values='id').fillna(0)

X_labels = list(tmpDF.index)

tmpDF.reset_index(inplace=True)



tmpDF['totalNG'] = tmpDF.sum(axis=1)

tmpDF['Entire home/apt'] = tmpDF['Entire home/apt'].div(tmpDF['totalNG']).mul(100)

tmpDF['Private room'] = tmpDF['Private room'].div(tmpDF['totalNG']).mul(100)

tmpDF['Shared room'] = tmpDF['Shared room'].div(tmpDF['totalNG']).mul(100)



X = np.arange(len(X_labels))

width = 0.25



# Create figure and define properties

fig, ax = plt.subplots(figsize=(22,7))

plt.title('Airbnb room type distribution per borough', fontsize=15)



rects1 = ax.bar(X - 0.25, tmpDF['Entire home/apt'], width, label='Entire home/apt', color = colPlots[0], alpha = 0.65)

rects2 = ax.bar(X , tmpDF['Private room'], width, label='Private room', color = colPlots[1], alpha = 0.65)

rects3 = ax.bar(X + 0.25, tmpDF['Shared room'], width, label='Shared room', color = colPlots[2], alpha = 0.65)



for spine in plt.gca().spines.values():

    spine.set_visible(False)

    

plt.grid(b=False)

plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')

plt.xticks(fontsize = 13)

plt.rcParams['axes.facecolor'] = 'white'

b,t = plt.ylim()

plt.ylim(top=(t*1.15))

plt.legend(frameon=False, bbox_to_anchor=(0.91,1.01), loc="upper left", fontsize = 12)

ax.set_xticks(X)

ax.set_xticklabels(X_labels)



# Put values on top of every bar

autolabel(ax, rects1, '%', fontS = 12)

autolabel(ax, rects2, '%', fontS = 12)

autolabel(ax, rects3, '%', fontS = 12)
# Create the data

tmpDF = fullDF.groupby(['room_type', 'neighbourhood_group']).agg({'price' : np.mean}).reset_index()

tmpDF['price'] = tmpDF['price'].map('{:,.2f}'.format)

tmpDF['price'] = tmpDF['price'].astype(float)

tmpDF = tmpDF.pivot(index='room_type', columns='neighbourhood_group', values='price').fillna(0)

X_labels = list(tmpDF.index)

tmpDF.reset_index(inplace=True)



X = np.arange(len(X_labels))

width = 0.15



# Create figure, define properties and plot it

fig, ax = plt.subplots(figsize=(22,7))

plt.title('Airbnb room type price per borough', fontsize=15)



rects1 = ax.bar(X - 0.3, tmpDF['Bronx'], width, label='Bronx', color = colPlots[0], alpha = 0.65)

rects2 = ax.bar(X - 0.15, tmpDF['Brooklyn'], width, label='Brooklyn', color = colPlots[1], alpha = 0.65)

rects3 = ax.bar(X , tmpDF['Manhattan'], width, label='Manhattan', color = colPlots[2], alpha = 0.65)

rects4 = ax.bar(X + 0.15, tmpDF['Queens'], width, label='Queens', color = colPlots[3], alpha = 0.65)

rects5 = ax.bar(X + 0.3, tmpDF['Staten Island'], width, label='Staten Island', color = colPlots[4], alpha = 0.65)



for spine in plt.gca().spines.values():

    spine.set_visible(False)

    

plt.grid(b=False)

plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')

plt.xticks(fontsize = 11)

plt.rcParams['axes.facecolor'] = 'white'

b,t = plt.ylim()

plt.ylim(top=(t*1.15))

plt.legend(frameon=False, bbox_to_anchor=(0.85,1), loc="upper left", fontsize = 12)

ax.set_xticks(X)

ax.set_xticklabels(X_labels)



# Put values on top of every bar

autolabel(ax, rects1, '$', fontS = 12)

autolabel(ax, rects2, '$', fontS = 12)

autolabel(ax, rects3, '$', fontS = 12)

autolabel(ax, rects4, '$', fontS = 12)

autolabel(ax, rects5, '$', fontS = 12)
# Create the map

mMap = folium.Map(location=[40.683356, -73.911270], zoom_start = 10, width = 600, height = 600, tiles='cartodbpositron')



# Create the data



# Entire Home

EH= tmpDF.iloc[0,:].reset_index()

EH.columns = ['borough', 'price']

EH.drop(0, inplace=True)

NYC_EH = NYC_BGJ.merge(EH, left_on='borough', right_on='borough', how='inner').fillna(0)

vMinEH = round(NYC_EH['price'].min(),0)

vMaxEH = round(NYC_EH['price'].max(),0)



# Private Room

PR = tmpDF.iloc[1,:].reset_index()

PR.columns = ['borough', 'price']

PR.drop(0, inplace=True)

NYC_PR = NYC_BGJ.merge(PR, left_on='borough', right_on='borough', how='inner').fillna(0)

vMinPR = round(NYC_PR['price'].min(),0)

vMaxPR = round(NYC_PR['price'].max(),0)



# Shared Room

SR = tmpDF.iloc[1,:].reset_index()

SR.columns = ['borough', 'price']

SR.drop(0, inplace=True)

NYC_SR = NYC_BGJ.merge(SR, left_on='borough', right_on='borough', how='inner').fillna(0)

vMinSR = round(NYC_SR['price'].min(),0)

vMaxSR = round(NYC_SR['price'].max(),0)



# Add geoJson density layers

mMap = foliumGJ(mMap, df = NYC_EH, key = 'borough', 

                countV = 'price', lstColor = colGeoJson[0], 

                aliases = ['Borough:','Average price:'], 

                vmin = vMinEH, vmax = vMaxEH,

                name = 'Entire home/apt')



mMap = foliumGJ(mMap, df = NYC_PR, key = 'borough', 

                countV = 'price', lstColor = colGeoJson[1], 

                aliases = ['Borough:','Average price:'], 

                vmin = vMinPR, vmax = vMaxPR,

                name = 'Private room')



mMap = foliumGJ(mMap, df = NYC_SR, key = 'borough', 

                countV = 'price', lstColor = colGeoJson[2], 

                aliases = ['Borough:','Average price:'], 

                vmin = vMinSR, vmax = vMaxSR,

                name = 'Shared room')



# Add title and layer control

title_html = '''

             <h3 align="left" style="font-size:16px"><b>NYC boroughs (Airbnb places and average price per Room type)</b></h3>

             '''

mMap.get_root().html.add_child(folium.Element(title_html))

mMap.add_child(folium.map.LayerControl(collapsed=False))



# Show map

mMap
# Create the data

brooklynDF_R = fullDF[fullDF['neighbourhood_group'] == 'Brooklyn'].groupby(['neighbourhood']).count()['id'].reset_index()

brooklynDF_R['borough'] = 'Brooklyn'

brooklynDF_R2 = brooklynDF_R.sort_values('id', ascending=False).head(10)

brooklynDFlst = list(brooklynDF_R2['neighbourhood'])



manhattanDF_R = fullDF[fullDF['neighbourhood_group'] == 'Manhattan'].groupby(['neighbourhood']).count()['id'].reset_index()

manhattanDF_R['borough'] = 'Brooklyn'

manhattanDF_R2 = manhattanDF_R.sort_values('id', ascending=False).head(10)

manhattanDFlst = list(manhattanDF_R2['neighbourhood'])



# Create figure

fig1 = plt.figure(figsize=(22, 7))



# Define first subplot, define properties and plot it

fig1.add_subplot(1,2,1)

_ = plt.barh(brooklynDF_R2['neighbourhood'], brooklynDF_R2['id'], color = colPlots[0], height = 0.85, alpha = 0.65)

for spine in plt.gca().spines.values():

    spine.set_visible(False)

plt.grid(b=False)

plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='on', labelbottom='off')

plt.yticks(fontsize = 12.5)

plt.rcParams['axes.facecolor'] = 'white'

plt.title('Brooklyn top-10 Airbnb Neighborhoods', fontsize=15)

for bar in _:

    width = bar.get_width()

    if width < 1000 : 

        sep = 105

    else:

        sep = 130

    plt.gca().text((bar.get_width()-sep), bar.get_y() + bar.get_height()/3, str(width),

                   ha='center', color='black', fontsize=12)



# Define second subplot, define properties and plot it

ax = fig1.add_subplot(1,2,2)

_2 = plt.barh(manhattanDF_R2['neighbourhood'], manhattanDF_R2['id'], color = colPlots[1], height = 0.85, alpha = 0.65)

for spine in plt.gca().spines.values():

    spine.set_visible(False)

plt.grid(b=False)

plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='on', labelright='off',  labelbottom='off')

plt.yticks(fontsize = 12.5)

plt.rcParams['axes.facecolor'] = 'white'

plt.title('Manhattan top-10 Airbnb Neighborhoods', fontsize=15)

for bar in _2:

    width = bar.get_width()

    if width < 1000 : 

        sep = 75

    else:

        sep = 90

    plt.gca().text((bar.get_width()-sep), bar.get_y() + bar.get_height()/3, str(width),

                   ha='center', color='black', fontsize=12)

plt.show()
# Create the data

brooklynDF_P = fullDF[fullDF['neighbourhood_group'] == 'Brooklyn'].groupby('neighbourhood').agg({'price' : np.mean}).reset_index()

brooklynDF_P['borough'] = 'Brooklyn'

brooklynDF_P['price'] = brooklynDF_P['price'].map('{:,.2f}'.format)

brooklynDF_P['price'] = brooklynDF_P['price'].astype(float)

brooklynDF_P2 = brooklynDF_P.sort_values('price', ascending=False).head(10)

brooklynDFlst = list(brooklynDF_P2['neighbourhood'])



manhattanDF_P = fullDF[fullDF['neighbourhood_group'] == 'Manhattan'].groupby('neighbourhood').agg({'price' : np.mean}).reset_index()

manhattanDF_P['borough'] = 'Manhattan'

manhattanDF_P['price'] = manhattanDF_P['price'].map('{:,.2f}'.format)

manhattanDF_P['price'] = manhattanDF_P['price'].astype(float)

manhattanDF_P2 = manhattanDF_P.sort_values('price', ascending=False).head(10)

manhattanDFlst = list(manhattanDF_P2['neighbourhood'])



# Create figure

fig1 = plt.figure(figsize=(22, 7))



# Define first subplot, define properties and plot it

fig1.add_subplot(1,2,1)

_ = plt.barh(brooklynDF_P2['neighbourhood'], brooklynDF_P2['price'], color = colPlots[0], height = 0.85, alpha = 0.65)

for spine in plt.gca().spines.values():

    spine.set_visible(False)

plt.grid(b=False)

plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='on', labelbottom='off')

plt.yticks(fontsize = 12.5)

plt.rcParams['axes.facecolor'] = 'white'

plt.title('Brooklyn top-10 Airbnb Neighborhoods', fontsize=15)

for bar in _:

    sep = 25

    width = bar.get_width()

    width = str(width) + '$'

    plt.gca().text((bar.get_width()-sep), bar.get_y() + bar.get_height()/3, str(width),

                   ha='center', color='black', fontsize=12)



# Define second subplot, define properties and plot it

ax = fig1.add_subplot(1,2,2)

_2 = plt.barh(manhattanDF_P2['neighbourhood'], manhattanDF_P2['price'], color = colPlots[1], height = 0.85, alpha = 0.65)

for spine in plt.gca().spines.values():

    spine.set_visible(False)

plt.grid(b=False)

plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='on', labelright='off',  labelbottom='off')

plt.yticks(fontsize = 12.5)

plt.rcParams['axes.facecolor'] = 'white'

plt.title('Manhattan top-10 Airbnb Neighborhoods', fontsize=15)

for bar in _2:

    sep=25

    width = bar.get_width()

    width = str(width) + '$'

    plt.gca().text((bar.get_width()-sep), bar.get_y() + bar.get_height()/3, str(width),

                   ha='center', color='black', fontsize=12)

plt.show()
# Create map

mMap = folium.Map(location=[40.649249, -73.950485], zoom_start = 11, width = 600, height = 600, tiles='cartodbpositron')



# Create the data

brooklynDF_R.columns = ['neighbourhood', 'count', 'borough']

NYC_Brooklyn_R = NYC_NGJ[NYC_NGJ['borough'] == 'Brooklyn']

NYC_Brooklyn_R = NYC_Brooklyn_R.merge(brooklynDF_R, left_on='neighborhood', right_on='neighbourhood', how='left').fillna(0)

NYC_Brooklyn_R.drop(67, inplace=True)

vMaxR = int(NYC_Brooklyn_R['count'].max())



NYC_Brooklyn_P = NYC_NGJ[NYC_NGJ['borough'] == 'Brooklyn']

NYC_Brooklyn_P = NYC_Brooklyn_P.merge(brooklynDF_P, left_on='neighborhood', right_on='neighbourhood', how='left').fillna(0)

NYC_Brooklyn_P.drop(67, inplace=True)

vMaxP = round(NYC_Brooklyn_P['price'].max(),0)



# Add geoJson density layer

mMap = foliumGJ(mMap, df = NYC_Brooklyn_R, key = 'neighborhood', 

                countV = 'count', lstColor = colGeoJson[0], 

                aliases = ['Neighborhood:','Airbnbs:'], 

                vmin = 0, vmax = vMaxR,

                name = 'Airbnb places per neighborhood', step = 5)

                    

# Add geoJson density layer

mMap = foliumGJ(mMap, df = NYC_Brooklyn_P, key = 'neighborhood', 

                countV = 'price', lstColor = colGeoJson[1], 

                aliases = ['Neighborhood:','Price:'], 

                vmin = 0, vmax = vMaxP,

                name = 'Average price per neighborhood', step = 5)



# Add title and layer control

title_html = '''

             <h3 align="left" style="font-size:16px"><b>Brooklyn (Airbnb places and average price)</b></h3>

             '''

mMap.get_root().html.add_child(folium.Element(title_html))

mMap.add_child(folium.map.LayerControl(collapsed=False))



# Show map

mMap
# Create map

mMap = folium.Map(location=[40.791685, -73.938714], zoom_start = 11, width = 600, height = 600, tiles='cartodbpositron')



# Create the data

manhattanDF_R.columns = ['neighbourhood', 'count', 'borough']

NYC_Manhattan_R = NYC_NGJ[NYC_NGJ['borough'] == 'Manhattan']

NYC_Manhattan_R = NYC_Manhattan_R.merge(manhattanDF_R, left_on='neighborhood', right_on='neighbourhood', how='left').fillna(0)

vMaxR = int(NYC_Manhattan_R['count'].max())



NYC_Manhattan_P = NYC_NGJ[NYC_NGJ['borough'] == 'Manhattan']

NYC_Manhattan_P = NYC_Manhattan_P.merge(manhattanDF_P, left_on='neighborhood', right_on='neighbourhood', how='left').fillna(0)

vMaxP = round(NYC_Manhattan_P['price'].max(),0)



# Add geoJson density layer

mMap = foliumGJ(mMap, df = NYC_Manhattan_R, key = 'neighborhood', 

                countV = 'count', lstColor = colGeoJson[0], 

                aliases = ['Neighborhood:','Airbnbs:'], 

                vmin = 0, vmax = vMaxR,

                name = 'Airbnb places per neighborhood', step = 5)

                    

# Add geoJson density layer

mMap = foliumGJ(mMap, df = NYC_Manhattan_P, key = 'neighborhood', 

                countV = 'price', lstColor = colGeoJson[1], 

                aliases = ['Neighborhood:','Price:'], 

                vmin = 0, vmax = vMaxP,

                name = 'Average price per neighborhood', step = 5)



# Add title and layer control

title_html = '''

             <h3 align="left" style="font-size:16px"><b>Manhattan (Airbnb places and average price)</b></h3>

             '''

mMap.get_root().html.add_child(folium.Element(title_html))

mMap.add_child(folium.map.LayerControl(collapsed=False))



# Show map

mMap
# Create the data

fullDF.loc[fullDF['number_of_reviews']<=0, 'have_rev'] = 0

fullDF.loc[fullDF['number_of_reviews']> 0, 'have_rev'] = 1



# Create figure and define properties

fig = plt.figure(figsize=(7,7))

plt.pie([fullDF[fullDF['have_rev'] == 1].count()['id'], fullDF[fullDF['have_rev'] == 0].count()['id']], 

         startangle = 90, textprops={'size': 12}, autopct='%1.1f%%', wedgeprops={'alpha' : 0.65}, 

         explode = (0, 0.025), labels = ['',''], colors = [colPlots[0], colPlots[1]])



# Plot the figure

plt.axis('off')

plt.legend(['With reviews','No reviews'],loc=4, frameon=False, fontsize = 12)

plt.title('Airbnb places with/without reviews', fontsize=15)

plt.show()
# Create the data

tmpDF = fullDF.groupby(['neighbourhood_group', 'have_rev']).count()['id'].reset_index()

tmpDF = tmpDF.pivot(index='neighbourhood_group', columns='have_rev', values='id').fillna(0)

X_labels = list(tmpDF.index)

tmpDF.reset_index(inplace=True)

tmpDF.columns = ['Borough', 'No Reviews', 'With Reviews']



tmpDF['totalR'] = tmpDF.sum(axis=1)

tmpDF['No Reviews'] = tmpDF['No Reviews'].div(tmpDF['totalR']).mul(100)

tmpDF['With Reviews'] = tmpDF['With Reviews'].div(tmpDF['totalR']).mul(100)



# Create the figures, define properties and plot it

fig = plt.figure(figsize=(22, 7))

for index, row in tmpDF.iterrows():

    ax = fig.add_subplot(1,5,index+1)

    _ = plt.pie([row['With Reviews'], row['No Reviews']], 

                startangle = 90, textprops={'size': 12.5}, autopct='%1.1f%%', 

                explode = (0, 0.025), wedgeprops={'alpha' : 0.65},

                labels = ['',''], colors = [colPlots[0], colPlots[1]])



    plt.axis('off')

    if index == 4 :

        plt.legend(['With reviews','No reviews'], bbox_to_anchor=(0.85,1), frameon=False, fontsize = 13)

    plt.title(row['Borough'], fontsize=14)
# Create the data

fullDF.loc[np.logical_and(fullDF['number_of_reviews']>0 ,fullDF['number_of_reviews']< 11), 'rev_range'] = '0' 

fullDF.loc[np.logical_and(fullDF['number_of_reviews']>10, fullDF['number_of_reviews']<= 25), 'rev_range'] = '1'

fullDF.loc[np.logical_and(fullDF['number_of_reviews']>25, fullDF['number_of_reviews']<= 50), 'rev_range'] = '2'

fullDF.loc[fullDF['number_of_reviews']> 50, 'rev_range'] = '3'



lstVal = list(fullDF.groupby('rev_range').count().iloc[:,0].reset_index().sort_values(by='rev_range')['id'])

lstLbl = ['Less than 10',' 10 - 25','26 - 50','More than 50']

lstColors = [colPlots[0], colPlots[1],colPlots[2], colPlots[3]]



# Create the figures, define properties and plot it

fig1, ax1 = plt.subplots(figsize = (6.5,6.5))

plt.pie(lstVal, autopct='%1.1f%%', startangle=90, pctdistance=0.65, 

        textprops={'size': 12}, colors = lstColors, explode = (0, 0.015, 0.01, 0.015),

        wedgeprops={'alpha' : 0.65})



#centre_circle = plt.Circle((0,0),0.55,fc='white') # draw circle

#fig = plt.gcf()

#fig.gca().add_artist(centre_circle) # Equal aspect ratio ensures that pie is drawn as a circle



ax1.axis('equal') 

plt.axis('off')

plt.legend(lstLbl,frameon=False, bbox_to_anchor=(0.915,1), loc="upper left", fontsize = 12)

plt.title('Airbnb places reviews distribution', fontsize=15)

plt.tight_layout()

plt.show()
# Create the data

tmpDF = fullDF.groupby(['neighbourhood_group','rev_range']).count().iloc[:,0].reset_index().sort_values(by='neighbourhood_group')

tmpDF = tmpDF.pivot(index='neighbourhood_group', columns='rev_range', values='id').fillna(0).reset_index()



tmpDF['totalNG'] = tmpDF.sum(axis=1)

for i in range(0,4):

    tmpDF[str(i)] = tmpDF[str(i)].div(tmpDF['totalNG']).mul(100)



X = np.arange(len(tmpDF))

width = 0.2



# Create the figures, define properties and plot it

fig, ax = plt.subplots(figsize=(22,7))

plt.title('Airbnb places reviews per borough', fontsize=15)



rects1 = ax.bar(X - 0.3, tmpDF['0'], width, label='Less than 10', color = colPlots[0], alpha = 0.65)

rects2 = ax.bar(X - 0.1, tmpDF['1'], width, label='10 - 25', color = colPlots[1], alpha = 0.65)

rects3 = ax.bar(X + 0.1, tmpDF['2'], width, label='26 - 50', color = colPlots[2], alpha = 0.65)

rects4 = ax.bar(X + 0.3, tmpDF['3'], width, label='More than 50', color = colPlots[3], alpha = 0.65)



for spine in plt.gca().spines.values():

    spine.set_visible(False)

    

plt.grid(b=False)

plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')

plt.xticks(fontsize = 13)

plt.rcParams['axes.facecolor'] = 'white'

b,t = plt.ylim()

plt.ylim(top=(t*1.15))

plt.legend(frameon=False, bbox_to_anchor=(0.9,1), loc="upper left", fontsize = 12)

ax.set_xticks(X)

ax.set_xticklabels(X_labels)



autolabel(ax, rects1, '%', 11)

autolabel(ax, rects2, '%', 11)

autolabel(ax, rects3, '%', 11)

autolabel(ax, rects4, '%', 11)


# Create the data

lstCols = ['neighbourhood_group', 'neighbourhood', 'latitude', 'longitude', 'room_type', 'price', 'number_of_reviews']

tmpDF_ = fullDF[lstCols].sort_values(by='number_of_reviews', ascending = False).reset_index(drop=True).head(100)

tmpDF_.columns = ['borough', 'neighborhood', 'lat', 'long', 'room_type', 'price', 'numRev']



borDF = pd.DataFrame(tmpDF_.groupby('borough').count()['room_type']).reset_index()

borDF.columns = ('borough', 'value')

borDF = NYC_BGJ.merge(borDF, left_on='borough', right_on='borough', how='left').fillna(0)



# Colors dictionary to paint markers

colorDict = {'1-10' : 'red', 

             '11-25' : 'cadetblue',

             '26-50' : 'lightblue', 

             '51-100' : 'lightgreen'}



# Create the map and the MarkerCluster

mMap = folium.Map(location=[40.683356, -73.911270], zoom_start = 10, width = 600, height = 600, tiles='cartodbpositron')

mc = MarkerCluster(name = 'aaa').add_to(mMap)



# Add borough boundings

colormap = folium.LinearColormap(colors = colGeoJson[5],

                                 vmin = 1, vmax = 5).to_step(5)



folium.GeoJson(borDF[['geometry','borough' ,'boroughCode', 'value']],

               style_function = lambda x: {"weight" : 0.15, 'color':'black','fillColor':colormap(x['properties']['boroughCode']), 'fillOpacity' : 0.5},

               smooth_factor=2.0,

               tooltip=folium.features.GeoJsonTooltip(fields=['borough' ,'value'],

                                                      aliases=['Borough: ', 'Airbnb places: '], 

                                                      labels=True)).add_to(mMap)



# Place markers

for index, row in tmpDF_.iterrows():

    

    if index < 10 :

        markCol = '1-10'

    elif index < 25:

        markCol = '11-25'

    elif index < 50:

        markCol = '26-50'

    else:

        markCol = '51-100'

        

    toolTip = str(index+1) + ' of 100'

    strPopup = 'Borough: <b>%s</b><br>Neighborhood: <b>%s</b><br>Reviews: <b>%s</b><br>Room Type: <b>%s</b><br>Price: <b>%.2f$</b>'%(row['borough'], 

                                                                                                                                     row['neighborhood'],

                                                                                                                                     row['numRev'],

                                                                                                                                     row['room_type'],

                                                                                                                                     row['price'])

    

    folium.Marker(location=[row['lat'], row['long']],

                               tooltip = toolTip,

                               popup= folium.Popup(strPopup, min_width = 185, max_width = 115),

                               icon=folium.Icon( prefix='fa', icon='home', color=colorDict[markCol])).add_to(mc)

    



# Create legend

legend_html =   '''

                <div style="position: fixed; 

                            background-color:white;

                            top: 0px; left: 625px; width: 150px; height: 122px; 

                            border:2px solid grey; z-index:9999; font-size:12px;

                            ">&nbsp; <b>Ranking</b> <br>

                              &nbsp; <i class="fa fa-home fa-2x" style="color:red"></i> &nbsp; <b>1-10 reviews</b><br>

                              &nbsp; <i class="fa fa-home fa-2x" style="color:cadetblue"></i> &nbsp; <b>11-25 reviews</b><br>

                              &nbsp; <i class="fa fa-home fa-2x" style="color:lightblue"></i> &nbsp; <b>26-50 reviews</b><br>

                              &nbsp; <i class="fa fa-home fa-2x" style="color:lightgreen"></i> &nbsp; <b>51-100 reviews</b>

                </div>

                ''' 

mMap.get_root().html.add_child(folium.Element(legend_html))



title_html = '''

             <h3 align="left" style="font-size:16px"><b>Top-100 Airbnbs in NYC</b></h3>

             '''

mMap.get_root().html.add_child(folium.Element(title_html))

    

mMap
print('----------------------------------------------------------')

for index, row in tmpDF_.head(10).iterrows():

    if index == 9 :

        print('{} place --> Borough : {}'.format(cardinalN(index+1), row['borough']))

    else :

        print('{} place -->  Borough : {}'.format(cardinalN(index+1), row['borough']))

    print('               Neighborhood: {}'.format(row['neighborhood']))

    print('               Room Type: {}'.format(row['room_type']))

    print('               Price: {:.2f}$'.format(row['price']))

    print('               Reviews: {}'.format(row['numRev']))

    print('----------------------------------------------------------')

    
# Create data

borDF = tmpDF_.groupby('room_type').count()['price'].reset_index()

lstValues = borDF['price']

lstLbl = borDF['room_type']

lstColors = [colPlots[1],colPlots[0],colPlots[2]] 



# Create figure and define properties

fig = plt.figure(figsize=(7,7))

plt.pie(lstValues,

         startangle = 12, textprops={'size': 12}, wedgeprops={'alpha' : 0.65}, 

         explode = (0.0075,0.0075,0.0075), colors = lstColors,  autopct='%1.1f%%',

         pctdistance = 1.125)



# Plot the figure

plt.axis('off')

plt.legend(lstLbl,frameon=False, bbox_to_anchor=(0.995,1), loc="upper left", fontsize = 12)

plt.title('Top-100 room type distribution', fontsize=15)

plt.show()
# Create data

rtDF = tmpDF_.groupby(['room_type', 'borough']).count()['price'].reset_index()

rtDF = rtDF.pivot(index='room_type', columns='borough', values='price').fillna(0)



# Create figure and define properties

ax = rtDF.plot.barh(stacked=True, figsize =(14,7), alpha = 0.65)



for spine in ax.spines:

    ax.spines[spine].set_visible(False)



ax.tick_params(axis=u'both', which=u'both',length=0, labelsize=12)

ax.legend(frameon = False, fontsize = 13, bbox_to_anchor=(1.25,1))

ax.get_xaxis().set_visible(False)

ax.set(ylabel='')

ax.set_title('Top-100 distribution by room type & borough', fontsize=15)



autolabel2(ax, 1.5,1.96,str(int(rtDF.loc['Shared room', 'Queens'])) + ' place')

autolabel2(ax, 7.5,0.96,str(int(rtDF.loc['Private room', 'Brooklyn'])) + ' places')

autolabel2(ax, 29.5,0.96,str(int(rtDF.loc['Private room', 'Manhattan'])) + ' places')

autolabel2(ax, 56,0.96,str(int(rtDF.loc['Private room', 'Queens'])) + ' places')

autolabel2(ax, 74,0.96,str(int(rtDF.loc['Private room', 'Staten Island'])) + ' place')

autolabel2(ax, 5.5,-0.04,str(int(rtDF.loc['Entire home/apt', 'Brooklyn'])) + ' places')

autolabel2(ax, 16.6,-0.04,str(int(rtDF.loc['Entire home/apt', 'Manhattan'])) + ' places')

autolabel2(ax, 27,-0.04,str(int(rtDF.loc['Entire home/apt', 'Queens'])) + ' places')

# Create data

tmpDF = fullDF[['neighbourhood_group', "minimum_nights"]]

tmpDF['minimum_nights'] = np.clip(tmpDF['minimum_nights'], 0, 100)



# Create figure and define properties

fig = plt.figure(figsize=(18,6))

_ = sns.violinplot(x="neighbourhood_group", y="minimum_nights", data=tmpDF, palette = colPlots[0:5])

_.set_xlabel('', fontsize = 15)

_.set_ylabel('Minimum Nights', fontsize = 15)

_.tick_params(axis='both', which='major', labelsize=13, length=0)

_.set(frame_on=False)

_.yaxis.set_label_coords(-0.05, 0.5)

plt.show()
# Create data

tmpDF = fullDF[['neighbourhood_group', 'minimum_nights', 'room_type']].sort_values(by='room_type')

tmpDF['minimum_nights'] = np.clip(tmpDF['minimum_nights'], 0, 100)



# Create figure and define properties

fig = plt.figure(figsize=(22,8))

_ = sns.violinplot(x="neighbourhood_group", y="minimum_nights", hue='room_type', data=tmpDF, palette = colPlots[0:5])

_.set_xlabel('')

_.set_ylabel('Minimum Nights', fontsize = 15)

_.tick_params(axis='both', which='major', labelsize=13, length=0)

_.set(frame_on=False)

_.yaxis.set_label_coords(-0.05, 0.5)

_.legend(frameon=False, bbox_to_anchor=(0.995,1), loc="upper left", fontsize = 12)

plt.show()
# Create the data

fullDF.loc[np.logical_and(fullDF['minimum_nights']>0 ,fullDF['minimum_nights']< 10), 'minN_range'] = '0' 

fullDF.loc[np.logical_and(fullDF['minimum_nights']>9, fullDF['minimum_nights']<= 25), 'minN_range'] = '1'

fullDF.loc[np.logical_and(fullDF['minimum_nights']>25, fullDF['minimum_nights']<= 50), 'minN_range'] = '2'

fullDF.loc[fullDF['minimum_nights']> 50, 'minN_range'] = '3'



lstVal = list(fullDF.groupby('minN_range').count().iloc[:,0].reset_index().sort_values(by='minN_range')['id'])

lstLbl = ['Less than 10',' 10 - 25','26 - 50','More than 50']

lstColors = [colPlots[0], colPlots[1],colPlots[2], colPlots[3]]



# Create the figures, define properties and plot it

fig1, ax1 = plt.subplots(figsize = (6.5,6.5))

plt.pie(lstVal, autopct='%1.1f%%', startangle=-270, pctdistance=1.15, 

        textprops={'size': 12}, colors = lstColors, explode = (0.015, 0.015, 0.015, 0.015),

        wedgeprops={'alpha' : 0.65})



ax1.axis('equal') 

plt.axis('off')

plt.legend(lstLbl,frameon=False, bbox_to_anchor=(0.995,1), loc="upper left", fontsize = 12)

plt.title('Minimum nights count distribution', fontsize=15)

plt.tight_layout()

plt.show()
# Create figure and define properties

fig = plt.figure(figsize=(7,7))

plt.pie([fullDF[fullDF['availability_365']>0].count()['id'], fullDF[fullDF['availability_365']==0].count()['id']], 

         startangle = 90, textprops={'size': 12}, autopct='%1.1f%%', wedgeprops={'alpha' : 0.65}, 

         explode = (0, 0.025), labels = ['',''], colors = [colPlots[0], colPlots[1]])



# Plot the figure

plt.axis('off')

plt.legend(['With value','Without value'],loc=4, frameon=False, fontsize = 12)

plt.title('Airbnb places with/without Availability 365 value', fontsize=15)

plt.show()
# Create the figures, define properties and plot it

fig = plt.figure(figsize=(9,7))

sns.distplot(fullDF[fullDF['availability_365']>0]['availability_365'], bins = 10, axlabel = 'Days available')



for spine in plt.gca().spines.values():

    spine.set_visible(False)

    

plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')

plt.xticks(fontsize = 13)

plt.title('Availability 365 distribution and KDE', fontsize=15)

plt.show()
# Create the data

hostDF = pd.DataFrame(fullDF.groupby('host_id').count()['id']).reset_index()



# Create figure and define properties

fig = plt.figure(figsize=(7,7))

plt.pie([hostDF[hostDF['id']==1].count()['id'], hostDF[hostDF['id']>1].count()['id']], 

         startangle = 90, textprops={'size': 12}, autopct='%1.1f%%', wedgeprops={'alpha' : 0.65}, 

         explode = (0, 0.025), labels = ['',''], colors = [colPlots[0], colPlots[1]])



# Plot the figure

plt.axis('off')

plt.legend(['1 place','More than 1 place'],bbox_to_anchor=(0.925,0.925), frameon=False, fontsize = 12)

plt.title('Distribution of hosts with 1 or more records', fontsize=15)

plt.show()
# Create the data

hostDF = hostDF[hostDF['id']>1].sort_values(by='id', ascending=False).head(20)

dctNames = {row[2] : row[3] for row in fullDF[fullDF['host_id'].isin(list(hostDF['host_id']))].values}

hostDF['host_name'] = hostDF['host_id'].map(dctNames)



# Create figure and define properties

fig1 = plt.figure(figsize=(18, 10))

_ = plt.barh(hostDF['host_name'], hostDF['id'], color = colPlots[0], height = 0.85, alpha = 0.65)



for spine in plt.gca().spines.values():

    spine.set_visible(False)

    

plt.grid(b=False)

plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='on', labelbottom='off')

plt.yticks(fontsize = 12.5)

plt.rcParams['axes.facecolor'] = 'white'

plt.title('Top-20 hosts', fontsize=15)



for bar in _:

    width = bar.get_width()

    plt.gca().text((bar.get_width()-5), bar.get_y() + bar.get_height()/3, str(width),

                   ha='center', color='black', fontsize=12)



plt.show()
# Create joined names variable

text = " ".join(str(name) for name in fullDF['name'])



# Create stopword list

stopwords = set(STOPWORDS)

stopwords.update(['home', 'apt', 'room', 'rent', 'private', 'apartment', 'bedroom', 'bed', 'NYC', 'house'])



# Create mask array

mask = np.array(Image.open('/kaggle/input/nyc-borough-geojson/NYC_Boroughs.png'))



# create coloring from image

image_colors = ImageColorGenerator(mask)



# Generate a word cloud image

wordcloud = WordCloud(stopwords=stopwords, background_color="white", mode="RGBA",

                      width = 1000, height = 1000, max_words = 450, mask=mask).generate(text)



# Display the generated image:

plt.figure(figsize=[12,12])

plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear")

plt.axis("off")



plt.show()