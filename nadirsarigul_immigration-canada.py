# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df = pd.read_excel("/kaggle/input/immigration-to-canada-ibm-dataset/Canada.xlsx",
                     sheet_name='Canada by Citizenship',
                     skiprows = range(20),
                     skipfooter = 2)
                     
           
                   
df.shape
df.head()
# print the dimensions of the dataframe
print(df.shape)
# clean up the dataset to remove unnecessary columns (eg. REG) 
df.drop(["AREA", "REG", "DEV", "Type","Coverage"], axis = 1, inplace=True)
# let's rename the columns so that they make sense
df.rename(columns = {'OdName':'Country', 'AreaName':'Continent','RegName':'Region'}, inplace= True)
# for sake of consistency, let's also make all column labels of type string
df.columns = list(map(str, df.columns))
# set the country name as index - useful for quickly looking up countries using .loc method
df.set_index("Country", inplace=True)
df.head()
# add total column
df['Total'] = df.sum(axis=1)
# years that we will be using in this lesson - useful for plotting later on
years = list(map(str, range(1980, 2014)))

%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches 
mpl.style.use("ggplot")
print('Matplotlib version: ', mpl.__version__)
# let's create a new dataframe for these three countries 
three_country = df.loc[['Denmark',"Norway","Sweden"],:]

# let's take a look at our dataframe
three_country
# compute the proportion of each category with respect to the total
total_values = sum(three_country["Total"])
category_proportions =[(float(value) / total_values) for value in three_country["Total"]]

for i, proportion in enumerate(category_proportions):
    print(three_country.index.values[i] + ": " +str(proportion) )
width = 40 # width of chart
height =10 #height of chart

total_num_tiles = width*height #total number of tiles
print('Total number of tile is ', total_num_tiles)
# compute the number of tiles for each catagory
tiles_per_category = [round(proportion * total_num_tiles) for proportion in category_proportions]

# print out number of tiles per category
for i, tiles in enumerate(tiles_per_category):
    print(three_country.index.values[i] + ": " + str(tiles))
# initialize the waffle chart as an empty matrix
waffle_chart = np.zeros((height, width))

# define indices to loop through waffle chart
category_index = 0
tile_index = 0

# populate the waffle chart
for col in range(width):
    for row in range(height):
        tile_index += 1

        # if the number of tiles populated for the current category is equal to its corresponding allocated tiles...
        if tile_index > sum(tiles_per_category[0:category_index]):
            # ...proceed to the next category
            category_index += 1       
            
        # set the class value to an integer, which increases with class
        waffle_chart[row, col] = category_index
        
print ('Waffle chart populated!')
#Let's take a peek at how the matrix looks like.
waffle_chart
# instantiate a new figure object 
fig = plt.figure()

# use matshow to display the waffle chart
colormap = plt.cm.coolwarm
plt.matshow(waffle_chart, cmap =colormap)
plt.colorbar()

# instantiate a new figure object
fig = plt.figure()

# use matshow to display the waffle chart
colormap = plt.cm.coolwarm
plt.matshow(waffle_chart, cmap = colormap)
plt.colorbar()

#get the axis
ax = plt.gca()

#set minor ticks
ax.set_xticks(np.arange(-.5, (width),1), minor =True)
ax.set_yticks(np.arange(-.5, (height), 1), minor=True)

#add grilines based on minor ticks
ax.grid(which ="minor", color="w", linestyle = "-", linewidth=2)

plt.xticks([])
plt.yticks([])
# instantiate a new figure object
fig =plt.figure()

# use matshow to display the waffle chart
colormap = plt.cm.coolwarm
plt.matshow(waffle_chart, cmap = colormap)
plt.colorbar

#get the axis
ax = plt.gca()

#set minor ticks
ax.set_xticks(np.arange(-.5, (width), 1), minor = True)
ax.set_yticks(np.arange(-.5, (height), 1), minor =True)

# add gridlines based on minor ticks
ax.grid(which="minor", color="w", linestyle="-", linewidth =2)

plt.xticks([])
plt.yticks([])

# compute cumulative sum of individual categories to match color schemes between chart and legend
values_cumsum  = np.cumsum(three_country["Total"])
total_values = values_cumsum[len(values_cumsum)-1]

#create legend 
legend_handles = []
for i, category in enumerate(three_country.index.values):
    label_str = category + ' (' + str(three_country['Total'][i]) + ')'
    color_val = colormap(float(values_cumsum[i])/total_values)
    legend_handles.append(mpatches.Patch(color=color_val, label=label_str))

# add legend to chart
plt.legend(handles=legend_handles,
           loc='lower center', 
           ncol=len(three_country.index.values),
           bbox_to_anchor=(0., -0.2, 0.95, .1)
          )
def create_waffle_chart(categories, values, height, width, colormap,value_sign=""):
    
    
    # compute the proportion of each category with respect to the total
    total_values = sum(values)
    category_proportions = [(float(value) / total_values) for value in values]
    # compute the total number of tiles
    total_num_tiles = width*height #total number of tiles
    print("Total number of tiles is ", total_num_tiles)
    
    # print out number of tiles per category
    for i, tiles in enumerate(tiles_per_category):
        print(three_country.index.values[i] + ": " + str(tiles))
        
    # initialize the waffle chart as an empty matrix
    waffle_chart =np.zeros((height, width))
    
    # define indices to loop through waffle chart
    category_index = 0
    tile_index = 0
    
    # populate the waffle chart
    for col in range(width):
        for row in range(height):
            tile_index += 1
            
            # if the number of tiles populated for the current category 
            # is equal to its corresponding allocated tiles...
            if tile_index > sum(tiles_per_category[0:category_index]):
                # ...proceed to the next category
                category_index += 1 
                
            # set the class value to an integer, which increases with class
            waffle_chart[row, col] =category_index
            
    # instantiate a new figure object
    fig =plt.figure()
    
    # use matshow to display the waffle chart
    colormap = plt.cm.coolwarm
    plt.matshow(waffle_chart, cmap = colormap)
    plt.colorbar()
    
    #get the axis
    ax = plt.gca()
    
    # set minor ticks
    ax.set_xticks(np.arange(-.5, (width), 1), minor =True)
    ax.set_yticks(np.arange(-.5, (height), 1), minor = True)
    
    # add dridlines based on minor ticks
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)

    plt.xticks([])
    plt.yticks([])

     # compute cumulative sum of individual categories to match color schemes between chart and legend
    values_cumsum = np.cumsum(values)
    total_values = values_cumsum[len(values_cumsum) - 1]

    # create legend
    legend_handles = []
    for i, category in enumerate(categories):
        if value_sign == '%':
            label_str = category + ' (' + str(values[i]) + value_sign + ')'
        else:
            label_str = category + ' (' + value_sign + str(values[i]) + ')'
            
        color_val = colormap(float(values_cumsum[i])/total_values)
        legend_handles.append(mpatches.Patch(color=color_val, label=label_str))

    # add legend to chart
    plt.legend(
        handles=legend_handles,
        loc='lower center', 
        ncol=len(categories),
        bbox_to_anchor=(0., -0.2, 0.95, .1)
    )   
        
width = 40
height = 10

categories = three_country.index.values
values = three_country["Total"]

colormap = plt.cm.coolwarm

create_waffle_chart(categories, values,height,width, colormap)
df.head()
total_immigration = df["Total"].sum()
total_immigration
# install wordcloud
!conda install -c conda-forge wordcloud==1.4.1 --yes

# import package and its set of stopwords
from wordcloud import WordCloud, STOPWORDS

print ('Wordcloud is installed and imported!')
max_words =90
word_string = ""
for country in df.index.values:
    # check if country's name is a single-word name
    if len(country.split(' ')) == 1:
        repeat_num_times = int(df.loc[country,'Total']/float(total_immigration)*max_words)
        word_string = word_string +((country + " ")*repeat_num_times)
        
# display the generated text
word_string
# create the word cloud
wordcloud = WordCloud(background_color = "black").generate(word_string)
print('Word cloud created!')
# display the cloud
fig = plt.figure()
fig.set_figwidth(14)
fig.set_figheight(18)

plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# install seaborn
!conda install -c anaconda seaborn --yes

# import library
import seaborn as sns

print('Seaborn installed and imported!')
# we can use the sum() method to get the total population per year
df_tot = pd.DataFrame(df[years].sum(axis=0))

# change the years to type float (useful for regression later on)
df_tot.index = map(float, df_tot.index)
# reset the index to put in back in as a column in the df_tot dataframe
df_tot.reset_index(inplace = True)

#rename columns
df_tot.columns =['year', 'total']
#view the final dataframe
df_tot.head()
import seaborn as sns
ax = sns.regplot(x = 'year', y= "total", data=df_tot)
ax = sns.regplot(x= "year", y = "total", data = df_tot, color="green")
#Let's blow up the plot a little bit so that it is more appealing to the sight.
plt.figure(figsize=(15, 10))

ax = sns.regplot(x="year", y ="total", data=df_tot, color="green", marker="+")
plt.figure(figsize=(15, 10))
ax = sns.regplot(x='year', y='total', data=df_tot, color='green', marker='+', scatter_kws={'s': 200})

ax.set(xlabel="Year", ylabel ="Total immigration") #add x - and y-labels
ax.set_title("Total Immigration to Canada from 1980-2013") #add title
plt.figure(figsize = (15,10))

sns.set(font_scale=1.5)

ax = sns.regplot(x="year", y="total", data= df_tot, color="green", marker = "+", scatter_kws ={"s":200})
ax.set(xlabel="Year", ylabel= "Total Immigration")
ax.set_title("Total Immigration to Canada from 1980 - 2013")
three_country.head()
three_total = pd.DataFrame(three_country[years].sum(axis=0))
three_total.index = map(float, three_total.index)

three_total.reset_index(inplace=True)
three_total.columns = ['year',"total"]
three_total.head()
plt.figure(figsize=(10,10))
ax= sns.regplot(x="year", y="total", data= three_total, color="orange", marker="+")

sns.set(font_scale=1.5)
sns.set_style('whitegrid')


ax.set(xlabel ='Year', ylabel="Total Immigration")
ax.set_title("Total immigration of three country to Canada ")
!conda install -c conda-forge folium=0.5.0 --yes
import folium

print('Folium installed and imported!')
#define the world map
world_map = folium.Map()

#display world map
world_map

# define the world map centered around Canada with a low zoom level
world_map = folium.Map(location = [56.130, -106.35], zoom_start=4)

#display world map
world_map
# define the world map centered around Canada with a higher zoom level
world_map = folium.Map(location = [56.130, -105.35], zoom_start = 8)

world_map
mexico_latitude = 23.6345 
mexico_longitude = -102.5528

Mexico_map = folium.Map(location=[mexico_latitude, mexico_longitude], zoom_start=4)
Mexico_map
# create a Stamen Toner map of the world centered around Canada
world_map = folium.Map(location= [56.130, -106.35], zoom_start=4, tiles = 'Stamen Toner')
world_map
# create a Stamen Toner map of the world centered around Canada
world_map = folium.Map(location=[56.130, -103.35], zoom_start=4, tiles="Stamen Terrain")
world_map
# create a world map with a Mapbox Bright style.
world_map = folium.Map(tiles= 'Mapbox Bright')

world_map
Mexico_map = folium.Map(location=[mexico_latitude, mexico_longitude], zoom_start=6, tiles= "Stamen Terrain")
Mexico_map
df = pd.read_excel("/kaggle/input/immigration-to-canada-ibm-dataset/Canada.xlsx",
                     sheet_name='Canada by Citizenship',
                     skiprows = range(20),
                     skipfooter = 2)
                     
df.head()
# download countries geojson file
!wget --quiet https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DV0101EN/labs/Data_Files/world_countries.json -O world_countries.json
    
print('GeoJSON file downloaded!')
# clean up the dataset to remove unnecessary columns (eg. REG) 
df.drop(['AREA','REG','DEV','Type','Coverage'], axis=1, inplace=True)

# let's rename the columns so that they make sense
df.rename(columns={'OdName':'Country', 'AreaName':'Continent','RegName':'Region'}, inplace=True)

# for sake of consistency, let's also make all column labels of type string
df.columns = list(map(str, df.columns))

# add total column
df['Total'] = df.sum(axis=1)

# years that we will be using in this lesson - useful for plotting later on
years = list(map(str, range(1980, 2014)))
print ('data dimensions:', df.shape)
df.head()
df.columns
world_geo = r'world_countries.json' # geojson file

# create a plain world map
world_map = folium.Map(location=[0, 0], zoom_start=2, tiles='Mapbox Bright')
# generate choropleth map using the total immigration of each country to Canada from 1980 to 2013
world_map.choropleth(
    geo_data=world_geo,
    data=df,
    columns=['Country', 'Total'],
    key_on='feature.properties.name',
    fill_color='Yl0rRd', 
    fill_opacity=0.7, 
    line_opacity=0.2,
    legend_name='Immigration to Canada'
)
world_map
world_geo = r'world_countries.json'

# create a numpy array of length 6 and has linear spacing from the minium total immigration to the maximum total immigration
threshold_scale = np.linspace(df['Total'].min(),
                              df['Total'].max(),
                              6, dtype=int)
threshold_scale = threshold_scale.tolist() # change the numpy array to a list
threshold_scale[-1] = threshold_scale[-1] + 1 # make sure that the last value of the list is greater than the maximum immigration

# let Folium determine the scale.
world_map = folium.Map(location=[0, 0], zoom_start=2, tiles='Mapbox Bright')
world_map.choropleth(
    geo_data=world_geo,
    data=df,
    columns=['Country', 'Total'],
    key_on='feature.properties.name',
    threshold_scale=threshold_scale,
    fill_color='YlOrRd', 
    fill_opacity=0.7, 
    line_opacity=0.2,
    legend_name='Immigration to Canada',
    reset=True
)
world_map
