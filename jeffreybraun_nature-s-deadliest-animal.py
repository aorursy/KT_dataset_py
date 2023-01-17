import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import seaborn as sns

df_animal = pd.read_csv('/kaggle/input/massachusetts-arbovirus-survalliance-data-201419/animal_cases.csv')
df_animal.dropna(inplace=True)
df_human = pd.read_csv('/kaggle/input/massachusetts-arbovirus-survalliance-data-201419/human_arbovirus_cases.csv')
df_mosquito = pd.read_csv('/kaggle/input/massachusetts-arbovirus-survalliance-data-201419/mosquito_totals.csv')
df_testing = pd.read_csv('/kaggle/input/massachusetts-arbovirus-survalliance-data-201419/total_tests_per_year.csv')
from folium import plugins
from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="jeff_braun")

for i in range(df_human.shape[0]):
    town = df_human.loc[i, 'County']
    loc = town + ' County, Massachusetts, United States of America'
    location = geolocator.geocode(loc)
    df_human.loc[i, 'Latitude'] = location.latitude
    df_human.loc[i, 'Longitude'] = location.longitude
    

for i in range(df_animal.shape[0]):
    town = df_animal.loc[i, 'Town or City']
    loc = town + ', Massachusetts, United States of America'
    location = geolocator.geocode(loc)
    df_animal.loc[i, 'Latitude'] = location.latitude
    df_animal.loc[i, 'Longitude'] = location.longitude

from branca.element import Template, MacroElement

# Thank you to the author of this jupyter notebook for the legend code:
# https://nbviewer.jupyter.org/gist/talbertc-usgs/18f8901fc98f109f2b71156cf3ac81cd

template = """
{% macro html(this, kwargs) %}

<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>jQuery UI Draggable - Default functionality</title>
  <link rel="stylesheet" href="//code.jquery.com/ui/1.12.1/themes/base/jquery-ui.css">

  <script src="https://code.jquery.com/jquery-1.12.4.js"></script>
  <script src="https://code.jquery.com/ui/1.12.1/jquery-ui.js"></script>
  
  <script>
  $( function() {
    $( "#maplegend" ).draggable({
                    start: function (event, ui) {
                        $(this).css({
                            right: "auto",
                            top: "auto",
                            bottom: "auto"
                        });
                    }
                });
});

  </script>
</head>
<body>

 
<div id='maplegend' class='maplegend' 
    style='position: absolute; z-index:9999; border:2px solid grey; background-color:rgba(255, 255, 255, 0.8);
     border-radius:6px; padding: 10px; font-size:14px; right: 20px; bottom: 20px;'>
     
<div class='legend-title'>Legend</div>
<div class='legend-scale'>
  <ul class='legend-labels'>
    <li><span style='background:red;opacity:1;'></span>Human</li>
    <li><span style='background:green;opacity:1;'></span>Animal</li>
    <li><span style='background:blue;opacity:0.2;'></span>Mosquito</li>

  </ul>
</div>
</div>
 
</body>
</html>

<style type='text/css'>
  .maplegend .legend-title {
    text-align: left;
    margin-bottom: 5px;
    font-weight: bold;
    font-size: 90%;
    }
  .maplegend .legend-scale ul {
    margin: 0;
    margin-bottom: 5px;
    padding: 0;
    float: left;
    list-style: none;
    }
  .maplegend .legend-scale ul li {
    font-size: 80%;
    list-style: none;
    margin-left: 0;
    line-height: 18px;
    margin-bottom: 2px;
    }
  .maplegend ul.legend-labels li span {
    display: block;
    float: left;
    height: 16px;
    width: 30px;
    margin-right: 5px;
    margin-left: 0;
    border: 1px solid #999;
    }
  .maplegend .legend-source {
    font-size: 80%;
    color: #777;
    clear: both;
    }
  .maplegend a {
    color: #777;
    }
</style>
{% endmacro %}"""

import folium

mass_map = folium.Map([42.368002, -71.922023], zoom_start=8)

map_title = 'Arbovirus Surveillance (2014-2019)'
title_html = '''
             <h3 align="center" style="font-size:16px"><b>{}</b></h3>
             '''.format(map_title) 
mass_map.get_root().html.add_child(folium.Element(title_html))

human_color = 'red'
animal_color = 'green'

for i in range(df_human.shape[0]):
    folium.CircleMarker([df_human.loc[i, 'Latitude'], df_human.loc[i, 'Longitude']],
                        radius=15,
                        popup='Onset Date: ' + df_human.loc[i, 'Onset Date'],
                        fill_color = human_color,
                        line_color = human_color,
                        fill_opacity = 1,
                        ).add_to(mass_map)
    
for i in range(df_animal.shape[0]):
    folium.CircleMarker([df_animal.loc[i, 'Latitude'], df_animal.loc[i, 'Longitude']],
                        radius=15,
                        popup='Onset Year: ' + str(df_animal.loc[i, 'Onset Year']),
                        fill_color= animal_color,
                        fill_opacity = 1,
                        ).add_to(mass_map)

macro = MacroElement()
macro._template = Template(template)

mass_map.get_root().add_child(macro)

mass_map
from tqdm.notebook import tqdm

town_list = list(df_mosquito.Town.unique())
town_dict = {}

for town in town_list:
    loc = town + ', Massachusetts, United States of America'
    location = geolocator.geocode(loc)
    lat = location.latitude
    long = location.longitude
    town_dict[town] = [lat, long]
    
def get_lat(town):
    return town_dict[town][0]

def get_long(town):
    return town_dict[town][1]

df_mosquito['Latitude'] = df_mosquito.Town.apply(lambda x: get_lat(x))
df_mosquito['Longitude'] = df_mosquito.Town.apply(lambda x: get_long(x))
town_counts = dict(df_mosquito.Town.value_counts())
mosquito_color = 'blue'

for town in town_list:
    loc = town_dict[town]
    num = float(town_counts[town])
    label = str(int(num)) + ' mosquitoes'
    folium.CircleMarker([loc[0], loc[1]],
                        radius = num/4,
                        fill_color = mosquito_color,
                        fill_opacity = 0.2,
                        popup = label,
                        ).add_to(mass_map)

mass_map
template = """
{% macro html(this, kwargs) %}

<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>jQuery UI Draggable - Default functionality</title>
  <link rel="stylesheet" href="//code.jquery.com/ui/1.12.1/themes/base/jquery-ui.css">

  <script src="https://code.jquery.com/jquery-1.12.4.js"></script>
  <script src="https://code.jquery.com/ui/1.12.1/jquery-ui.js"></script>
  
  <script>
  $( function() {
    $( "#maplegend" ).draggable({
                    start: function (event, ui) {
                        $(this).css({
                            right: "auto",
                            top: "auto",
                            bottom: "auto"
                        });
                    }
                });
});

  </script>
</head>
<body>   
<div id='maplegend' class='maplegend' 
    style='position: absolute; z-index:9999; border:2px solid grey; background-color:rgba(255, 255, 255, 0.8);
     border-radius:6px; padding: 10px; font-size:14px; right: 20px; bottom: 20px;'>
     
<div class='legend-title'>Legend</div>
<div class='legend-scale'>
  <ul class='legend-labels'>
    <li><span style='background:blue;opacity:0.2;'></span>Mosquito</li>

  </ul>
</div>
</div>
 
</body>
</html>

<style type='text/css'>
  .maplegend .legend-title {
    text-align: left;
    margin-bottom: 5px;
    font-weight: bold;
    font-size: 90%;
    }
  .maplegend .legend-scale ul {
    margin: 0;
    margin-bottom: 5px;
    padding: 0;
    float: left;
    list-style: none;
    }
  .maplegend .legend-scale ul li {
    font-size: 80%;
    list-style: none;
    margin-left: 0;
    line-height: 18px;
    margin-bottom: 2px;
    }
  .maplegend ul.legend-labels li span {
    display: block;
    float: left;
    height: 16px;
    width: 30px;
    margin-right: 5px;
    margin-left: 0;
    border: 1px solid #999;
    }
  .maplegend .legend-source {
    font-size: 80%;
    color: #777;
    clear: both;
    }
  .maplegend a {
    color: #777;
    }
</style>
{% endmacro %}"""

df_pop = pd.read_csv('/kaggle/input/massachusetts-population-full/mass_population.csv')

df_pop = df_pop[['NAME','POPESTIMATE2016']]
df_pop.dropna(inplace=True)
df_pop.drop_duplicates(inplace=True)
df_pop.sort_values(['POPESTIMATE2016'], ascending=False,inplace=True)
df_pop.reset_index(inplace=True)
df_pop.drop(columns=['index'], inplace=True)
df_pop.drop(index = 0, inplace=True)
df_pop.reset_index(inplace=True)
df_pop = df_pop[~df_pop.NAME.str.contains(" County")]
df_pop.reset_index(inplace=True)
df_pop = df_pop.drop(columns = ['level_0', 'index'])

df_pop.replace('Town city', '', regex=True, inplace=True)

df_pop_plot = df_pop[0:200]

for i in range(df_pop_plot.shape[0]):
    town = df_pop_plot.loc[i, 'NAME']
    loc = town + ', Massachusetts, United States of America'
    location = geolocator.geocode(loc)
    if location != None:
        df_pop_plot.loc[i, 'Latitude'] = location.latitude
        df_pop_plot.loc[i, 'Longitude'] = location.longitude
    else:
        df_pop_plot.loc[i, 'Latitude'] = np.nan
        df_pop_plot.loc[i, 'Longitude'] = np.nan
        
df_pop_plot.dropna(inplace=True)
    
mass_pop_map = folium.Map([42.368002, -71.922023], zoom_start=8)

map_title = 'Population (2016 Estimate) overlayed with Arbovirus Positive Mosquitoes'
title_html = '''
             <h3 align="center" style="font-size:16px"><b>{}</b></h3>
             '''.format(map_title) 
mass_pop_map.get_root().html.add_child(folium.Element(title_html))

df_pop_plot.POPESTIMATE2016 = df_pop_plot.POPESTIMATE2016 * (1/10000)

locationArr = df_pop_plot[['Latitude', 'Longitude', 'POPESTIMATE2016']]

# plot heatmap
mass_pop_map.add_child(plugins.HeatMap(locationArr))

for town in town_list:
    loc = town_dict[town]
    num = float(town_counts[town])
    label = str(int(num)) + ' mosquitoes'
    folium.CircleMarker([loc[0], loc[1]],
                        radius = num/4,
                        fill_color = mosquito_color,
                        fill_opacity = 0.2,
                        popup = label,
                        ).add_to(mass_pop_map)
    
macro = MacroElement()
macro._template = Template(template)

mass_pop_map.get_root().add_child(macro)
    
mass_pop_map

from sklearn import preprocessing
from sklearn.linear_model import LinearRegression

df_pop = pd.read_csv('/kaggle/input/massachusetts-population-full/mass_population.csv')

df_pop = df_pop[['NAME','POPESTIMATE2016']]
df_pop.dropna(inplace=True)
df_pop.drop_duplicates(inplace=True)
df_pop.sort_values(['POPESTIMATE2016'], ascending=False,inplace=True)
df_pop.reset_index(inplace=True)
df_pop.drop(columns=['index'], inplace=True)
df_pop.drop(index = 0, inplace=True)
df_pop.reset_index(inplace=True)
df_pop = df_pop[~df_pop.NAME.str.contains(" County")]
df_pop.reset_index(inplace=True)
df_pop = df_pop.drop(columns = ['level_0', 'index'])

df_pop.replace(' Town city', '', regex=True, inplace=True)
df_pop.replace([" city", " town"], ["", ""], regex=True, inplace=True)
df_town = df_mosquito.Town.value_counts().to_frame('arbovirus_pos_mosquitoes')

for i in df_town.index:
    val = df_pop[df_pop.NAME == i]['POPESTIMATE2016']
    if len(val) != 0:
        df_town.loc[i, 'pop'] = int(val)
    else:
        df_town.loc[i, 'pop'] = np.nan
   
df_town.dropna(inplace=True)
df_town = (df_town - df_town.mean())/df_town.std()

print("Correlation: Pearson Method")
print(df_town.corr())

model = LinearRegression()
model.fit(df_town['pop'].values[:,np.newaxis], df_town['arbovirus_pos_mosquitoes'].values)

plt.figure(figsize=(7,7))
df_town.drop('Boston', inplace=True)
sns.scatterplot(data = df_town, x = 'pop', y = 'arbovirus_pos_mosquitoes')
plt.title("Population vs. Number of Arbovirus Positive Mosquitoes (Scaled)")
plt.show()
    