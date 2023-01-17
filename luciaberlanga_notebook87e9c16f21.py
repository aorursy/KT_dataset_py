# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
diccionario = pd.read_csv('../input/diccionario/diccionario.csv')
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

import geopandas
import pandas as pd
import numpy as np
import folium
import gc
from folium.plugins import TimeSliderChoropleth
from scipy.signal import find_peaks

PATH_DATA = '../input/prueba/serie_historica_acumulados.csv'
PATH_GEO_JSON = '../input/spain-geojson/shapefiles_ccaa_espana.geojson'

d_name = {
'AN':'Andalucía',
'AR':'Aragón',
'AS':'Asturias',
'IB':'Baleares',
'CN':'Canarias',
'CB':'Cantabria',
'CM':'Castilla La Mancha',
'CL':'Castilla y León',
'CT':'Cataluña',
'CE':'Ceuta',
'VC':'C. Valenciana',
'EX':'Extremadura',
'GA':'Galicia',
'MD':'Madrid',
'ML':'Melilla',
'MC':'Murcia',
'NC':'Navarra',
'PV':'País Vasco',
'RI':'La Rioja'
}

d_ccaa = {
'Andalucía': 'Andalucía',
'Aragón': 'Aragón',
'Asturias': 'Principado de Asturias',
'Baleares': 'Islas Baleares',
'Canarias': 'Islas Canarias',
'Cantabria':'Cantabria',
'Castilla La Mancha': 'Castilla-La Mancha',
'Castilla y León': 'Castilla y León',
'Cataluña': 'Cataluña',
'Ceuta': 'Ceuta y Melilla',
'C. Valenciana': 'Comunidad Valenciana',
'Extremadura': 'Extremadura',
'Galicia': 'Galicia',
'Madrid': 'Comunidad de Madrid',
'Melilla': 'Ceuta y Melilla',
'Murcia': 'Región de Murcia',
'Navarra': 'Comunidad Foral de Navarra',
'País Vasco': 'País Vasco',
'La Rioja': 'La Rioja'
}

d_ccaa_id = {
'Andalucía': "1",
'Aragón' : "2",
'Principado de Asturias': "3",
'Islas Baleares': "4",
'Islas Canarias': "5",
'Cantabria': "6",
'Castilla-La Mancha': "7",
'Castilla y León': "8",
'Cataluña': "9",
'Ceuta y Melilla': "10",
'Comunidad Valenciana': "11",
'Extremadura': "12",
'Galicia': "13",
'Comunidad de Madrid' : "14",
'Ceuta y Melilla': "15",
'Región de Murcia': "16",
'Comunidad Foral de Navarra': "17",
'País Vasco': "18",
'La Rioja': "19"
}

d_ccaa_population = {
'Andalucía': 8414240,
'Aragón' : 1319291,
'Principado de Asturias': 1022800,
'Islas Baleares': 1149460,
'Islas Canarias': 2153389,
'Cantabria': 581078,
'Castilla-La Mancha': 2032863,
'Castilla y León': 2399548,
'Cataluña': 7675217,
'Ceuta y Melilla': 171264,
'Comunidad Valenciana': 5003769,
'Extremadura': 1067710,
'Galicia': 2699499,
'Comunidad de Madrid' : 6663394,
'Ceuta y Melilla': 171264,
'Región de Murcia': 1493898,
'Comunidad Foral de Navarra': 654214,
'País Vasco': 2207776,
'La Rioja': 316798
}

def get_hex_colors(df, data_to_color, cmap = matplotlib.cm.Reds, log = False):
    
    '''
    This function takes the following arguments
        1. df:pandas DataFrame with the data.
        2. data_to_color: the column name with data based on which we want to create the color scale.
        3. cmap: colors you want to plot. You can use this to communicate different messages. For example: greens --> good, greys --> deaths.
                default is matplotlib.cm.Reds
                more about colormaps: https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html
        3. log: if data has huge outliers, we can create the color map with a logarithic normalization. This way, the outliers won't "pale" our other data.
                default is False.
        
    '''
    
    cmap = cmap # define the color pallete you want. You can use Reds, Blues, Greens etc
    my_values = df[data_to_color] # get the value you wan to convert to colors
    
    mini = min(my_values) # get the min to normalize
    maxi= max(my_values) # get the max to normalize
    
    LOGMIN = 0.01 # arbitrary lower bound for log scale
    
    if log: 
        norm = matplotlib.colors.LogNorm(vmin=max(mini,LOGMIN), vmax=maxi) # normalize log data
    else:
        norm = matplotlib.colors.Normalize(vmin=mini, vmax=maxi) # create a color range
        
    colors = {value:matplotlib.colors.rgb2hex(cmap(norm(value))[:3]) for value in sorted(list(set(my_values)))} # create a dictionary with the total_infected or deaths as keys and colors as values
    
    return colors

def get_hex_colors_2(value, cats):
    '''
    Color paletter used from this website:
    
    https://colorbrewer2.org/#type=sequential&scheme=Reds&n=9
    
    The color selection will be based on the percentile each value is in.
    '''
    if value == 0:
        return "#FFFFFF"
    elif value in cats[0]:
        return "#fff5f0"
    elif value in cats[1]:
        return "#fee0d2"
    elif value in cats[2]:
        return "#fcbba1"
    elif value in cats[3]:
        return "#fc9272"
    elif value in cats[4]:
        return "#fb6a4a"
    elif value in cats[5]:
        return "#ef3b2c"
    elif value in cats[6]:
        return "#cb181d"
    elif value in cats[7]:
        return "#a50f15"
    elif value in cats[8]:
        return "#67000d"
    else:
        return "#000000"

def correct_date(date_str):
    list_dates = date_str.split("/")
    day = list_dates[0]
    month = list_dates[1]
    year = list_dates[2]
    
    if len(day) == 1:
        day = "0" + day
    if len(month) == 1:
        month = "0" + month
        
    return "/".join([day, month, year])
df = pd.read_csv(PATH_DATA, encoding='latin-1')
df.fillna(0, inplace = True)
df["CCAA"] = df["CCAA"].map(d_name)
df["CCAA_for_Folium"] = df["CCAA"].map(d_ccaa)
df["id"] = df["CCAA_for_Folium"].map(d_ccaa_id)

df["Population"] = df["CCAA_for_Folium"].map(d_ccaa_population)


df.rename(columns = {"FECHA":"DATE",
                    "CASOS":"TOTAL_INFECTED",
                    "Hospitalizados":"REQUIERED_HOSPITALIZATION",
                    "UCI":"REQUIERED_ADVANCED_CARE",
                    "Fallecidos":"TOTAL_DEATHS",
                    "Recuperados":"CURED"}, inplace = True)
df = df.loc[df.DATE != 0, :]
df["CCAA"].isnull().sum()

df["NEW_DATE"] = df["DATE"].apply(lambda x: correct_date(str(x)))
df["DATE"] = pd.to_datetime(df["NEW_DATE"], format='%d/%m/%Y')

df["DATE_for_Folium"] = (df["DATE"].astype(int)// 10**9).astype('U10')

df = df[["id", "CCAA", "CCAA_for_Folium", "DATE", "DATE_for_Folium", "TOTAL_INFECTED", "REQUIERED_HOSPITALIZATION", "REQUIERED_ADVANCED_CARE", "TOTAL_DEATHS", "CURED","Population"]]
df["id"].astype(np.int16)
df.head()

gdf = geopandas.read_file(PATH_GEO_JSON)
gdf["id"] = gdf["name_1"].map(d_ccaa_id) # create a numerical id for each ccaa
gdf = gdf[["id", "shape_leng","shape_area","geometry"]] # extract the id and the geometry (coordinates of each ccaa)
gdf["geometry"] = gdf["geometry"].simplify(0.1, preserve_topology = False)
gdf["id"].astype(int)
gdf.head()
df = pd.merge(df, diccionario, how='left', left_on='CCAA', right_on='ccaa')
m = folium.Map(location = (40, 0), zoom_start = 5.5)

folium.Choropleth(
    geo_data = gdf,
    name = 'choropleth',
    data = df.loc[df['DATE']=='2/4/2020', :],
    columns = ['id', 'cases_per_cienmil'],
    key_on='feature.properties.id',
    fill_color='RdPu',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name = 'Total infected cases in Spain by region'
).add_to(m)

m
# pintar poblacion 
m = folium.Map(location = (40, 0), zoom_start = 5.5)

folium.Choropleth(
    geo_data = gdf,
    name = 'choropleth',
    data = df,
    columns = ['id', 'Population'],
    key_on='feature.properties.id',
    fill_color='RdPu',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name = 'Total population in Spain by region'
).add_to(m)

m
#----------------------------------------------------------------------------------
data_to_color = "TOTAL_DEATHS"
cats, bins =  pd.qcut(df[data_to_color].unique()[np.argsort(df[data_to_color].unique())], q = 9, retbins = True)
cats = cats.unique()

#----------------------------------------------------------------------------------

# value we will iterate in order to create the styledict
ccaas = list(df["id"].unique())
dates = list(df["DATE_for_Folium"].unique())

# create the color dict and color column
df["COLORS"] = df[data_to_color].apply(get_hex_colors_2, args = [cats]) # we create a colum in the df so that we can iterate and create the styledict

# creates the styledict for the map
styledict = {}

# iterate the populate the styledict
for ccaa in ccaas:
    styledict[str(ccaa)] = {date: {'color': df[(df["id"] == ccaa) & (df["DATE_for_Folium"] == date)]["COLORS"].values[0],
                                   'opacity': 0.6} for date in dates}
    
# creates and renders the Folium map
m = folium.Map(location=(40, 0), tiles='OpenStreetMap', zoom_start=6)

g = TimeSliderChoropleth(
    gdf.set_index("id").to_json(), # get's the coordinates for each id 
    styledict = styledict # styledict contains for each id the timestamp and the color to plot.
)

m.add_child(g)

#--------------------------------------------------------------------------------------
# Let's create a legend for folium
# https://nbviewer.jupyter.org/gist/talbertc-usgs/18f8901fc98f109f2b71156cf3ac81cd

from branca.element import Template, MacroElement

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
    <li><span style='background:#FFFFFF;opacity:0.6;'></span>No cases</li>
    <li><span style='background:#fff5f0;opacity:0.6;'></span>1 Quantile</li>
    <li><span style='background:#fee0d2;opacity:0.6;'></span>2 Quantile</li>
    <li><span style='background:#fcbba1;opacity:0.6;'></span>3 Quantile</li>
    <li><span style='background:#fc9272;opacity:0.6;'></span>4 Quantile</li>
    <li><span style='background:#fb6a4a;opacity:0.6;'></span>5 Quantile</li>
    <li><span style='background:#ef3b2c;opacity:0.6;'></span>6 Quantile</li>
    <li><span style='background:#cb181d;opacity:0.6;'></span>7 Quantile</li>
    <li><span style='background:#a50f15;opacity:0.6;'></span>8 Quantile</li>
    <li><span style='background:#67000d;opacity:0.6;'></span>9 Quantile</li>
    <li><span style='background:#000000;opacity:0.6;'></span>Other</li>
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

macro = MacroElement()
macro._template = Template(template)

m.get_root().add_child(macro)