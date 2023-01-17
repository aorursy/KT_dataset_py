import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt

import plotly.express as px

import plotly.offline as py

import plotly.graph_objs as go

import plotly.io as pio

py.offline.init_notebook_mode(connected=True)

import geopandas as gpd

import os

import folium

import gc

from folium.plugins import TimeSliderChoropleth

from scipy.signal import find_peaks

from sklearn import preprocessing

import time

from datetime import datetime

from scipy import integrate, optimize

import warnings

warnings.filterwarnings("ignore")
PATH_DATA = '/kaggle/input/covid19inspain/agregados.csv'

PATH_GEO_JSON = '/kaggle/input/spain-geojson/shapefiles_ccaa_espana.geojson'

data = pd.read_csv(PATH_DATA,delimiter=",", encoding="latin1", skiprows=range(1730,1739))
print(data.head())

print(data.tail(10))
data.fillna(0)#missing values with 0
#changing name of columns

data.rename(columns={"FECHA":"Date",

              "PCR+":"Infected",

             "Hospitalizados":"Hospitalized",

             "Fallecidos":"Deaths",

             "Recuperados":"Cured",

             "UCI":"ICU",

             "CASOS":"Cases"},inplace= True)
data.replace({"AN":"Andalucía","AR":"Aragón","AS":"Asturias",

                "IB":"Baleares","CN":"Canarias","CB":"Cantabria",

                 "CM":"Castilla La Mancha","CL":"Castilla y León","CT":"Cataluña",

              "CE":"Ceuta","VC":"C. Valenciana","EX":"Extremadura","GA":"Galicia",

             "MD":"Madrid","ML":"Melilla","MC":"Murcia","NC":"Navarra",

             "PV":"País Vasco","RI":"La Rioja"},inplace=True)
data.isnull().sum()
#convert the date values and create a new column called NEW_DATE.

data.Date = pd.to_datetime(data.Date, format="%d/%m/%Y")
data["NEW_DATE"] = data.Date.apply(lambda x: x.strftime("%d %b, %Y"))
total_s = data.groupby(["Date","NEW_DATE"])["Date","Cases","Infected","TestAc+","Deaths","Hospitalized","ICU"].sum().reset_index()

total_s.head()
#create variables with infectef & dead daily people

aux = total_s.Infected.to_list()



daily=[]



for i in range(len(aux)-1):

    b = aux[i+1] - aux[i]

    daily.append(b)

    

daily.insert(0,0)   



total_s["Daily_Infected"] = daily
aux = total_s.Deaths.to_list()



daily=[]



for i in range(len(aux)-1):

    b = aux[i+1] - aux[i]

    daily.append(b)

    

daily.insert(0,0)   



total_s["Daily_Deaths"] = daily
aux = total_s.Cases.to_list()



daily=[]



for i in range(len(aux)-1):

    b = aux[i+1] - aux[i]

    daily.append(b)

    

daily.insert(0,0)   



total_s["Daily_Cases"] = daily

total_s.head()
data_infected = data[data.Date>"20-02-2020"]
fig = px.bar(data_infected, x="CCAA", y="Infected", color="CCAA",

              animation_frame="NEW_DATE", animation_group="CCAA", range_y=[0,data.Infected.max()+1000],title= "Infections by regions over time")

fig.show()
fig = px.area(total_s, x= "Date", y = "Daily_Deaths", title= "Daily deaths in Spain", color_discrete_sequence = ['red'])

fig.show()
total_madrid = data[data.CCAA=="Madrid"].groupby("Date")["Date","Infected","Deaths","Hospitalized","ICU"].sum().reset_index()
aux_m = total_madrid.melt(id_vars="Date", value_vars=("Infected","Deaths","ICU","Hospitalized"), value_name="Count" , var_name= "Status")
fig = px.bar(aux_m, x= "Date", y = "Count", color="Status", title= "Actual situation in Madrid")

fig.show()
fig = go.Figure(data=[

    go.Bar(name='Infections', x=total_s['Date'], y=total_s['Daily_Infected']),

    go.Bar(name='Deaths', x=total_s['Date'], y=total_s['Daily_Deaths'])

])

# Change the bar mode

fig.update_layout(barmode='overlay', title='Daily Case and Death count(Spain)',

                 annotations=[dict(x='2020-03-15', y=1407, xref="x", yref="y", text="Lockdown Imposed(15th March)", showarrow=True, arrowhead=1, ax=-100, ay=-200)])

fig.show()
for i in data.CCAA.unique(): 

    

    a = i.replace(".","")

    a = a.replace(" ","_")

    

    exec('df_{}=data[data.CCAA == i]'.format(a))

    

    exec('aux_a = df_{}.Infected.to_list()'.format(a))

    

    

    daily=[]

    for i in range(len(aux_a)-1):

        b = aux_a[i+1] - aux_a[i]

        daily.append(b)

    

    daily.insert(0,0)   



    exec('df_{}["Daily_infected"] = daily'.format(a))

    

    exec('aux_d = df_{}.Deaths.to_list()'.format(a))

    

    

    daily=[]

    for i in range(len(aux_d)-1):

        b = aux_d[i+1] - aux_d[i]

        daily.append(b)

    

    daily.insert(0,0)   



    exec('df_{}["Daily_deaths"] = daily'.format(a))

df_daily_infected = pd.DataFrame({"Date":data.Date.unique(),

                                 "Madrid":df_Madrid["Daily_infected"].values,

                                 "Cataluña":df_Cataluña["Daily_infected"].values,

                                 "Andalucia":df_Andalucía["Daily_infected"].values,

                                 "Castilla La Mancha":df_Castilla_La_Mancha["Daily_infected"].values,

                                 "Castilla y Leon":df_Castilla_y_León["Daily_infected"].values,

                                 "País Vasco":df_País_Vasco["Daily_infected"].values})
aux_i = df_daily_infected.melt(id_vars="Date", value_vars=("Madrid","Cataluña","Andalucia","Castilla La Mancha","Castilla y Leon","País Vasco"), value_name="Count" , var_name= "CCAA")
aux_1=aux_i[aux_i.Date>"18-04-2020"]
fig = px.bar (aux_1, x= "Date", y = "Count", color="CCAA", title= "Daily infections in Spain (Top 6)")

fig.show()
df_daily_fatalities = pd.DataFrame({"Date":data.Date.unique(),

                                 "Madrid":df_Madrid["Daily_deaths"].values,

                                 "Cataluña":df_Cataluña["Daily_deaths"].values,

                                 "Valencia":df_C_Valenciana["Daily_deaths"].values,

                                 "Castilla La Mancha":df_Castilla_La_Mancha["Daily_deaths"].values,

                                 "Castilla y Leon":df_Castilla_y_León["Daily_deaths"].values,

                                 "País Vasco":df_País_Vasco["Daily_deaths"].values})
aux_f = df_daily_fatalities.melt(id_vars="Date", value_vars=("Madrid","Cataluña","Valencia","Castilla La Mancha","Castilla y Leon","País Vasco"), value_name="Count" , var_name= "CCAA")
fig = px.line (aux_f, x= "Date", y = "Count", color="CCAA", title= "Daily Death in Spain (Top 6)")

fig.show()
fig = px.bar (aux_f, x= "Date", y = "Count", color="CCAA", title= "Daily Deaths in Spain (Top 6)")

fig.show()
datewise_spain=data.groupby(["Date"]).agg({"Infected":'sum',"Deaths":'sum'})
spain_increase_confirm=[]

spain_increase_deaths=[]

for i in range(datewise_spain.shape[0]-1):

    spain_increase_confirm.append(((datewise_spain["Infected"].iloc[i+1])/datewise_spain["Infected"].iloc[i]))

    spain_increase_deaths.append(((datewise_spain["Deaths"].iloc[i+1])/datewise_spain["Deaths"].iloc[i]))

spain_increase_confirm.insert(0,1)

spain_increase_deaths.insert(0,1)

datewise_spain["WeekOfYear"]=datewise_spain.index.weekofyear



week_num_spain=[]

spain_weekwise_confirmed=[]

spain_weekwise_recovered=[]

spain_weekwise_deaths=[]

w=1

for i in list(datewise_spain["WeekOfYear"].unique()):

    spain_weekwise_confirmed.append(datewise_spain[datewise_spain["WeekOfYear"]==i]["Infected"].iloc[-1])

    spain_weekwise_deaths.append(datewise_spain[datewise_spain["WeekOfYear"]==i]["Deaths"].iloc[-1])

    week_num_spain.append(w)

    w=w+1

    

fig=go.Figure()

fig.add_trace(go.Scatter(x=week_num_spain, y=spain_weekwise_confirmed,

                    mode='lines+markers',

                    name='Weekly Growth of Confirmed Cases'))

fig.add_trace(go.Scatter(x=week_num_spain, y=spain_weekwise_deaths,

                    mode='lines+markers',

                    name='Weekly Growth of Death Cases'))

fig.update_layout(title="Weekly Growth of different types of Cases in Spain",

                 xaxis_title="Week Number",yaxis_title="Number of Cases",legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
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
df = pd.read_csv(PATH_DATA, delimiter=",", encoding="latin1", skiprows=range(1730,1739))
df.rename(columns = {"FECHA":"DATE",

                    "CASOS":"CASES",

                     "PCR+":"TOTAL_INFECTED",

                    "Hospitalizados":"REQUIERED_HOSPITALIZATION",

                    "UCI":"REQUIERED_ADVANCED_CARE",

                    "Fallecidos":"TOTAL_DEATHS"}, inplace = True)



df.fillna(0, inplace = True)

df["CCAA"] = df["CCAA"].map(d_name)

df["CCAA_for_Folium"] = df["CCAA"].map(d_ccaa)

df["id"] = df["CCAA_for_Folium"].map(d_ccaa_id)



df["Population"] = df["CCAA_for_Folium"].map(d_ccaa_population)

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
df["NEW_DATE"] = df["DATE"].apply(correct_date)
df["DATE"] = pd.to_datetime(df["NEW_DATE"], format='%d/%m/%Y')



df["DATE_for_Folium"] = (df["DATE"].astype(int)// 10**9).astype('U10')



df = df[["id", "CCAA", "CCAA_for_Folium", "DATE", "DATE_for_Folium", "TOTAL_INFECTED", "REQUIERED_HOSPITALIZATION", "REQUIERED_ADVANCED_CARE", "TOTAL_DEATHS","Population"]]

df["id"].astype(np.int16)

df.head()
gdf = gpd.read_file(PATH_GEO_JSON)

gdf["id"] = gdf["name_1"].map(d_ccaa_id) # create a numerical id for each ccaa

gdf = gdf[["id", "shape_leng","shape_area","geometry"]] # extract the id and the geometry (coordinates of each ccaa)

gdf["geometry"] = gdf["geometry"].simplify(0.1, preserve_topology = False)

gdf["id"].astype(int)

gdf.head()


m = folium.Map(location = (40, 0), zoom_start = 5.5)



folium.Choropleth(

    geo_data = gdf,

    name = 'choropleth',

    data = df[df["DATE"] == max(df["DATE"])],

    columns = ['id', 'TOTAL_INFECTED'],

    key_on='feature.properties.id',

    fill_color='RdPu',

    fill_opacity=0.7,

    line_opacity=0.2,

    legend_name = 'Total infected cases in Spain by region'

).add_to(m)



m


m = folium.Map(location = (40, 0), zoom_start = 5.5)



folium.Choropleth(

    geo_data = gdf,

    name = 'choropleth',

    data = df,

    columns = ['id', 'TOTAL_DEATHS'],

    key_on='feature.properties.id',

    fill_color='RdPu',

    fill_opacity=0.7,

    line_opacity=0.2,

    legend_name = 'Total deaths in Spain by region'

).add_to(m)



m
#----------------------------------------------------------------------------------

data_to_color = "TOTAL_INFECTED"

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

m = folium.Map(location=(40, 0), tiles='OpenStreetMap', zoom_start=5.5)



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



df["Infected_1000h"] = df["TOTAL_INFECTED"]/(df["Population"]/1000)

df["Mortality_rate"] = df["TOTAL_DEATHS"] / df["TOTAL_INFECTED"]

df.fillna(0, inplace = True)

df.head()
plt.figure(figsize = (20, 10))



for ccaa in sorted(list(df["CCAA"].unique())):

    

    x = df["DATE"].unique()

    y = df[df["CCAA"] == ccaa]["Mortality_rate"]

    

    plt.plot(x, y, label = ccaa)

    plt.title("Evolution of Mortality rate over time")

    plt.legend()

    plt.xticks(rotation=90)


x = [day for day in range(len(df["DATE"].unique()))]



fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(30,20))

# plt.setp(axes, ylim=(0 ,max(df["Mortality rate"])))

ccaas = list(df["CCAA"].unique())



i = 0



for col_axes in axes:

    for ax in col_axes:

        if i < len(ccaas):

            ccaa = ccaas[i]

            y = df[df["CCAA"] == ccaa]["Mortality_rate"].values

            

            ipeaks, _ = find_peaks(y)

            

            ax.plot(x, y, color = "k", alpha = 0.7)

            ax.scatter(ipeaks, np.array(y)[ipeaks], color = "red", label = "Local peaks of mortality")

            ax.scatter(x[list(y).index(np.max(y))], np.max(y), color = "k", marker = "o", alpha = 0.7, s = 250, label = "Max of mortality")

            

            ax.set_title("Mortality rate {}".format(ccaa))

            ax.legend()

            ax.grid()

            i += 1



fig.delaxes(axes[3, 4])
total_df = df.set_index("DATE").resample("D")[["TOTAL_INFECTED", "REQUIERED_HOSPITALIZATION", "REQUIERED_ADVANCED_CARE", "TOTAL_DEATHS","Population"]].sum()

total_df = total_df[total_df["Population"] > 0]

total_df["TOTAL_INFECTED_1000H"] = total_df["TOTAL_INFECTED"]/(total_df["Population"]/1000)

total_df["TOTAL_DEATHS_1000H"] = total_df["TOTAL_DEATHS"]/(total_df["Population"]/1000)



# get the data

x = list(total_df.index)

y_1 = list(total_df["TOTAL_INFECTED_1000H"]) # 1 axis

y_2 = list(total_df["TOTAL_DEATHS_1000H"]) # 2 axis



# create the figures

fig, ax = plt.subplots(figsize = (15, 7))

plot1 = ax.plot(x, y_1, color = "r", label = "Total infected per 1000 habitants") # plot the first data

plt.xticks(rotation=90) # rotate the date



ax2 = ax.twinx() # create a secondary axis

plot2 = ax2.plot(x, y_2, color = "k", label = "Total deaths per 1000 habitants") # plot the second data

fig.tight_layout()

plt.title("Evolution of total infected cases and total deaths per 1000 habitants")



# create a common legend

lns = plot1 + plot2

labs = [l.get_label() for l in lns]

ax.legend(lns, labs, loc=0)



# prettify

ax.grid()

ax.set_xlabel("Date")

ax.set_ylabel("Ratio of total infected per 1000 habitants")

ax2.set_ylabel("Ratio of total deaths per 1000 habitants")
total_df["SHIFT_7_DAYS"] = total_df["TOTAL_INFECTED"].shift(-7)

total_df["SHIFT_14_DAYS"] = total_df["TOTAL_INFECTED"].shift(-14)
x = np.array([x for x in range(len(total_df.index))])

y_informed = total_df["TOTAL_INFECTED"]

y_real_7_days = total_df["SHIFT_7_DAYS"]

y_real_14_days = total_df["SHIFT_14_DAYS"]

width = np.min(np.diff(x))/3



fig = plt.figure(figsize = (20, 10))



ax = fig.add_subplot(111)

ax.bar(x - width, y_informed, width, color = 'b', label = 'Known cases', alpha = 0.5)

ax.bar(x, y_real_7_days, width, color = 'r', label = '"Real Cases" shift 1 week', alpha = 0.4)

ax.bar(x + width, y_real_14_days, width, color='k', label = '"Real Cases" shift 2 week', alpha = 0.3)

ax.set_xlabel('Days since first infected case.')



plt.title("Known infected cases vs 'Real Cases' with a 1 and 2 week shift")

plt.axvline(x=17, lw = 1, alpha = 0.3, ymax = 0.4, color = "purple")

plt.annotate("8 March manifestation held", xy= (15, 80000), color = "purple")



textstr = '\n'.join((

    r'Known cases vs "Real Cases" 1 week shift: {:,.0f}'.format(total_df.iloc[17]["SHIFT_7_DAYS"] - total_df.iloc[17]["TOTAL_INFECTED"]),

    r'Known cases vs "Real Cases" 2 week shift: {:,.0f}'.format(total_df.iloc[17]["SHIFT_14_DAYS"] - total_df.iloc[17]["TOTAL_INFECTED"])))



props = dict(boxstyle='round', facecolor='purple', alpha=0.5)



# place a text box in upper left in axes coords

ax.text(0.05, 0.6, textstr, transform=ax.transAxes, fontsize=14,

        verticalalignment='top', bbox=props)



plt.legend()
short_df = df[df["DATE"] == max(df["DATE"])][["CCAA", "Mortality_rate"]].sort_values("Mortality_rate", ascending = False)

x = short_df["CCAA"]

y = short_df["Mortality_rate"]



mean_y = np.mean(y)

mean_y



plt.figure(figsize = (10, 5))

plt.scatter(x, y, c= "red", alpha = 0.5)

plt.title("Mortality rate by region")



plt.xticks(rotation=90)

plt.axhline(mean_y, c = "k", alpha = 0.5, lw = 1)

plt.annotate('Mean mortality is {}%'.format(round(mean_y * 100, 2)),

             xy=(12, mean_y),

             xycoords='data',

             xytext=(50, 50), 

             textcoords='offset points',

             arrowprops=dict(arrowstyle="->", color = "k", alpha = 0.5),

             color = "k")
infected = pd.read_csv('/kaggle/input/covid19inspain/ccaa_covid19_casos_long.csv')

uci_beds = pd.read_csv('/kaggle/input/covid19-in-spain/ccaa_camas_uci_2017.csv')

recovered = pd.read_csv('/kaggle/input/covid19-in-spain/ccaa_covid19_altas_long.csv')

death = pd.read_csv('/kaggle/input/covid19-in-spain/ccaa_covid19_fallecidos_long.csv')

hospitalized = pd.read_csv('/kaggle/input/covid19-in-spain/ccaa_covid19_hospitalizados_long.csv')

masks = pd.read_csv('/kaggle/input/covid19-in-spain/ccaa_covid19_mascarillas.csv')

icu = pd.read_csv('/kaggle/input/covid19-in-spain/ccaa_covid19_uci_long.csv')

national = pd.read_csv('/kaggle/input/covid19-in-spain/nacional_covid19.csv')

age_range = pd.read_csv('/kaggle/input/covid19-in-spain/nacional_covid19_rango_edad.csv')
max_date = infected['fecha'].max()
def dateplot(x, y, **kwargs):

    ax = plt.gca()

    data = kwargs.pop("data")

    data.plot(x=x, y=y, ax=ax, grid=False, **kwargs)
infected['fecha'] = pd.to_datetime(infected['fecha'])

hospitalized['fecha'] = pd.to_datetime(hospitalized['fecha'])

icu['fecha'] = pd.to_datetime(icu['fecha'])

recovered['fecha'] = pd.to_datetime(recovered['fecha'])

death['fecha'] = pd.to_datetime(death['fecha'])
infected = infected[infected['CCAA']!= 'Total']

g = sns.FacetGrid(infected, col="CCAA", col_wrap=5, height=3.5)

g = g.map_dataframe(dateplot, "fecha", "total").set(yscale='log')

g = g.map(plt.fill_between, 'fecha', 'total', alpha=0.2).set_titles("{col_name} CCAA")

g = g.set_titles("{col_name}")

plt.subplots_adjust(top=0.92)

g = g.fig.suptitle('Evolution of total infected in CCAA (log scale)')
icu= icu[icu['CCAA']!= 'Total']

g = sns.FacetGrid(icu, col="CCAA", col_wrap=5, height=3.5)

g = g.map_dataframe(dateplot, "fecha", "total").set(yscale='log')

g = g.map(plt.fill_between, 'fecha', 'total', alpha=0.2).set_titles("{col_name} CCAA")

g = g.set_titles("{col_name}")

plt.subplots_adjust(top=0.92)

g = g.fig.suptitle('Evolution of total ICU patients in CCAA (log scale)')
hospitalized = hospitalized[hospitalized['CCAA']!= 'Total']

g = sns.FacetGrid(hospitalized, col="CCAA", col_wrap=5, height=3.5)

g = g.map_dataframe(dateplot, "fecha", "total").set(yscale='log')

g = g.map(plt.fill_between, 'fecha', 'total', alpha=0.2).set_titles("{col_name} CCAA")

g = g.set_titles("{col_name}")

plt.subplots_adjust(top=0.92)

g = g.fig.suptitle('Evolution of total hospitalized in CCAA (Log Scale) ')
recovered = recovered[recovered['CCAA']!= 'Total']

g = sns.FacetGrid(recovered, col="CCAA", col_wrap=5, height=3.5)

g = g.map_dataframe(dateplot, "fecha", "total").set(yscale='log')

g = g.map(plt.fill_between, 'fecha', 'total', alpha=0.2).set_titles("{col_name} CCAA")

g = g.set_titles("{col_name}")

plt.subplots_adjust(top=0.92)

g = g.fig.suptitle('Evolution of total recovered in CCAA (Log Scale)')
death = death[death['CCAA']!= 'Total']

g = sns.FacetGrid(death, col="CCAA", col_wrap=5, height=3.5)

g = g.map_dataframe(dateplot, "fecha", "total").set(yscale='log')

g = g.map(plt.fill_between, 'fecha', 'total', alpha=0.2).set_titles("{col_name} CCAA")

g = g.set_titles("{col_name}")

plt.subplots_adjust(top=0.92)

g = g.fig.suptitle('Evolution of total deaths in CCAA (log scale)')
infected_last = infected[infected['fecha']== max_date]

recovered_last = recovered[recovered['fecha']== max_date]

hospitalized_last = hospitalized[hospitalized['fecha']== max_date]

death_last = death[death['fecha']== max_date]

uci_last = icu[icu['fecha']== max_date]
df_an = pd.DataFrame(data ={'Infected': infected_last['total'].values,

                            'Hospitalized':hospitalized_last['total'].values,

                            'ICU':uci_last['total'].values,

                            'Recovered': recovered_last['total'].values,

                            'Death':death_last['total'].values},

                             index = infected_last['CCAA'])
df_total = df_an[df_an.index=='Total'] 

df_an= df_an[df_an.index!='Total']
d = pd.to_datetime(str(max_date)).strftime('%Y-%m-%d')

title = 'COVID-2019'

chart_title = title + ' as of ' + d

ccaa = df_an.index.to_list()

print('Number of CCAA with confirmed cases = ',len(ccaa))



# Looks lot have hit a limit of Sunburst chart

max_ccaa = df_an.index.unique()

ids = ccaa

labels = ccaa

parents = [title] * len(ccaa)

values = df_an['Infected'].to_list()



classifications = df_an.columns.drop('Infected').values



for cty in ccaa: 

    for c in classifications:

        ids = ids + [cty + '-' + c]

        parents = parents + [cty]

        labels = labels + [c]

        values = values + [df_an.loc[cty][c]]



trace = go.Sunburst(

    ids=ids,

    labels=labels,

    parents=parents,

    values=values,

    outsidetextfont={"size": 20, "color": "#377eb8"},

#     leaf={"opacity": 0.4},

    marker={"line": {"width": 2}}

)



layout = go.Layout(

    title = chart_title + "<br>(click on CCAA to view details)",

    margin = go.layout.Margin(t=100, l=0, r=0, b=0),

    sunburstcolorway=["#636efa","#ef553b","#00cc96"]

)



fig = go.Figure([trace], layout)



py.iplot(fig)
age_range= age_range[age_range['rango_edad']!='Total']

age_range= age_range[age_range['rango_edad']!='80 y +']

no_gender = age_range[age_range['sexo']=='ambos']
g = sns.catplot(x="rango_edad", y="casos_confirmados", hue="sexo", data=no_gender, kind="bar", height=5,aspect=3,palette="muted")

g.despine(left=True)

g.set_ylabels("Total infected")
last = age_range[age_range.iloc[:,0]== age_range.iloc[:,0].max()]
for i in range(last['ingresos_uci'].shape[0]):

    if last.iloc[i,5] == 'i':

        last.iloc[i,5] = 0

        

last['ingresos_uci']= last['ingresos_uci'].astype(int)
last['casos_confirmados'] = last['casos_confirmados'] / np.linalg.norm(last['casos_confirmados'])

last['hospitalizados'] = last['hospitalizados'] / np.linalg.norm(last['hospitalizados'])

last['ingresos_uci'] = last['ingresos_uci'] / np.linalg.norm(last['ingresos_uci'])

last['fallecidos'] = last['fallecidos'] / np.linalg.norm(last['fallecidos'])

last_ambos = last[last['sexo']=='ambos']

last_gender = last[last['sexo']!='ambos']

plt.figure(figsize=(15,5))

plt.plot(last_ambos['rango_edad'], last_ambos['casos_confirmados'],color = 'green',label='Total infected')

plt.plot(last_ambos['rango_edad'], last_ambos['hospitalizados'],color = 'red',label='Hospitalized')

plt.plot( last_ambos['rango_edad'], last_ambos['ingresos_uci'],color = 'yellow',label='UCI')

plt.plot( last_ambos['rango_edad'], last_ambos['fallecidos'],color = 'black',label='Death')

plt.title('COVID-19 vs age groups')

plt.legend()
plt.figure(figsize= (10,5))

sns.relplot(x='rango_edad',y ='casos_confirmados', hue = 'sexo',kind='line',data = last_gender,height=5,aspect=4)

plt.title('Comparison between men and women: Total infections')
plt.figure(figsize= (10,5))

sns.relplot(x='rango_edad',y ='hospitalizados', hue = 'sexo',kind='line',data = last_gender,height=5,aspect=4)

plt.title('Comparison between men and women: Hospitalized')
plt.figure(figsize= (10,5))

sns.relplot(x='rango_edad',y ='ingresos_uci', hue = 'sexo',kind='line',data = last_gender,height=5,aspect=4)

plt.title('Comparison between men and women: ICU')
plt.figure(figsize= (10,5))

sns.relplot(x='rango_edad',y ='fallecidos', hue = 'sexo',kind='line',data = last_gender,height=5,aspect=4)

plt.title('Comparison between men and women: Death')
es_covid = pd.merge(infected,death,how='outer',left_on=['fecha','cod_ine','CCAA'],right_on=['fecha','cod_ine','CCAA'],suffixes=('_confirmed','_deaths')).merge(icu,how='outer', left_on=['fecha','cod_ine','CCAA'], right_on=['fecha','cod_ine','CCAA']).merge(hospitalized, how='outer', left_on=['fecha','cod_ine','CCAA'], right_on=['fecha','cod_ine','CCAA'],suffixes=('_uci', '_hosp')).merge(recovered, how='outer', left_on=['fecha','cod_ine','CCAA'], right_on=['fecha','cod_ine','CCAA'])

es_covid["fecha"] = pd.to_datetime(es_covid['fecha'])
es_covid = es_covid.sort_values(by=['CCAA', 'fecha'])

es_covid['diff_total_confirmed'] = es_covid.groupby(['CCAA'])['total_confirmed'].diff().fillna(es_covid['total_confirmed'])

es_covid['diff_total_deaths'] = es_covid.groupby(['CCAA'])['total_deaths'].diff().fillna(es_covid['total_deaths'])

es_covid['diff_total_recovered'] = es_covid.groupby(['CCAA'])['total'].diff().fillna(es_covid['total'])



es_covid['diff_total_confirmed'].fillna(0, inplace=True)

es_covid['diff_total_deaths'].fillna(0, inplace=True)

es_covid['diff_total_recovered'].fillna(0, inplace=True)



es_covid['day_num'] = preprocessing.LabelEncoder().fit_transform(es_covid.fecha)



display(es_covid.loc[es_covid['fecha'] > '2020-05-20'])
missings_count = {col:es_covid[col].isnull().sum() for col in es_covid.columns}

print(pd.DataFrame.from_dict(missings_count, orient='index').nlargest(30, 0))

del missings_count
es_dic_pob = {'CCAA': ['Andalucía','Aragón','Asturias','Baleares','Canarias','Cantabria','Castilla y León','Castilla La Mancha','Cataluña','C. Valenciana','Extremadura','Galicia','Madrid','Murcia','Navarra','País Vasco','La Rioja','Ceuta','Melilla'],

          'hombres': [4147167,650694,488137,572757,1065971,281801,1181401,1016954,3770123,2465342,5285,1298964,3187312,747615,323631,1073074,156179,42912,43894],

          'mujeres': [4267073,668597,534663,576703,1087418,299277,1218147,1015909,3905094,2538427,53921,1400535,3476082,746283,330583,1134702,160619,41865,42593]

               }       

es_poblacion = pd.DataFrame(es_dic_pob, columns = ['CCAA','hombres', 'mujeres'])

es_poblacion.reset_index().set_index('CCAA')

es_poblacion['total'] = es_poblacion['hombres'] + es_poblacion['mujeres']



del es_dic_pob

es_poblacion
# Susceptible equation

def susceptibility(N, s, i, beta):

    si = -beta*s*i

    return si



# Infected equation

def infection(N, s, i, beta, gamma):

    inf = beta*s*i - gamma*i

    return inf



# Recovered/deceased equation

def recovery(N, i, gamma):

    rec = gamma*i

    return rec
# Runge-Kutta method of 4rth order for 3 dimensions (susceptible s, infected i snd recovered r)

def rK4(N, s, i, r, susceptibility, infection, recovery, beta, gamma, hs):

    s1 = susceptibility(N, s, i, beta)*hs

    i1 = infection(N, s, i, beta, gamma)*hs

    r1 = recovery(N, i, gamma)*hs

    sk = s + s1*0.5

    ik = i + i1*0.5

    rk = r + r1*0.5

    s2 = susceptibility(N, sk, ik, beta)*hs

    i2 = infection(N, sk, ik, beta, gamma)*hs

    r2 = recovery(N, ik, gamma)*hs

    sk = s + s2*0.5

    ik = i + i2*0.5

    rk = r + r2*0.5

    s3 = susceptibility(N, sk, ik, beta)*hs

    i3 = infection(N, sk, ik, beta, gamma)*hs

    r3 = recovery(N, ik, gamma)*hs

    sk = s + s3

    ik = i + i3

    rk = r + r3

    s4 = susceptibility(N, sk, ik, beta)*hs

    i4 = infection(N, sk, ik, beta, gamma)*hs

    r4 = recovery(N, ik, gamma)*hs

    s = s + (s1 + 2*(s2 + s3) + s4)/6

    i = i + (i1 + 2*(i2 + i3) + i4)/6

    r = r + (r1 + 2*(r2 + r3) + r4)/6

    return s, i, r
def SIR(N, b0, beta, gamma, hs):



    # Initial condition

    s = float(N-1)/N -b0

    i = float(1)/N +b0

    r = 0.



    sus, inf, rec= [],[],[]

    for j in range(10000): # Run for a certain number of time-steps

        sus.append(s)

        inf.append(i)

        rec.append(r)

        s,i,r = rK4(N, s, i, r, susceptibility, infection, recovery, beta, gamma, hs)



    return sus, inf, rec

N = es_poblacion['total'].sum()

b0 = 0

beta = 0.7

gamma = 0.2

hs = 0.1



sus, inf, rec = SIR(N, b0, beta, gamma, hs)

f = plt.figure(figsize=(8,5)) 

plt.plot(sus, 'b.', label='susceptible');

plt.plot(inf, 'r.', label='infected');

plt.plot(rec, 'c.', label='recovered/deceased');

plt.title('SIR Model')

plt.xlabel("time", fontsize=10);

plt.ylabel("Fraction of population", fontsize=10);

plt.legend(loc='best')

plt.xlim(0,1000)

plt.show()



del N, b0, beta, gamma, hs, sus, inf, rec, f
def sir_model(y, x, beta, gamma):

    sus = -beta * y[0] * y[1] / N

    rec = gamma * y[1] 

    inf = -(sus + rec)

    return sus, inf, rec
def estimateParametersSIR(ccaa, initialDay):

    country_df = pd.DataFrame()

    country_df['ConfirmedCases'] = es_covid.loc[es_covid['CCAA']==ccaa].total_confirmed.diff().fillna(0)

    # This cut it's caused by try visual fits over results

    country_df =  country_df[initialDay:]

    country_df['day_count'] = list(range(1,len(country_df)+1))



    ydata = [i for i in country_df.ConfirmedCases]

    xdata = country_df.day_count

    ydata = np.array(ydata, dtype=float)

    xdata = np.array(xdata, dtype=float)



    N = es_poblacion.loc[es_poblacion['CCAA']==ccaa].total

    inf0 = ydata[0]

    sus0 = N - inf0

    rec0 = 0.0



    def sir_model(y, x, beta, gamma):

        sus = -beta * y[0] * y[1] / N

        rec = gamma * y[1]

        inf = -(sus + rec)

        return sus, inf, rec



    def fit_odeint(x, beta, gamma):

        return integrate.odeint(sir_model, (sus0, inf0, rec0), x, args=(beta, gamma))[:,1]



    popt, pcov = optimize.curve_fit(fit_odeint, xdata, ydata)

    fitted = fit_odeint(xdata, *popt)



    plt.plot(xdata, ydata, 'o')

    plt.plot(xdata, fitted)

    plt.title("Fit of SIR model for " +ccaa + " infected cases")

    plt.ylabel("Population infected")

    plt.xlabel("Days")

    plt.show()

    print("Optimal parameters: \nbeta =", popt[0], " \ngamma = ", popt[1])

    es_poblacion.at[es_poblacion['CCAA'] == ccaa,'ini_day'] = initialDay

    es_poblacion.at[es_poblacion['CCAA'] == ccaa,'beta'] = popt[0]

    es_poblacion.at[es_poblacion['CCAA'] == ccaa,'gamma'] = popt[1]
estimateParametersSIR('Andalucía', 16)
estimateParametersSIR('Aragón', 18)
estimateParametersSIR('Asturias', 15)
estimateParametersSIR('Baleares', 17)
estimateParametersSIR('C. Valenciana', 14)
estimateParametersSIR('Canarias', 16)
estimateParametersSIR('Castilla La Mancha', 9)
estimateParametersSIR('Castilla y León', 9)
estimateParametersSIR('Cataluña', 5)
estimateParametersSIR('Ceuta', 23)
estimateParametersSIR('Extremadura', 9)
estimateParametersSIR('Galicia', 18)

estimateParametersSIR('La Rioja', 12)
estimateParametersSIR('Madrid', 10)
estimateParametersSIR('Melilla', 22)
estimateParametersSIR('Murcia', 17)
estimateParametersSIR('Navarra', 12)
estimateParametersSIR('País Vasco', 9)