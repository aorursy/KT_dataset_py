# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import geopandas as gp

import seaborn as sns
import matplotlib as mpl
import matplotlib.pylab as plt
plt.style.use('ggplot')

import folium
from folium import plugins
from io import StringIO

from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
import plotly.plotly as py
from plotly import tools
import plotly.figure_factory as ff
import plotly.tools as tls

import numpy

init_notebook_mode(connected=True)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
print(os.listdir("../input/cpe-data"))


# Any results you write to the current directory are saved as output.
print(os.listdir("../input/cpe-data/Dept_11-00091"))
print(os.listdir("../input/cpe-data/Dept_23-00089"))
print(os.listdir("../input/cpe-data/Dept_35-00103"))
print(os.listdir("../input/cpe-data/Dept_37-00049"))
print(os.listdir("../input/cpe-data/Dept_37-00027"))
print(os.listdir("../input/cpe-data/Dept_49-00009"))
print("Get the list of files in department Dept_11-00091")
print("--------------------------------------------------------------")
print(os.listdir("../input/cpe-data/Dept_11-00091/11-00091_ACS_data"))
print("--------------------------------------------------------------")
print("")
print("--------------------------------------------------------------")
print(os.listdir("../input/cpe-data/Dept_11-00091/11-00091_ACS_data/11-00091_ACS_education-attainment-over-25"))
print("--------------------------------------------------------------")
print("")
print("--------------------------------------------------------------")
print(os.listdir("../input/cpe-data/Dept_11-00091/11-00091_ACS_data/11-00091_ACS_poverty"))
print("--------------------------------------------------------------")
print("")
print("--------------------------------------------------------------")
print(os.listdir("../input/cpe-data/Dept_11-00091/11-00091_ACS_data/11-00091_ACS_education-attainment"))
print("--------------------------------------------------------------")

df_shape_d11 = gp.read_file("../input/cpe-data/Dept_11-00091/11-00091_Shapefiles")
print("Rows & columns of ACS_variable_descriptions ", df_shape_d11.shape)
print("")
print("--------------------------------------------------------------")
print(os.listdir("../input/cpe-data/Dept_11-00091/11-00091_ACS_data/11-00091_ACS_owner-occupied-housing"))
print("--------------------------------------------------------------")
print("")
print("")
df_edu_o25 = pd.read_csv("../input/cpe-data/Dept_11-00091/11-00091_ACS_data/11-00091_ACS_education-attainment-over-25/ACS_16_5YR_B15003_with_ann.csv")
print("Rows & columns of ACS_variable_descriptions ", df_edu_o25.shape)
df_edu_o25.head(3)
print("")
print("--------------------------------------------------------------")
print(os.listdir("../input/cpe-data/Dept_11-00091/11-00091_ACS_data/11-00091_ACS_owner-occupied-housing"))
print("--------------------------------------------------------------")
print("")
df_own = pd.read_csv("../input/cpe-data/Dept_11-00091/11-00091_ACS_data/11-00091_ACS_owner-occupied-housing/ACS_16_5YR_S2502_with_ann.csv")
print("Rows & columns of ACS_variable_descriptions ", df_own.shape)
df_own.head(3)
print("")
print("--------------------------------------------------------------")
print(os.listdir("../input/cpe-data/Dept_11-00091/11-00091_ACS_data/11-00091_ACS_race-sex-age"))
print("--------------------------------------------------------------")
print("")
df_race = pd.read_csv("../input/cpe-data/Dept_11-00091/11-00091_ACS_data/11-00091_ACS_race-sex-age/ACS_15_5YR_DP05_with_ann.csv")
print("Rows & columns of ACS_variable_descriptions ", df_race.shape)
df_race.head(3)
print("")
print("--------------------------------------------------------------")
print(os.listdir("../input/cpe-data/Dept_11-00091/11-00091_ACS_data/11-00091_ACS_poverty"))
print("--------------------------------------------------------------")
print("")
df_pov = pd.read_csv("../input/cpe-data/Dept_11-00091/11-00091_ACS_data/11-00091_ACS_poverty/ACS_16_5YR_S1701_with_ann.csv")
print("Rows & columns of ACS_variable_descriptions ", df_pov.shape)
df_pov.head(3)
print("")
print("--------------------------------------------------------------")
print(os.listdir("../input/cpe-data/Dept_11-00091/11-00091_ACS_data/11-00091_ACS_education-attainment"))
print("--------------------------------------------------------------")
print("")
df_edu = pd.read_csv("../input/cpe-data/Dept_11-00091/11-00091_ACS_data/11-00091_ACS_education-attainment/ACS_16_5YR_S1501_with_ann.csv")
print("Rows & columns of ACS_variable_descriptions ", df_edu.shape)
df_edu.head(3)
print(os.listdir("../input/cpe-data/Dept_11-00091/11-00091_ACS_data/11-00091_ACS_race-sex-age/"))
basepath = "../input/cpe-data/Dept_11-00091/11-00091_ACS_data/11-00091_ACS_race-sex-age/"
df_11 = pd.read_csv(basepath + "ACS_15_5YR_DP05_with_ann.csv")
df_11.head(3)
a_df = pd.read_csv(basepath + "ACS_15_5YR_DP05_metadata.csv")
a_df.head()
single_age_df = df_11[["HC01_VC26", "HC01_VC27", "HC01_VC28", "HC01_VC29"]][1:]
ops = [1, 0.85, 0.75, 0.65, 0.55, 0.45]
traces = []
for i, col in enumerate(single_age_df.columns):
    nm = a_df[a_df["GEO.id"] == col]["Id"].iloc(0)[0].replace("Estimate; SEX AND AGE - ", "")
    trace = go.Bar(x=df_11["GEO.display-label"][1:], y=single_age_df[col], name=nm, marker=dict(opacity=0.6))
    traces.append(trace)
layout = dict(barmode="stack", title="Breakdown by age ", margin=dict(b=100), height=600, legend=dict(x=-0.1, y=1, orientation="h"))
fig = go.Figure(data=traces, layout=layout)
iplot(fig)
single_race_df = df_11[["HC01_VC49", "HC01_VC50", "HC01_VC51", "HC01_VC56", "HC01_VC64", "HC01_VC69"]][1:]
ops = [1, 0.85, 0.75, 0.65, 0.55, 0.45]
traces = []
for i, col in enumerate(single_race_df.columns):
    nm = a_df[a_df["GEO.id"] == col]["Id"].iloc(0)[0].replace("Estimate; RACE - One race - ", "")
    trace = go.Bar(x=df_11["GEO.display-label"][1:], y=single_race_df[col], name=nm, marker=dict(opacity=0.6))
    traces.append(trace)
layout = dict(barmode="stack", title="Breakdown by Population of Race ", margin=dict(b=100), height=600, legend=dict(x=-0.1, y=1, orientation="h"))
fig = go.Figure(data=traces, layout=layout)
iplot(fig)
df_samp = pd.read_csv("../input/cpe-data/ACS_variable_descriptions.csv")
print("Rows & columns of ACS_variable_descriptions ", df_samp.shape)
df_samp.head(10).drop(0)
distBos = "../input/cpe-data/Dept_11-00091/11-00091_Shapefiles/boston_police_districts_f55.shp"
One = gp.read_file(distBos)
One.head()
print(os.listdir("../input/cpe-data/Dept_11-00091/11-00091_Shapefiles"))
dist_dept11="""../input/cpe-data/Dept_11-00091/11-00091_Shapefiles/boston_police_districts_f55.shp"""
convertOne = gp.read_file(dist_dept11)
mapAll = folium.Map([42.3, -71],zoom_start=10, height=400, tiles='OpenStreetMap',API_key='wrobstory.map-12345678')
folium.GeoJson(convertOne).add_to(mapAll)
mapAll
print(os.listdir("../input/cpe-data/Dept_23-00089"))
print("================================================================")
print(os.listdir("../input/cpe-data/Dept_23-00089/23-00089_ACS_data"))
print("================================================================")
print(os.listdir("../input/cpe-data/Dept_23-00089/23-00089_Shapefiles"))
print(os.listdir("../input/cpe-data/Dept_23-00089/23-00089_Shapefiles"))
dist_dept23_shp="""../input/cpe-data/Dept_23-00089/23-00089_Shapefiles/Indianapolis_Police_Zones.shp"""
convert23Shp = gp.read_file(dist_dept23_shp)
map23 = folium.Map([39.8, -86.1],zoom_start=10, height=400, tiles='OpenStreetMap',API_key='wrobstory.map-12345678')
folium.GeoJson(convert23Shp).add_to(map23)
map23
distBos = "../input/cpe-data/Dept_23-00089/23-00089_Shapefiles/Indianapolis_Police_Zones.shp"
One = gp.read_file(distBos)
One.head()

f, ax = plt.subplots(1, figsize=(20, 15))
One.plot(column="POLICEZONE", ax=ax, cmap='Accent',legend=True);
plt.title("Police zone - Indianapolis")
plt.show()
print(os.listdir("../input/cpe-data/Dept_35-00103"))
print("================================================================")
print(os.listdir("../input/cpe-data/Dept_35-00103/35-00103_ACS_data"))
print("================================================================")
print(os.listdir("../input/cpe-data/Dept_35-00103/35-00103_Shapefiles"))
df_prepped_d35 = pd.read_csv("../input/cpe-data/Dept_35-00103/35-00103_UOF-OIS-P_prepped.csv")
print("Rows & columns of ACS_variable_descriptions ", df_prepped_d35.shape)
df_prepped_d35.head(4)
map35 = folium.Map([35.22, -80.89], height=400, zoom_start=10, tiles='OpenStreetMap')
for j, rown in df_prepped_d35[1:].iterrows():
    if str(rown["LOCATION_LONGITUDE"]) != "nan":
        lon = float(rown["LOCATION_LATITUDE"])
        lat = float(rown["LOCATION_LONGITUDE"])
        folium.Circle([lon, lat], radius=250, color='crimson', fill=True).add_to(map35)
map35
map35 = folium.Map([35.22, -80.89], height=400, zoom_start=10, tiles='OpenStreetMap')
for j, rown in df_prepped_d35[1:].iterrows():
    if str(rown["LOCATION_LONGITUDE"]) != "nan":
        lon = float(rown["LOCATION_LATITUDE"])
        lat = float(rown["LOCATION_LONGITUDE"])
        
        if rown["SUBJECT_INJURY_TYPE"] == "Non-Fatal Injury":
            col = "blue"
        elif rown["SUBJECT_INJURY_TYPE"]== "Fatal Injury":
            col = "red"
        else:
            col = "green"
            
        folium.Circle([lon, lat], radius=400, color=col, fill=True).add_to(map35)
map35
map35 = folium.Map([35.22, -80.89], height=400, zoom_start=10, tiles='OpenStreetMap')
for j, rown in df_prepped_d35[1:].iterrows():
    if str(rown["LOCATION_LONGITUDE"]) != "nan":
        lon = float(rown["LOCATION_LATITUDE"])
        lat = float(rown["LOCATION_LONGITUDE"])
        
        if rown["SUBJECT_GENDER"] == "Male":
            col = "blue"
        else:
            col = "green"
            
        folium.Circle([lon, lat], radius=400, color=col, fill=True).add_to(map35)
map35
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
    <li><span style='background:black;opacity:0.7;'></span>Black</li>
    <li><span style='background:blue;opacity:0.7;'></span>White</li>
    <li><span style='background:yellow;opacity:0.7;'></span>Hispanic</li>
    <li><span style='background:red;opacity:0.7;'></span>Others</li>

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


map35 = folium.Map([35.22, -80.89], height=600, zoom_start=10, tiles='OpenStreetMap')
for j, rown in df_prepped_d35[1:].iterrows():
    if str(rown["LOCATION_LONGITUDE"]) != "nan":
        lon = float(rown["LOCATION_LATITUDE"])
        lat = float(rown["LOCATION_LONGITUDE"])
        
        if rown["SUBJECT_RACE"] == "Black":
            col = "black"
        elif rown["SUBJECT_RACE"]== "White":
            col = "blue"
        elif rown["SUBJECT_RACE"]== "Hispanic":
            col = "yellow"
        else:
            col = "red"
            
        folium.Circle([lon, lat], radius=250, color=col, fill=True).add_to(map35)
map35.get_root().add_child(macro)
map35
map35 = folium.Map([35.22, -80.89], height=400, zoom_start=10, tiles='OpenStreetMap')
for j, rown in df_prepped_d35[1:].iterrows():
    if str(rown["LOCATION_LONGITUDE"]) != "nan":
        lon = float(rown["LOCATION_LATITUDE"])
        lat = float(rown["LOCATION_LONGITUDE"])
        
        if str(rown["SUBJECT_AGE_IN_YEARS"]) >= "60":
            col = "green"
        elif str(rown["SUBJECT_AGE_IN_YEARS"]) >= "40" and str(rown["SUBJECT_AGE_IN_YEARS"]) < "60":
            col = "blue"
        elif str(rown["SUBJECT_AGE_IN_YEARS"]) >= "20" and str(rown["SUBJECT_AGE_IN_YEARS"]) < "40":
            col = "red"
        else:
            col = "black"
            
        folium.Circle([lon, lat], radius=250, color=col, fill=True).add_to(map35)
map35
dept35= "../input/cpe-data/Dept_35-00103/35-00103_UOF-OIS-P_prepped.csv"
datap5= pd.read_csv(dept35)[1:]
datap5["INCIDENT_DATE"] = pd.to_datetime(datap5["INCIDENT_DATE"]).astype(str)
datap5["MonthDate"] = datap5["INCIDENT_DATE"].apply(lambda x : x.split("-")[0] +'-'+ x.split("-")[1] + "-01")
tmp = datap5.groupby("MonthDate").agg({"SUBJECT_INJURY_TYPE" : "count"}).reset_index()

tmp

trace1 = go.Scatter(x=tmp["MonthDate"], y=tmp.SUBJECT_INJURY_TYPE, name="Month wise Incidents")

data = [trace1]
layout = go.Layout(height=400, title="Incidents in CMPD")
fig = go.Figure(data, layout)
iplot(fig)
print(os.listdir("../input/cpe-data/Dept_35-00103/35-00103_Shapefiles"))
dist_dept35_shp="""../input/cpe-data/Dept_35-00103/35-00103_Shapefiles/CMPD_Police_Division_Offices.shp"""
convert35Shp = gp.read_file(dist_dept35_shp)
map35 = folium.Map([35.15637558, -80.75600938],zoom_start=10, height=400, tiles='OpenStreetMap',API_key='wrobstory.map-12345678')
folium.GeoJson(convert35Shp).add_to(map35)
map35
print(os.listdir("../input/cpe-data/Dept_37-00049/37-00049_Shapefiles"))
dist_dept35 = """../input/cpe-data/Dept_37-00049/37-00049_Shapefiles/EPIC.shp"""
convert35_1Shp = gp.read_file(dist_dept35)
map351 = folium.Map([32.7, -96.7],zoom_start=10, height=400, tiles='OpenStreetMap',API_key='wrobstory.map-12345678')
folium.GeoJson(convert35_1Shp).add_to(map351)
map351
distBos = "../input/cpe-data/Dept_37-00049/37-00049_Shapefiles/EPIC.shp"
One = gp.read_file(distBos)
One.head()

f, ax = plt.subplots(1, figsize=(20, 15))
One.plot(column="Name", ax=ax, cmap='Accent',legend=True);
plt.title("Name")
plt.show()