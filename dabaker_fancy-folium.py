import folium

import branca

import pandas as pd

print(folium.__file__)

print(folium.__version__)
df_2009_2011 = pd.read_csv('../input/2000-16-traffic-flow-england-scotland-wales/accidents_2009_to_2011.csv',

                           usecols=['Longitude','Latitude','Number_of_Vehicles',

                           'Number_of_Casualties','LSOA_of_Accident_Location',

                           'Day_of_Week','Light_Conditions','Weather_Conditions',

                           'Road_Surface_Conditions','Year','Date','Time'])

df_2009_2011.info()
df = df_2009_2011[(df_2009_2011['Year']==2010) & (df_2009_2011['LSOA_of_Accident_Location']=='E01004736')]

print(len(df))

df.head()
#location is the mean of every lat and long point to centre the map.

location = df['Latitude'].mean(), df['Longitude'].mean()



#A basemap is then created using the location to centre on and the zoom level to start.

m = folium.Map(location=location,zoom_start=15)



#Each location in the DataFrame is then added as a marker to the basemap points are then added to the map

for i in range(0,len(df)):

    folium.Marker([df['Latitude'].iloc[i],df['Longitude'].iloc[i]]).add_to(m)

        

m
location = df['Latitude'].mean(), df['Longitude'].mean()

m = folium.Map(location=location,zoom_start=15)



for i in range(0,len(df)):

       

    popup = folium.Popup('Accident', parse_html=True) 

    folium.Marker([df['Latitude'].iloc[i],df['Longitude'].iloc[i]],popup=popup).add_to(m)

m
#There are a number of accidents with multiple casualties

df['Number_of_Casualties'].value_counts()
location = df['Latitude'].mean(), df['Longitude'].mean()

m = folium.Map(location=location,zoom_start=15)



#The num of casulaties for each accident can be determined and the colour assigned then added to the basemap.

for i in range(0,len(df)):

    num_of_casualties = df['Number_of_Casualties'].iloc[i]

    if num_of_casualties == 1:

        color = 'blue'

    elif num_of_casualties == 2:

        color = 'green'

    else:

        color = 'red'

    

    popup = folium.Popup('Accident', parse_html=True) 

    folium.Marker([df['Latitude'].iloc[i],df['Longitude'].iloc[i]],popup=popup,icon=folium.Icon(color=color, icon='info-sign')).add_to(m)



m
def fancy_html(row):

    i = row

    

    Number_of_Vehicles = df['Number_of_Vehicles'].iloc[i]                             

    Number_of_Casualties = df['Number_of_Casualties'].iloc[i]                           

    Date = df['Date'].iloc[i]

    Time = df['Time'].iloc[i]                                           

    Light_Conditions = df['Light_Conditions'].iloc[i]                               

    Weather_Conditions = df['Weather_Conditions'].iloc[i]                             

    Road_Surface_Conditions = df['Road_Surface_Conditions'].iloc[i]

    

    left_col_colour = "#2A799C"

    right_col_colour = "#C5DCE7"

    

    html = """<!DOCTYPE html>

<html>



<head>

<h4 style="margin-bottom:0"; width="300px">{}</h4>""".format(Date) + """



</head>

    <table style="height: 126px; width: 300px;">

<tbody>

<tr>

<td style="background-color: """+ left_col_colour +""";"><span style="color: #ffffff;">Number of Vehicles</span></td>

<td style="width: 200px;background-color: """+ right_col_colour +""";">{}</td>""".format(Number_of_Vehicles) + """

</tr>

<tr>

<td style="background-color: """+ left_col_colour +""";"><span style="color: #ffffff;">Casualties</span></td>

<td style="width: 200px;background-color: """+ right_col_colour +""";">{}</td>""".format(Number_of_Casualties) + """

</tr>

<tr>

<td style="background-color: """+ left_col_colour +""";"><span style="color: #ffffff;">Time</span></td>

<td style="width: 200px;background-color: """+ right_col_colour +""";">{}</td>""".format(Time) + """

</tr>

<tr>

<td style="background-color: """+ left_col_colour +""";"><span style="color: #ffffff;">Light Conditions</span></td>

<td style="width: 200px;background-color: """+ right_col_colour +""";">{}</td>""".format(Light_Conditions) + """

</tr>

<tr>

<td style="background-color: """+ left_col_colour +""";"><span style="color: #ffffff;">Weather Conditions</span></td>

<td style="width: 200px;background-color: """+ right_col_colour +""";">{}</td>""".format(Weather_Conditions) + """

</tr>

<tr>

<td style="background-color: """+ left_col_colour +""";"><span style="color: #ffffff;">Road Conditions</span></td>

<td style="width: 200px;background-color: """+ right_col_colour +""";">{}</td>""".format(Road_Surface_Conditions) + """

</tr>

</tbody>

</table>

</html>

"""

    return html
location = df['Latitude'].mean(), df['Longitude'].mean()

m = folium.Map(location=location,zoom_start=15,min_zoom=5)



for i in range(0,len(df)):

    html = fancy_html(i)

 

    iframe = branca.element.IFrame(html=html,width=400,height=300)

    popup = folium.Popup(iframe,parse_html=True)

    

    folium.Marker([df['Latitude'].iloc[i],df['Longitude'].iloc[i]],

                  popup=popup,icon=folium.Icon(color=color, icon='info-sign')).add_to(m)



m
data_heat = df[['Latitude','Longitude','Number_of_Casualties']].values.tolist()
data_heat[0]
import folium.plugins as plugins



m = folium.Map(location=location, zoom_start=15)

#tiles='stamentoner'



plugins.HeatMap(data_heat).add_to(m)



m
df_Areas = pd.read_csv('../input/ukregions/Output_Area_to_Local_Authority_District_to_Lower_Layer_Super_Output_Area_to_Middle_Layer_Super_Output_Area_to_Local_Enterprise_Partnership_April_2017_Lookup_in_England_V2.csv',

                       usecols=['LAD16CD','LSOA11CD','LAD16NM'])

df_Areas.head()

df_Areas = pd.DataFrame(df_Areas[df_Areas['LAD16NM']=='Westminster']['LSOA11CD'])
df_West = pd.DataFrame(df_2009_2011[df_2009_2011['Year']==2010])

df_West.rename(columns={'LSOA_of_Accident_Location':'LSOA11CD'},inplace=True)

df_West = pd.merge(df_West,df_Areas,on='LSOA11CD')

print(len(df_West))

df_West.head()
location = location = df_West['Latitude'].mean(), df_West['Longitude'].mean()

data = df_West[['Latitude','Longitude','Number_of_Casualties']].values.tolist()
m = folium.Map(location=location, zoom_start=13)



plugins.HeatMap(data).add_to(m)



m
df_West['Date'] = pd.to_datetime(df_West['Date'])

df_West['Month'] = df_West['Date'].apply(lambda time: time.month)
data = [df_West[df_West['Month']==df_West['Month'].unique()[i]][['Latitude','Longitude']].values.tolist() 

        for i in range(len(df_West['Month'].unique()))]
monthDict = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 

            7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}



index = [monthDict[i] for i in sorted(df_West['Month'].unique())]
m = folium.Map(location=location,zoom_start=12)

hm = plugins.HeatMapWithTime(data=data,index=index)



hm.add_to(m)



m