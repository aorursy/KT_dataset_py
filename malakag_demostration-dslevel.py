#Let's import libraries we need

import pandas as pd

import geopandas as gpd

import folium
# importing geo json files 

dsg=gpd.read_file('/kaggle/input/srilanka-geo/Divisional_geo.json')
# loading first two rows of the data set

dsg.head(2)
#removing other data except "ADM3_PCODE" and "geometry"

dsg=dsg[['ADM3_PCODE','geometry']]
# loading aditional data

ds_a=pd.read_csv('/kaggle/input/srilanka-geo/ds_aditional_data.csv')

ds_a.rename(columns={'admin3Pcode':'ADM3_PCODE'}, inplace=True)

ds_a.head()
divisional=dsg.merge(ds_a,on="ADM3_PCODE")

divisional.head()
# Defining the style function and Highlight function 



style_function = lambda x: {'fillColor': '#ffffff', 

                            'color':'#000000', 

                            'fillOpacity': 0.1, 

                            'weight': 0.1}

highlight_function = lambda x: {'fillColor': '#000000', 

                                'color':'#000000', 

                                'fillOpacity': 0.50, 

                                'weight': 0.1}







# declaring the map with starting point

m = folium.Map(location=[7.8731, 80.7718], zoom_start=8)



choropleth=folium.Choropleth(

    geo_data=divisional,

    name='choropleth',

    data= divisional,

    columns=['ADM3_PCODE', 'risk_level'],

    key_on='properties.ADM3_PCODE',

    fill_color='RdPu',

    fill_opacity=0.7,

    line_opacity=0.2,

    

    legend_name='Covid-19 risk Level with dummy data',



).add_to(m)



# adding some geo loacation (location pins )



folium.Marker(

    location=[6.905655, 79.927357],

    popup='Point one',

    

    icon=folium.Icon(icon='cloud')

).add_to(m)



folium.Marker(

    location=[6.7106,79.9074],

    popup='Point two',

    icon=folium.Icon(color='red', icon='info-sign')

).add_to(m)



folium.Marker(

    location=[7.9403,81.01886823],

    popup='Point three',

    icon=folium.Icon(color='blue', icon='bar-chart')

).add_to(m)



folium.Marker(

    location=[9.3803, 80.3770],

    popup='Point four',

    icon=folium.Icon(color='red', icon='university')

).add_to(m)



#adding the same geo layer again on top of previouse layer to make interactivity 



INT = folium.features.GeoJson(

    divisional,

    style_function=style_function, 

    control=False,

    highlight_function=highlight_function, 

    tooltip=folium.features.GeoJsonTooltip(

        fields=['admin3Name_en','patients','population'],

        aliases=['Divisionl sec: ','Numer of Patients:','Population'],

        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;") 

    )

)

m.add_child(INT)

m.keep_in_front(INT)

folium.LayerControl().add_to(m)

#m

#m.save('divisionl_sec.html')


