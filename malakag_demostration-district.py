#Let's import libraries we need

import pandas as pd

import geopandas as gpd

import folium
# importing geo json files 





disg = gpd.read_file('/kaggle/input/srilanka-geo/District_geo.json')

# loading first two rows of the data set

disg.head(2)
#removing other data except "ADM2_PCODE" and "geometry"

disg=disg[['ADM2_PCODE','geometry']]



# loading aditional data that we want to load to final map other than  properties of Geojson file

dis_a=pd.read_csv('/kaggle/input/srilanka-geo/dristrict_aditional_data.csv')

dis_a.rename(columns={'admin2Pcode':'ADM2_PCODE'}, inplace=True)

dis_a.head()
#merging both geo data and adttioanl data into one data fame for generate map mapping key is '"ADM2_PCODE""



district=disg.merge(dis_a,on="ADM2_PCODE")

district.head()
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

m = folium.Map(location=[7.8731, 80.7718], zoom_start=7)



choropleth=folium.Choropleth(

    geo_data=district,

    name='choropleth',

    data= district,

    columns=['ADM2_PCODE', 'patients'],

    key_on='properties.ADM2_PCODE',

    fill_color='PuRd',

    fill_opacity=0.7,

    line_opacity=0.2,

    

    legend_name='Covid-19 risk Level',



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

    icon=folium.Icon(color='blue', icon='info-sign')

).add_to(m)



folium.Marker(

    location=[9.3803, 80.3770],

    popup='Point four',

    icon=folium.Icon(color='red', icon='info-sign')

).add_to(m)



#adding the same geo layer again on top of previouse layer to make interactivity 



INT = folium.features.GeoJson(

    district,

    style_function=style_function, 

    control=False,

    highlight_function=highlight_function, 

    tooltip=folium.features.GeoJsonTooltip(

        fields=['admin2Name_en','patients','population'],

        aliases=['District: ','Numer of Patients:','Population'],

        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;") 

    )

)

m.add_child(INT)

m.keep_in_front(INT)

folium.LayerControl().add_to(m)



#m.save('District_final.html')