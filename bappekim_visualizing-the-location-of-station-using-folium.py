import pandas as pd 

import random



# for visualization

import folium

import json
# load data

df = pd.read_csv('../input/air-pollution-in-seoul/AirPollutionSeoul/Measurement_summary.csv')

df_item = pd.read_csv('../input/air-pollution-in-seoul/AirPollutionSeoul/Original Data/Measurement_item_info.csv')

df_station = pd.read_csv('../input/air-pollution-in-seoul/AirPollutionSeoul/Original Data/Measurement_station_info.csv')


center = [37.541, 126.986] # center of Seoul

m = folium.Map(location=center, zoom_start=11) # set map



# load district information

with open('../input/seoul-geo-json/seoul_municipalities_geo.json',mode='rt',encoding='utf-8') as f:

    geo = json.loads(f.read())

    f.close()



# Add geojson to folium

folium.GeoJson(

    geo,

    name='seoul_municipalities'

).add_to(m)



# Add marker

for i in df_station.index[:25]: 

    popup_str = 'Station ' + str(df_station.loc[i, 'Station code'])

    folium.Marker(df_station.loc[i, ['Latitude', 'Longitude']],

                  popup=popup_str,

                  icon=folium.Icon(color='black')).add_to(m)



m # print



def get_criteria(df_item, item):

    criteria = df_item[df_item['Item name'] == item].iloc[0, 3:]

    return criteria



def seoulmap(df_day, df_item, item):

    criteria = get_criteria(df_item, item)

    

    dfm = df_day.copy()

    

    # set color of marker

    dfm['color'] = ''

    dfm.loc[dfm[item] <= criteria[3], 'color'] = 'red'

    dfm.loc[dfm[item] <= criteria[2], 'color'] = 'orange' # yellow

    dfm.loc[dfm[item] <= criteria[1], 'color'] = 'green'

    dfm.loc[dfm[item] <= criteria[0], 'color'] = 'blue'

    

    center = [37.541, 126.986] # center of Seoul

    m = folium.Map(location=center, zoom_start=11) # set map



    with open('../input/seoul-geo-json/seoul_municipalities_geo.json',mode='rt',encoding='utf-8') as f:

        geo = json.loads(f.read())

        f.close()



    folium.GeoJson(

        geo,

        name='seoul_municipalities'

    ).add_to(m)



    for i in dfm.index: 

        popup_str = 'Station ' + str(dfm.loc[i, 'Station code']) + ': ' + str(dfm.loc[i, item])

        folium.Marker(dfm.loc[i, ['Latitude', 'Longitude']],

                      popup=popup_str,

                      icon=folium.Icon(color=dfm.loc[i, 'color'])).add_to(m)

    

    return m



random.seed(0)

ind = random.randint(1, len(df))



day = df.loc[ind, 'Measurement date']

print(day)

df_day = df[df['Measurement date'] == day]



seoulmap(df_day, df_item, 'PM10')
seoulmap(df_day, df_item, 'PM2.5')
random.seed(1)

ind = random.randint(1, len(df))



day = df.loc[ind, 'Measurement date']

print(day)

df_day = df[df['Measurement date'] == day]



seoulmap(df_day, df_item, 'PM10')
seoulmap(df_day, df_item, 'PM2.5')