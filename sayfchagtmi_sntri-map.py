import folium
import pandas as pd
data = pd.read_csv('../input/cartographie-des-stations-des-sntri/cartographie_des_stations_sntri.csv')
data = data.dropna()
data.head()
data2 = pd.read_csv('../input/poll-offices-in-tunisia/VC_level_-registration_record_info-age_and_gender_info-10-08-2017.csv')
data2 = data2.dropna()
data2.head()
# Initialize an empty map
my_map = folium.Map(location=[34.9,9.5], tiles="Mapbox Bright", zoom_start=10)
# Add Marker for every location 
# I can add marker one by one on the map
for i in range(0,len(data)):
    folium.Circle([data.iloc[i]['Latitude'],data.iloc[i]['Longitude']],
                  popup=data.iloc[i]['Rapport Liste des'],
                  radius=int(data.iloc[i]['Zoom']*data.iloc[i]['Rayon']),
                  color='crimson',fill=True,fill_color='crimson').add_to(my_map)
infos = ""
for j in range(0,len(data2)):
    infos = "f_p51:  "+str(data2.iloc[j]['f_p51'])+"\n"+"m_p51:  "+str(data2.iloc[j]['m_p51'])+"\n"+"f_36_50:  "+str(data2.iloc[j]['f_36_50'])+"\n"+"m_36_50:  "+str(data2.iloc[j]['m_36_50'])+"\n"+"f_25_35:  "+str(data2.iloc[j]['f_25_35'])+"\n"+"m_25_35:  "+str(data2.iloc[j]['m_25_35'])+"\n"+"f_22_24:  "+str(data2.iloc[j]['f_22_24'])+"\n"+"m_22_24:  "+str(data2.iloc[j]['m_22_24'])+"\n"+"sum:  "+str(data2.iloc[j]['sum'])+"\n"+"polling:  "+str(data2.iloc[j]['polling'])+"\n"+"mun:  "+str(data2.iloc[j]['mun'])+"\n"+"gouv:  "+str(data2.iloc[j]['gouv'])+"\n"+"gouv_en:  "+str(data2.iloc[j]['gouv_en'])+"\n"
    folium.CircleMarker([data2.iloc[j]['lat'],data2.iloc[j]['lon']],
                  popup=infos,
                  radius = 0.5).add_to(my_map)
#View the map
    # The map may take a while to show up
my_map
# Save tha map in html format
my_map.save('../Map.html')

