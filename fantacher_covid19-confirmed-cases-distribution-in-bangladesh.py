from  geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="http")
import folium
import pandas as pd
df=pd.read_csv("/kaggle/input/covid19-confirmed-cases-in-bangladesh/BD_COVID19_data.csv")
df.head()
df.drop(['Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
df.head()
districts=df['Districts']
country ="Bangladesh"
lat=[]
long=[]
for district in districts:
    
    loc = geolocator.geocode(district +','+ country)
    if loc:
        lat.append(loc.latitude)
        long.append(loc.longitude)
    else:
        lat.append('loc.latitude')
        long.append('loc.longitude')
df['Latitude']=lat
df['Longitude']=long
df.isnull()
f = folium.Figure(width=1100, height=1000)
m=folium.Map([23.8107, 90.4126], zoom_start=8).add_to(f)
import colorama
from colorama import Fore, Style
fg= folium.map.FeatureGroup()
for lat, long,district,case in zip(df.Latitude, df.Longitude,df.Districts,df.Total_Cases):    
    test ='Total Confirmed case in '+district+' is: '+str(case)
    popup = folium.Popup(test, max_width=500)
    folium.Marker(location=[lat, long], popup=popup,icon=folium.Icon(color='blue',icon='info-sign')).add_to(m)
m.add_child(fg)
