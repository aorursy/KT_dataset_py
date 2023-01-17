import folium 
import geopandas as gpd
import pandas as pd
from folium.plugins import MarkerCluster
from folium.features import CustomIcon
from folium.plugins import MeasureControl
from folium.plugins import Draw
veri_ispark=pd.read_excel('../input/ibb-ak-veri/ispark-otoparklarna-ait-bilgiler.xlsx')
veri_kart=pd.read_excel('../input/ibb-kart-dolum-merkezleri/istanbulkart-dolum-merkezi-bilgileri.xlsx')
veri_sağlık=pd.read_csv('../input/health-institutions-datas-of-istanbul/salk-kurum-ve-kurulularna-ait-bilgiler.csv',sep=',',encoding='latin-1')
veri_ispark=veri_ispark.dropna(subset=['Enlem'])
veri_ispark=veri_ispark.dropna(subset=['Boylam'])
veri_kart=veri_kart.dropna(subset=["LONGITUDE"])
veri_kart=veri_kart.dropna(subset=["LATITUDE"])
veri_sağlık=veri_sağlık.dropna(subset=['ENLEM'])
veri_sağlık=veri_sağlık.dropna(subset=['BOYLAM'])
veri_kart.drop(veri_kart.loc[veri_kart.LONGITUDE==28.663911].index, inplace=True)
veri_kart.drop(veri_kart.loc[veri_kart.LONGITUDE==24.10052475].index, inplace=True)
veri_kart.drop(veri_kart.loc[veri_kart.LONGITUDE==29.844776].index, inplace=True)
veri_kart.drop(veri_kart.loc[veri_kart.LONGITUDE==28.9853446].index, inplace=True)
veri_kart.drop(veri_kart.loc[veri_kart.LONGITUDE==29.27964].index, inplace=True)
veri_kart.drop(veri_kart.loc[veri_kart.LONGITUDE==41.036632].index, inplace=True)
veri_kart.drop(veri_kart.loc[veri_kart.LONGITUDE==28.100232].index, inplace=True)
ispark=veri_ispark[['Enlem','Boylam']].values.tolist()
kart=veri_kart[['LATITUDE','LONGITUDE']].values.tolist()
sağlık=veri_sağlık[['ENLEM','BOYLAM']].values.tolist()
m=folium.Map(location=(41.1944, 28.9651),
            zoomstart=9,tiles='openstreetmap')
m
folium.TileLayer('cartodbdark_matter').add_to(m);
folium.TileLayer('cartodbpositron').add_to(m);
folium.TileLayer(
        tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr = 'Esri',
        name = 'Esri Satellite',
        overlay = False,
        control = True
       ).add_to(m);
tabaka1=folium.FeatureGroup(name="İspark Otopark")
tabaka2=folium.FeatureGroup(name='Kart Dolum Merkezleri')
tabaka3=folium.FeatureGroup(name='Sağlık Kuruluşları')
marker_cluster =MarkerCluster(data=ispark).add_to(tabaka1)
for point in range(len(ispark)):
    İSPARK= folium.Marker(ispark[point],popup=str('Park ID'),
                          icon=folium.Icon(color='blue', icon='car',prefix='fa')).add_to(marker_cluster)
marker_cluster=MarkerCluster(data=kart).add_to(tabaka2)
for point in range(len(kart)):
    KART=folium.Marker(kart[point],
                       icon=folium.Icon(color='green', icon='ticket',prefix='fa')).add_to(marker_cluster)
marker_cluster=MarkerCluster(data=sağlık).add_to(tabaka3)
for point in range(len(sağlık)):
    SAĞLIK=folium.Marker(sağlık[point],
                        icon=folium.Icon(color='red',icon='heartbeat',prefix='fa')).add_to(marker_cluster)
draw = Draw()#çizim araçlarını oluşturma
draw.add_to(m)#çizim araçlarını haritaya ekleme
m.add_child(tabaka1)#tabakaları haritaya ekleme
m.add_child(tabaka2)
m.add_child(tabaka3)
m.add_child(MeasureControl())#mesafeleri ölçmek için
m.add_child(folium.map.LayerControl())
m.save('index.html')
m
