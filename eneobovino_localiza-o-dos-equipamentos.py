import pandas as pd
import numpy as np
import folium
from folium.features import CustomIcon
eqp_list = pd.read_csv("../input/tabela_equipamento.csv")
positions = eqp_list[["numero_de_serie","latitude", "longitude"]]
positions = np.array(positions[~positions.duplicated()])
m = folium.Map(location=[-23.483, -47.4440], zoom_start=14, width=800, height=600, tiles='openstreetmap') # tiles: 'openstreetmap';'Stamen Terrain'
feature_group = folium.FeatureGroup("Locations")
for p in positions:
    feature_group.add_child(folium.Marker(location=[float(p[1]),float(p[2])], popup='Equipamento: '+str(p[0]).split(".")[0]))
m.add_child(feature_group)
m