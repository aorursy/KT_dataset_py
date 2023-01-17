import pandas as pd



filename = "../input/covid19-municpios-de-so-paulo/covid-19-municipios-sp.csv"

df = pd.read_csv(filename, sep=",")

df.head(3)
import folium



def generateBaseMap(default_location=[-22.292690, -48.558171], default_zoom_start=7):

    base_map = folium.Map(location=default_location, control_scale=True, zoom_start=default_zoom_start)

    return base_map



generateBaseMap()
data = df[['Latitude', 'Longitude', 'Obitos']].groupby(['Latitude', 'Longitude']).sum().reset_index().values.tolist()

data[:5]
from folium.plugins import HeatMap



map_confirmados = generateBaseMap()

HeatMap(data=data, radius=8, max_zoom=13).add_to(map_confirmados)

map_confirmados
# o código usado para gerar o mapa do gif acima

from folium.plugins import HeatMapWithTime



data_by_day = []

for day in df.Dia.sort_values().unique():

    data_by_day.append(df.loc[df.Dia == day, ['Latitude', 'Longitude', 'Obitos']].groupby(['Latitude', 'Longitude']).sum().reset_index().values.tolist())



mortes_confirmadas_map = generateBaseMap()

casos_covid_time = HeatMapWithTime(data=data_by_day, max_opacity=0.6,radius=13)

casos_covid_time.add_to(mortes_confirmadas_map)