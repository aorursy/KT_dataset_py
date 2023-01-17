import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import folium as fl
from folium.features import CustomIcon

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv(
    "../input/2290_01052016_31052016.csv",
    delimiter=",",
    dtype = {
        "numero_de_serie":int,
        "milissegundo":int,
        "faixa":int,
        "velocidade_entrada":int,
        "velocidade_saida":int,
        "classificacao":int,
        "tamanho": float,
        "placa":str,
        "tempo_ocupacao_laco":int
    },
    parse_dates=["data_hora"]
)
data.describe().transpose()
sent_list = pd.read_csv("../input/tabela_sentido.csv", delimiter=",")
class_list = pd.read_csv("../input/tabela_classificao.csv", delimiter=",")
eqp_list = pd.read_csv("../input/tabela_equipamento.csv", delimiter=",")
positions = eqp_list[["numero_de_serie", "latitude", "longitude"]]
positions = np.array(positions[~positions.duplicated()])
m = fl.Map(location=[-23.483, -47.4440], zoom_start = 14, width=800, height=600, tiles='openstreetmap')
featureGroup = fl.FeatureGroup("Locations")
for p in positions:
    featureGroup.add_child(fl.Marker(location=[float(p[1]), float(p[2])], popup = 'Equipamento: ' + str(p[0]).split(".")[0]))
    m.add_child(featureGroup)
m