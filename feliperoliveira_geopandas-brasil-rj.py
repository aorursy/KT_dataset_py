%matplotlib inline
import pandas as pd
import geopandas as gpd
import os
import zipfile
import shutil
import folium
from folium.plugins import MarkerCluster
from folium.plugins import HeatMap
#Ler os arquivos shape do estado do Rio de Janeiro 
rj_est=gpd.read_file('../input/datario/Mapas/RJ/33MUE250GC_SIR.shp')
#Visualização preliminar
rj_est.sample(10)
#Total de municípios no estado 
print(rj_est['NM_MUNICIP'].count(),'municípios')
Estado_Rio=rj_est.plot(color='white', edgecolor='black', figsize=(12,12))
cidade_rio=rj_est[rj_est.NM_MUNICIP=='RIO DE JANEIRO']
Cidade_Rio=cidade_rio.plot(color='white', edgecolor='black', figsize=(12,12))
#Salvar informações filtradas
dir='../input/datario/Mapas/RJ-MUNIC'
if not os.path.exists(dir):
    os.makedirs(dir)
dados_aluguel = pd.read_table('../input/datario/dados.txt',sep='\t')
dados_aluguel.sample(10)
bairros_dataframe=pd.DataFrame(dados_aluguel['Bairro'].unique())

print(dados_aluguel['Bairro'].count(), 'imóveis no total')
print(bairros_dataframe[0].count(), 'bairros no total')
# Criar geo data frame
from shapely.geometry import Point
x=zip(dados_aluguel.Longitude,dados_aluguel.Latitude)
geometry=[Point(x) for x in zip(dados_aluguel.Longitude,dados_aluguel.Latitude) ]
crs = {'proj': 'latlong', 'ellps': 'WGS84', 'datum': 'WGS84', 'no_defs': True}
geo_dados=gpd.GeoDataFrame(dados_aluguel,crs=crs,geometry=geometry)
geo_dados.head(10)
# Criar pasta no diretório local
dir='..\input\datario\Mapas\RJ-DATASET'
if not os.path.exists(dir):
    os.makedirs(dir) 
geo_dados.to_file(dir+'/DATASET.shp')
#Mudar crs do municipio
cidade_rio = cidade_rio.to_crs('+proj=utm +zone=23 +south +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=km +no_defs')
geo_dados = geo_dados.to_crs('+proj=utm +zone=23 +south +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=km +no_defs')
#Eliminar imóveis fora do limite da cidade
geo_dados=geo_dados[geo_dados['geometry'].within(cidade_rio.iloc[0].geometry)]

#Plotar
base=cidade_rio.plot(color='white', edgecolor='black', figsize=(12,12))
imoveis_cidade_rio=geo_dados.plot(ax=base,color='red',alpha=0.3)
metro = pd.read_table('../input/datario/metro.txt',sep='\t')
metro.sample(10)
print(metro['Nome'].count(), 'estações no total')
#Transformar o dataframe em geodataframe
from shapely.geometry import Point
x=zip(metro.Longitude,metro.Latitude)

# Criar geo data frame
geometry=[Point(x) for x in zip(metro.Longitude,metro.Latitude) ]
crs = {'proj': 'latlong', 'ellps': 'WGS84', 'datum': 'WGS84', 'no_defs': True}
metro_geo_dados=gpd.GeoDataFrame(metro,crs=crs,geometry=geometry)
#Converter crs
metro_geo_dados = metro_geo_dados.to_crs('+proj=utm +zone=23 +south +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=km +no_defs')
metro_geo_dados.to_file('..\input\datario\Mapas\RJ-DATASET\METRO.shp')

# Visualizar
base = cidade_rio.plot(color='white', edgecolor='black', figsize=(12,12))
geo_dados.plot(ax=base, color='red', alpha=0.1)
plot_metro=metro_geo_dados.plot(ax=base, color='navy', markersize= 30)
trem= gpd.read_file('../input/datario/Estaes_Trem.geojson')
trem = trem.to_crs('+proj=utm +zone=23 +south +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=km +no_defs')
trem.sample(5)
print(trem['Nome'].count(), 'estações no total')
#Apenas as estações dentro da cidade
trem = trem[trem.within(cidade_rio.iloc[0].geometry)]
base = cidade_rio.plot(color='white', edgecolor='black', figsize=(12,12))
geo_dados.plot(ax=base, color='red', alpha=0.1)
trem_rio=trem.plot(ax=base, color='darkgreen',marker='^', markersize= 30)
BRT= gpd.read_file('../input/datario/Estaes_BRT.geojson')
BRT = BRT.to_crs('+proj=utm +zone=23 +south +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=km +no_defs')
BRT.sample(5)
print(BRT['Nome'].count(), 'estações no total')
base = cidade_rio.plot(color='white', edgecolor='black', figsize=(12,12))
geo_dados.plot(ax=base, color='red', alpha=0.1)
BRT_cidade=BRT.plot(ax=base, color='black',marker='x',markersize= 30)
