%matplotlib inline

import pandas as pd

import geopandas as gpd

import os

import zipfile

import shutil

import folium

from folium.plugins import MarkerCluster

from folium.plugins import HeatMap
#Import file

PA_St=gpd.read_file('../input/brazil-in-maps-par/PA-IBGE/15SEE250GC_SIR.shp')

#Data in table format

PA_St.head()
PA_Map=PA_St.plot(cmap='YlOrBr', figsize=(12,12))
#Import file

BEL_Ct=gpd.read_file('../input/brazil-in-maps-par/PAR TODOS OS MUNICIPIOS/PARÁ TODOS OS MUNICIPIOS/BELÉM/BELÉM.shp')

#Data in table format

BEL_Ct.head()
BEL_Map=BEL_Ct.plot(cmap='Greens', figsize=(12,12))
#Import file

BEL_Ct_Ub=gpd.read_file('../input/brazil-in-maps-par/PAR TODOS OS MUNICIPIOS/PARÁ TODOS OS MUNICIPIOS/BELÉM/REGIÃO URBANA/BELÉM_URB.shp')

BEL_Ub_Map=BEL_Ct_Ub.plot(cmap='Reds', figsize=(12,12))
BEL_Ct_Ru=gpd.read_file('../input/brazil-in-maps-par/PAR TODOS OS MUNICIPIOS/PARÁ TODOS OS MUNICIPIOS/BELÉM/REGIÃO RURAL/BELÉM_RU.shp')

BEL_Ru_MAp=BEL_Ct_Ru.plot(cmap='GnBu_r', figsize=(12,12))
BEL_Ct_Nhg=gpd.read_file('../input/brazil-in-maps-par/PAR TODOS OS MUNICIPIOS/PARÁ TODOS OS MUNICIPIOS/BELÉM/BAIRROS/Campina de Icoaraci/Campina de Icoaraci.shp')

BEL_Nhg_Map=BEL_Ct_Nhg.plot(cmap='Spectral', figsize=(12,12))
BEL_Ct_Nhg=gpd.read_file('../input/brazil-in-maps-par/PAR TODOS OS MUNICIPIOS/PARÁ TODOS OS MUNICIPIOS/BELÉM/BAIRROS/Batista Campos/Batista Campos.shp')

BEL_Nhg_Map=BEL_Ct_Nhg.plot(cmap='Spectral', figsize=(12,12))
import warnings

warnings.filterwarnings('ignore')
neighbourhood= BEL_Ct_Ub.dissolve(by='NM_BAIRRO')

#Add informations

neighbourhood['Area']=neighbourhood.area

neighbourhood['NM_BAIRRO']=neighbourhood.index

neighbourhood.head()
# Set a default crs 

crs = {'init': 'epsg:4326'}

neighbourhood.to_crs(crs,inplace=True)



#Define the centroids

y=neighbourhood.centroid.y.iloc[0]

x=neighbourhood.centroid.x.iloc[0]



#Map



base = folium.Map([y, x], zoom_start=11, tiles='OpenStreetMap')

base.choropleth(neighbourhood)

base
#Base Map

base = folium.Map([y, x], zoom_start=11, tiles='OpenStreetMap')



for i in range(len(neighbourhood)):

    geo = folium.GeoJson(neighbourhood[i: i + 1],name=neighbourhood['NM_BAIRRO'][i])

    label = '{} - {} Area (Km²)'.format(neighbourhood['NM_BAIRRO'][i], neighbourhood['Area'][i])

    folium.Popup(label).add_to(geo)

    geo.add_to(base)

    

#Layers cntrol 

folium.LayerControl().add_to(base)



# #Save

base.save('Belém (PA).html')



#Show

base