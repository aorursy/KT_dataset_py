# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas
plt.rcParams['figure.figsize'] = [15, 10]

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
Data=pd.read_csv("../input/countries of the world.csv")
Data["Country"]=Data["Country"].str.strip()
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
world=world.rename(columns={"name":"Country"})
D1=Data
D2=world
D1.loc[D1["Country"]=="Bahamas, The","Country"]="Bahamas"
#'Congo, Dem. Rep.'       'Dem. Rep. Congo',
#'Congo, Repub. of the'   'Congo',
D1.loc[D1["Country"]=="Congo, Dem. Rep.","Country"]="Dem. Rep. Congo"
D1.loc[D1["Country"]=="Congo, Repub. of the","Country"]="Congo"
#"Cote d'Ivoire"      "Côte d'Ivoire"
D2.loc[D2["Country"]== "Côte d'Ivoire","Country"]="Cote d'Ivoire" 
D1.loc[D1["Country"]=='Korea, North',"Country"]="North Korea"
D2.loc[D2["Country"]== "Dem. Rep. Korea","Country"]="North Korea"
#'Korea, North',                   Dem. Rep. Korea
D2.loc[D2["Country"]== "Korea","Country"]="South Korea"
D1.loc[D1["Country"]== "Korea, South","Country"]="South Korea"
D2.loc[D2["Country"]== "Bosnia and Herz.","Country"]="Bosnia & Herzegovina"
#Bosnia & Herzegovina' Bosnia and Herz
#Western Sahara   W. Sahara
D2.loc[D2["Country"]== "W. Sahara","Country"]="Western Sahara" 
#'Czech Republic',      'Czech Rep.',
#'Dominican Republic',  'Dominican Rep.',
D2.loc[D2["Country"]=='Czech Rep.',"Country"]="Czech Republic"
D2.loc[D2["Country"]=='Dominican Rep.',"Country"]="Dominican Republic"
#Trinidad & Tobago   Trinidad and Tobago
D2.loc[D2["Country"]=="Trinidad and Tobago","Country"]="Trinidad & Tobago"
#'Equatorial Guinea"  'Eq. Guinea'
D2.loc[D2["Country"]=="Eq. Guinea","Country"]="Equatorial Guinea"
#Solomon Islands  Solomon Is.
D2.loc[D2["Country"]=="Solomon Is.","Country"]="Solomon Islands"
#Burma   "Myanmar"
D1.loc[D1["Country"]=="Burma","Country"]="Myanmar"
#'Gambia, The', "Gambia
D1.loc[D1["Country"]=='Gambia, The',"Country"]="Gambia"
#'East Timor' Timor-Leste
D1.loc[D1["Country"]=='East Timor',"Country"]="Timor-Leste"

#Converted the geopandas dataframe to pandas dataframe and then performed merge operation
D1=pd.DataFrame(D1)
D2=pd.DataFrame(D2)
D3=pd.merge(D1,D2,left_on="Country",right_on="Country",how='inner')
D4=D3
#Performed some preprocessing
D4["Pop. Density (per sq. mi.)"]=D4["Pop. Density (per sq. mi.)"].str.replace(",","")
D4["Coastline (coast/area ratio)"]=D4["Coastline (coast/area ratio)"].str.replace(",","")
D4["Net migration"]=D4["Net migration"].str.replace(",","")
D4["Infant mortality (per 1000 births)"]=D4["Infant mortality (per 1000 births)"].str.replace(",","")
D4["Literacy (%)"]=D4["Literacy (%)"].str.replace(",",".")
D4["Phones (per 1000)"]=D4["Phones (per 1000)"].str.replace(",","")
D4["Arable (%)"]=D4["Arable (%)"].str.replace(",",".")
D4["Crops (%)"]=D4["Crops (%)"].str.replace(",",".")
D4["Other (%)"]=D4["Other (%)"].str.replace(",",".")
D4["Birthrate"]=D4["Birthrate"].str.replace(",","")
D4["Deathrate"]=D4["Deathrate"].str.replace(",","")
D4["Agriculture"]=D4["Agriculture"].str.replace(",","")
D4["Industry"]=D4["Industry"].str.replace(",","")
D4["Service"]=D4["Service"].str.replace(",","")

D4["Pop. Density (per sq. mi.)"]=pd.to_numeric(D4["Pop. Density (per sq. mi.)"])
D4["Coastline (coast/area ratio)"]=pd.to_numeric(D4["Coastline (coast/area ratio)"])

D4["Net migration"]=pd.to_numeric(D4["Net migration"])
D4["Infant mortality (per 1000 births)"]=pd.to_numeric(D4["Infant mortality (per 1000 births)"])

D4["Literacy (%)"]=pd.to_numeric(D4["Literacy (%)"])
D4["Phones (per 1000)"]=pd.to_numeric(D4["Phones (per 1000)"])

D4["Arable (%)"]=pd.to_numeric(D4["Arable (%)"])
D4["Crops (%)"]=pd.to_numeric(D4["Crops (%)"])

D4["Other (%)"]=pd.to_numeric(D4["Other (%)"])
D4["Birthrate"]=pd.to_numeric(D4["Birthrate"])

D4["Deathrate"]=pd.to_numeric(D4["Deathrate"])
D4["Agriculture"]=pd.to_numeric(D4["Agriculture"])

D4["Industry"]=pd.to_numeric(D4["Industry"])
D4["Service"]=pd.to_numeric(D4["Service"])
#Converted the whole merged data to geopandas dataframe
FData=geopandas.GeoDataFrame(D4)

#Final merged data
FData
#Population Map
fig, ax = plt.subplots(1,figsize=(15, 8))
FData.plot(column="Population",cmap="Reds",ax=ax,edgecolor='black', linewidth=1)
ax.axis('off')
vmin = FData['Population'].min()
vmax = FData['Population'].max()
sm = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm._A = []
cbar = fig.colorbar(sm)
#Area Data
fig, ax = plt.subplots(1,figsize=(15, 8))
FData.plot(column="Area (sq. mi.)",cmap="YlOrBr",ax=ax,edgecolor='black', linewidth=1)
ax.axis("off")
vmin = FData['Area (sq. mi.)'].min()
vmax = FData['Area (sq. mi.)'].max()
sm = plt.cm.ScalarMappable(cmap='YlOrBr', norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm._A = []
cbar = fig.colorbar(sm)
#Population Density Data
fig, ax = plt.subplots(1,figsize=(15, 8))
PDD=FData
PDD=PDD.fillna(0)
PDD.plot(column="Pop. Density (per sq. mi.)", cmap='PuRd',ax=ax,edgecolor='black', linewidth=1)
vmin = PDD['Pop. Density (per sq. mi.)'].min()
vmax = PDD['Pop. Density (per sq. mi.)'].max()
sm = plt.cm.ScalarMappable(cmap='PuRd', norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm._A = []
cbar = fig.colorbar(sm)
#Coastline
fig, ax = plt.subplots(1,figsize=(15, 8))
Coast_Data=FData
Coast_Data=Coast_Data.fillna(0)
Coast_Data.plot(column="Coastline (coast/area ratio)",cmap="Blues",ax=ax,edgecolor='black', linewidth=1)
vmin = Coast_Data['Coastline (coast/area ratio)'].min()
vmax = Coast_Data['Coastline (coast/area ratio)'].max()
sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm._A = []
cbar = fig.colorbar(sm)
#Net Migration
fig, ax = plt.subplots(1,figsize=(15, 8))
Net_Migration_Data=FData
Net_Migration_Data=FData.fillna(0)
Net_Migration_Data.plot(column="Net migration",cmap="BuPu",ax=ax,edgecolor='black', linewidth=1)
ax.axis('off')
vmin = Net_Migration_Data['Net migration'].min()
vmax = Net_Migration_Data['Net migration'].max()
sm = plt.cm.ScalarMappable(cmap='BuPu', norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm._A = []
cbar = fig.colorbar(sm)
#Net Infant Morality Rate
fig, ax = plt.subplots(1,figsize=(15, 8))
IMR=FData
IMR=IMR.fillna(0)
IMR.plot(column="Infant mortality (per 1000 births)",cmap="PuRd",ax=ax,edgecolor='black', linewidth=1)
ax.axis('off')
vmin = IMR['Infant mortality (per 1000 births)'].min()
vmax = IMR['Infant mortality (per 1000 births)'].max()
sm = plt.cm.ScalarMappable(cmap='PuRd', norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm._A = []
cbar = fig.colorbar(sm)
#GDP per Capita
fig, ax = plt.subplots(1,figsize=(15, 8))
GDP_Data=FData
GDP_Data=GDP_Data.fillna(0)
GDP_Data.plot(column="GDP ($ per capita)",cmap="magma",ax=ax,edgecolor='black', linewidth=1)
ax.axis('off')
vmin = GDP_Data['GDP ($ per capita)'].min()
vmax = GDP_Data['GDP ($ per capita)'].max()
sm = plt.cm.ScalarMappable(cmap='magma', norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm._A = []
cbar = fig.colorbar(sm)
#Literacy %
fig, ax = plt.subplots(1,figsize=(15, 8))
Literacy_Data=FData
Literacy_Data=Literacy_Data.fillna(0)
Literacy_Data.plot(column="Literacy (%)",cmap="GnBu",ax=ax,edgecolor='black', linewidth=1)
ax.axis('off')
vmin = Literacy_Data['Literacy (%)'].min()
vmax = Literacy_Data['Literacy (%)'].max()
sm = plt.cm.ScalarMappable(cmap='GnBu', norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm._A = []
cbar = fig.colorbar(sm)
#Phones
fig, ax = plt.subplots(1,figsize=(15, 8))
Phone_Data=FData
Phone_Data=Phone_Data.fillna(0)
Phone_Data.plot(column="Phones (per 1000)",cmap="Reds",ax=ax,edgecolor='black', linewidth=1)
ax.axis('off')
vmin = Phone_Data['Phones (per 1000)'].min()
vmax = Phone_Data['Phones (per 1000)'].max()
sm = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm._A = []
cbar = fig.colorbar(sm)
#Birthrate
fig, ax = plt.subplots(1,figsize=(15, 8))
Birth_Data=FData
Birth_Data=Birth_Data.fillna(0)
Birth_Data.plot(column="Birthrate",cmap="Reds",ax=ax,edgecolor='black', linewidth=1)
ax.axis('off')
vmin = Birth_Data['Birthrate'].min()
vmax = Birth_Data['Birthrate'].max()
sm = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm._A = []
cbar = fig.colorbar(sm)
#Deathrate
fig, ax = plt.subplots(1,figsize=(15, 8))
Death_Data=FData
Death_Data=Death_Data.fillna(0)
Death_Data.plot(column="Deathrate",cmap="Greys",ax=ax,edgecolor='black', linewidth=1)
ax.axis('off')
vmin = Death_Data['Deathrate'].min()
vmax = Death_Data['Deathrate'].max()
sm = plt.cm.ScalarMappable(cmap='Greys', norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm._A = []
cbar = fig.colorbar(sm)

#Agriculture
fig, ax = plt.subplots(1,figsize=(15, 8))
Agri_Data=FData
Agri_Data=Agri_Data.fillna(0)
Agri_Data.plot(column="Agriculture",cmap="Greens",ax=ax,edgecolor='black', linewidth=1)
ax.axis('off')
vmin = Agri_Data['Agriculture'].min()
vmax = Agri_Data['Agriculture'].max()
sm = plt.cm.ScalarMappable(cmap='Greens', norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm._A = []
cbar = fig.colorbar(sm)

#Industry
fig, ax = plt.subplots(1,figsize=(15, 8))
Indus_Data=FData
Indus_Data=Indus_Data.fillna(0)
Indus_Data.plot(column="Industry",cmap="Reds",ax=ax,edgecolor='black', linewidth=1)
ax.axis('off')
vmin = Indus_Data['Industry'].min()
vmax = Indus_Data['Industry'].max()
sm = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm._A = []
cbar = fig.colorbar(sm)
fig, ax = plt.subplots(1,figsize=(15, 8))
Service_Data=FData
Service_Data=Service_Data.fillna(0)
Service_Data.plot(column="Service",cmap="Wistia",ax=ax,edgecolor='black', linewidth=1)
ax.axis('off')
vmin = Service_Data['Service'].min()
vmax = Service_Data['Service'].max()
sm = plt.cm.ScalarMappable(cmap='Wistia', norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm._A = []
cbar = fig.colorbar(sm)
#So i think these maps tells a lot