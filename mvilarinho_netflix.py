# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

datos=pd.read_csv("/kaggle/input/netflix-shows/netflix_titles.csv")
datos.head()
pelisEsp=datos.loc[datos['country']=='Spain'].sort_values(by=['release_year'],ascending=False)
pelisEsp.head()
listaAnos=list(pelisEsp['release_year'])
pelisEspGraf=pelisEsp.groupby(['release_year']).size().reset_index()
pelisEspGraf[0]
import matplotlib.cm as cm
from matplotlib import colors

colorMap=cm.get_cmap('viridis')
rs=np.linspace(0,1,pelisEspGraf['release_year'].__len__())
cor=[]
for r in rs:
    cor.append(colors.to_rgba(colorMap(r)))
_,ax=plt.subplots(figsize=(25,10))
ax.set_title("Número de Producións Españolas en Netflix",size=40)
ax.set_xlabel("Ano de Produción",size=20)
ax.set_ylabel("Número de Producións",size=20)
ax.bar(pelisEspGraf['release_year'],height=pelisEspGraf[0],color=cor, tick_label=pelisEspGraf['release_year'])
cadea='Hola Mundo'
cadea.split(" ")[0]
pelis=datos.loc[datos.type=='Movie']
pelisSort=pelis.duration.sort_values()
pelisSort.str.split(" ",expand=True)
duracion=pelisSort.str.split(" ",expand=True)[0]
duracion=pd.DataFrame((map(int,duracion)))
duracion=duracion.sort_values(0)
duracion.head()
bins=[x for x in range(0,255,5)]
counts, bins = np.histogram(duracion,bins=bins)
plt.figure(figsize=(20,8))
plt.hist(bins[:-1], bins, weights=counts,color='darkblue')
plt.plot(bins[:-1],counts)
plt.title("Histograma de Frecuencias",size=20)
#/kaggle/input/tour-de-france-winners/stage_data.csv
#/kaggle/input/tour-de-france-winners/tdf_stages.csv
#/kaggle/input/tour-de-france-winners/tdf_winners.csv

datos=pd.read_csv("/kaggle/input/tour-de-france-winners/stage_data.csv")
datos.head()
datosInd=datos.loc[datos.rider.str.find('Indurain')>-1]


datosInd=datosInd.loc[datosInd['rank']=='1']
datosInd
#/kaggle/input/fifa-world-cup/WorldCups.csv
datos=pd.read_csv("/kaggle/input/fifa-world-cup/WorldCupPlayers.csv")
datos.head()
datosEsp=datos.loc[datos['Team Initials']=="ESP"]
datosEsp.tail(11)