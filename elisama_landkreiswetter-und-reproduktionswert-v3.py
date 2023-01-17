# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import geopandas as gpd

import numpy as np

from pathlib import Path

import plotly.offline as py

import plotly.express as px

import cufflinks as cf

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error as MAE

from sklearn.cluster import DBSCAN

from scipy.stats import linregress

from scipy.stats import pearsonr

bw = pd.read_csv('/kaggle/input/landkreise/Landkreise und Wetter.csv', encoding ='latin1', sep=";")

sh = pd.read_csv('/kaggle/input/schleswigholstein/Landkreise und Wetter SH.csv', encoding ='latin1', sep=";")

b = pd.read_csv('/kaggle/input/bayernniedersachsenges3/Landkreise und Wetter Bayern.csv', encoding ='latin1', sep=";")

nis = pd.read_csv('/kaggle/input/bayernniedersachsenges3/Landkreise und Wetter Niedersachsen.csv', encoding ='latin1', sep=";")

gesamt = pd.read_csv('/kaggle/input/gesamt4/Landkreise und Wetter gesamt 4.csv', encoding ='latin1', sep=";")
bw
bwcorr = bw.corr()

bwcorr['Änderung in % zu Vorwoche']
pearsonr(bw['Änderung in % zu Vorwoche'], bw['temp avg Woche'])
# Daten nach Luftfeutigkeit gefiltert

hum70bw = bw[bw['humidity avg Woche'] < 70.0]

hum60bw = bw[bw['humidity avg Woche'] < 60.0]

hum50bw = bw[bw['humidity avg Woche'] < 50.0]

hum55bw = bw[bw['humidity avg Woche'] < 55.0]
# 1 --> korreliert

# 0 --> kein Zusammenhang

#-1 --> korriliert entegengesetzt
hum70bwcorr = hum70bw.corr()

hum70bwcorr['Änderung in % zu Vorwoche']
pearsonr(hum70bw['Änderung in % zu Vorwoche'], hum70bw['temp avg Woche'])
hum60bwcorr = hum60bw.corr()

hum60bwcorr['Änderung in % zu Vorwoche']
pearsonr(hum60bw['Änderung in % zu Vorwoche'], hum60bw['temp avg Woche'])
hum50bwcorr = hum50bw.corr()

hum50bwcorr['Änderung in % zu Vorwoche']
pearsonr(hum50bw['Änderung in % zu Vorwoche'], hum50bw['temp avg Woche'])
hum50bw.info()
hum55bwcorr = hum55bw.corr()

hum55bwcorr['Änderung in % zu Vorwoche']
pearsonr(hum55bw['Änderung in % zu Vorwoche'], hum55bw['temp avg Woche'])
hum70bwcorr.plot(kind='scatter', x='temp avg Woche',y='Änderung in % zu Vorwoche', c='blue', title='Streuung der Korrelation - Luftfeuchtigkeit unter 70%').axis([-1,1,-1,1])

hum60bwcorr.plot(kind='scatter', x='temp avg Woche',y='Änderung in % zu Vorwoche', c='blue', title='Streuung der Korrelation - Luftfeuchtigkeit unter 60%').axis([-1,1,-1,1])
hum55bwcorr.plot(kind='scatter', x='temp avg Woche',y='Änderung in % zu Vorwoche', c='blue', title='Streuung der Korrelation - Luftfeuchtigkeit unter 55%').axis([-1,1,-1,1])
hum50bwcorr.plot(kind='scatter', x='temp avg Woche',y='Änderung in % zu Vorwoche', c='blue', title='Streuung der Korrelation - Luftfeuchtigkeit unter 50%').axis([-1,1,-1,1])
temp15bw = bw[bw['temp avg Woche'] < 15.0] 

temp10bw = bw[bw['temp avg Woche'] < 10.0]

temp5bw = bw[bw['temp avg Woche'] < 5.0]
temp15bw = temp15bw.corr()

temp15bw['Änderung in % zu Vorwoche']
temp10bw = temp10bw.corr()

temp10bw['Änderung in % zu Vorwoche']
temp5bw = temp5bw.corr()

temp5bw['Änderung in % zu Vorwoche']
#je kälter es wird umso stärke steigt die Anzahl
temp15bw.plot(kind='scatter', x='humidity avg Woche',y='Änderung in % zu Vorwoche', c='blue', title='Streuung der Korrelation - Temperatur unter 15°').axis([0,100,-1000,1000])
temp10bw.plot(kind='scatter', x='humidity avg Woche',y='Änderung in % zu Vorwoche', c='blue', title='Streuung der Korrelation - Temperatur unter 10°').axis([0,100,-1000,1000])
temp5bw.plot(kind='scatter', x='humidity avg Woche',y='Änderung in % zu Vorwoche', c='blue', title='Streuung der Korrelation - Temperatur unter 5°').axis([0,100,-1000,1000])
#Darstellung der Korrelation in einer Heatmap (Luftfeuchtigkeit < 47%)

mask = np.zeros_like(hum55bwcorr)

mask[np.triu_indices_from(mask)] = True

heatmapG55bw = sns.heatmap(hum55bwcorr, mask = mask, annot=True, cmap="YlGnBu")

heatmapG55bw.set_title('Korrelation - Luftfeuchtigkeit unter 55%')
sh
shcorr = sh.corr()

shcorr['Änderung in % zur Vorwoche']
# Daten nach Luftfeutigkeit gefiltert

hum70sh = sh[sh['humidity avg Woche'] < 70.0]

hum60sh = sh[sh['humidity avg Woche'] < 60.0]





#hum55 hat hier bereits nur 4 Einträge

#hum 60 hat noch 15 Einträge

#hum58 hat bereits nur noch 5 Einträge
hum70shcorr = hum70sh.corr()

hum70shcorr['Änderung in % zur Vorwoche']
hum60shcorr = hum60sh.corr()

hum60shcorr['Änderung in % zur Vorwoche']
hum70shcorr.plot(kind='scatter', x='temp avg Woche',y='Änderung in % zur Vorwoche', c='blue', title='Streuung der Korrelation - Luftfeuchtigkeit unter 70%').axis([-1,1,-1,1])
hum60shcorr.plot(kind='scatter', x='temp avg Woche',y='Änderung in % zur Vorwoche', c='blue', title='Streuung der Korrelation - Luftfeuchtigkeit unter 60%').axis([-1,1,-1,1])
temp5sh = sh[sh['temp avg Woche'] < 5]

temp10sh = sh[sh['temp avg Woche'] < 10]

temp15sh = sh[sh['temp avg Woche'] < 15]
temp5sh.info()
temp5shcorr = temp5sh.corr()

temp5shcorr['Änderung in % zur Vorwoche']
temp10shcorr = temp10sh.corr()

temp10shcorr['Änderung in % zur Vorwoche']
temp15shcorr = temp15sh.corr()

temp15shcorr['Änderung in % zur Vorwoche']
temp5sh.plot(kind='scatter', x='humidity avg Woche',y='Änderung in % zur Vorwoche', c='blue', title='Streuung der Korrelation - Temperatur unter 5 Grad').axis([0,100,-1000,1000])
temp10sh.plot(kind='scatter', x='humidity avg Woche',y='Änderung in % zur Vorwoche', c='blue', title='Streuung der Korrelation - Temperatur unter 10 Grad').axis([0,100,-1000,1000])
temp15sh.plot(kind='scatter', x='humidity avg Woche',y='Änderung in % zur Vorwoche', c='blue', title='Streuung der Korrelation - Temperatur unter 15 Grad').axis([0,100,-1000,1000])
bcorr = b.corr()

bcorr['Änderung in % zu Vorwoche']
pearsonr(b['Änderung in % zu Vorwoche'], b['temp avg Woche'])
# Daten nach Luftfeutigkeit gefiltert

hum70b = b[b['humidity avg Woche'] < 70.0]

hum60b = b[b['humidity avg Woche'] < 60.0]

hum50b = b[b['humidity avg Woche'] < 50.0]

#hum47b.info()
hum70bcorr = hum70b.corr()

hum70bcorr['Änderung in % zu Vorwoche']
pearsonr(hum70b['Änderung in % zu Vorwoche'], hum70b['temp avg Woche'])
hum60bcorr = hum60b.corr()

hum60bcorr['Änderung in % zu Vorwoche']
pearsonr(hum60b['Änderung in % zu Vorwoche'], hum60b['temp avg Woche'])
hum50bcorr = hum50b.corr()

hum50bcorr['Änderung in % zu Vorwoche']
pearsonr(hum50b['Änderung in % zu Vorwoche'], hum50b['temp avg Woche'])
niscorr = nis.corr()

niscorr['Änderung in % zu Vorwoche']
pearsonr(nis['Änderung in % zu Vorwoche'], nis['temp avg Woche'])
# Daten nach Luftfeutigkeit gefiltert

hum70nis = nis[nis['humidity avg Woche'] < 70.0]

hum60nis = nis[nis['humidity avg Woche'] < 60.0]

hum50nis = nis[nis['humidity avg Woche'] < 50.0]
hum70niscorr = hum70nis.corr()

hum70niscorr['Änderung in % zu Vorwoche']
pearsonr(hum70nis['Änderung in % zu Vorwoche'], hum70nis['temp avg Woche'])
hum60niscorr = hum60nis.corr()

hum60niscorr['Änderung in % zu Vorwoche']
pearsonr(hum60nis['Änderung in % zu Vorwoche'], hum60nis['temp avg Woche'])
hum50niscorr = hum50nis.corr()

hum50niscorr['Änderung in % zu Vorwoche']
pearsonr(hum50nis['Änderung in % zu Vorwoche'], hum50nis['temp avg Woche'])
gesamt
gesamtcorr = gesamt.corr()

gesamtcorr['Änderung in % zu Vorwoche']
# Daten nach Luftfeutigkeit gefiltert

hum70gesamt = gesamt[gesamt['humidity avg Woche'] < 70.0]

hum60gesamt = gesamt[gesamt['humidity avg Woche'] < 60.0]

hum50gesamt = gesamt[gesamt['humidity avg Woche'] < 50.0]

hum47gesamt = gesamt[gesamt['humidity avg Woche'] < 47.0]

hum70gesamtcorr = hum70gesamt.corr()

hum70gesamtcorr['Änderung in % zu Vorwoche']
hum60gesamtcorr = hum60gesamt.corr()

hum60gesamtcorr['Änderung in % zu Vorwoche']
hum50gesamtcorr = hum50gesamt.corr()

hum50gesamtcorr['Änderung in % zu Vorwoche']
hum47gesamtcorr = hum47gesamt.corr()

hum47gesamtcorr['Änderung in % zu Vorwoche']
hum70gesamtcorr.plot(kind='scatter', x='temp avg Woche',y='Änderung in % zu Vorwoche', c='blue', title='Streuung der Korrelation - Luftfeuchtigkeit unter 70%').axis([-1,1,-1,1])
hum60gesamtcorr.plot(kind='scatter', x='temp avg Woche',y='Änderung in % zu Vorwoche', c='blue', title='Streuung der Korrelation - Luftfeuchtigkeit unter 60%').axis([-1,1,-1,1])
hum50gesamtcorr.plot(kind='scatter', x='temp avg Woche',y='Änderung in % zu Vorwoche', c='blue', title='Streuung der Korrelation - Luftfeuchtigkeit unter 50%').axis([-1,1,-1,1])
hum47gesamtcorr.plot(kind='scatter', x='temp avg Woche',y='Änderung in % zu Vorwoche', c='blue', title='Streuung der Korrelation - Luftfeuchtigkeit unter 47%').axis([-1,1,-1,1])
temp5gesamt = gesamt[gesamt['temp avg Woche'] < 5]

temp10gesamt = gesamt[gesamt['temp avg Woche'] < 10]

temp15gesamt = gesamt[gesamt['temp avg Woche'] < 15]
temp5gesamtcorr = temp5gesamt.corr()

temp5gesamtcorr['Änderung in % zu Vorwoche']
temp10gesamtcorr = temp10gesamt.corr()

temp10gesamtcorr['Änderung in % zu Vorwoche']
temp15gesamtcorr = temp15gesamt.corr()

temp15gesamtcorr['Änderung in % zu Vorwoche']
temp5gesamt.plot(kind='scatter', x='humidity avg Woche',y='Änderung in % zu Vorwoche', c='blue', title='Streuung der Korrelation - Temperatur unter 5 Grad').axis([0,100,-1000,1000])
temp10gesamt.plot(kind='scatter', x='humidity avg Woche',y='Änderung in % zu Vorwoche', c='blue', title='Streuung der Korrelation - Temperatur unter 10 Grad').axis([0,100,-1000,1000])
temp15gesamt.plot(kind='scatter', x='humidity avg Woche',y='Änderung in % zu Vorwoche', c='blue', title='Streuung der Korrelation - Temperatur unter 15 Grad').axis([0,100,-1000,1000])