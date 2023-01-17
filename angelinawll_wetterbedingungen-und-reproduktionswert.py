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

## weather = pd.read_csv('/kaggle/input/germanyweather/WetterDeutschland.csv', encoding ='latin1', sep=";")

# latin1 gibt die westeuropäische Sprachen wieder

rRate = pd.read_csv('/kaggle/input/rrate/r-rate.csv',encoding='latin1',sep=";")

weatherMuenchen = pd.read_csv('/kaggle/input/weathermuenchen/WetterMuenchen.csv', encoding ='latin1', sep=";")

weatherDresden = pd.read_csv('/kaggle/input/weathercity/WetterDresden.csv', encoding ='latin1', sep=";")

weatherDuesseldorf = pd.read_csv('/kaggle/input/weathercity/WetterDuesseldorf.csv', encoding ='latin1', sep=";")

weatherHamburg = pd.read_csv('/kaggle/input/weathercity/WetterHamburg.csv', encoding ='latin1', sep=";")

weatherGermany = pd.read_csv('/kaggle/input/deutschland/Deutschland.csv', encoding ='latin1', sep=",")

rkiData = pd.read_csv('/kaggle/input/rkidaten/RKI_COVID19.csv', sep=",")

 #anderer Versuch Datei auszulesen 

 #with open('/kaggle/input/germanyweather/WetterDeutschland.csv','rb') as f:

 #weather = f.read()
# Unbennennung der Spalten für eine erleichterte Filterung

weatherGermany = weatherMuenchen.rename(columns={'Temperature (ø C)': 'Temperatur Avg', 'Dew Point (ø C)': 'Dew Point Avg','Humidity (%)': 'Humidity pct' })

weatherMuenchen= weatherMuenchen.rename(columns={'Temperature (ø C)': 'Temperatur Avg', 'Dew Point (ø C)': 'Dew Point Avg','Humidity (%)': 'Humidity pct' })

weatherDresden= weatherDresden.rename(columns={'Temperature (ø C)': 'Temperatur Avg', 'Dew Point (ø C)': 'Dew Point Avg','Humidity (%)': 'Humidity pct' })

weatherDuesseldorf= weatherDuesseldorf.rename(columns={'Temperature (ø C)': 'Temperatur Avg', 'Dew Point (ø C)': 'Dew Point Avg','Humidity (%)': 'Humidity pct' })

weatherHamburg = weatherHamburg.rename(columns={'Temperature (ø C)': 'Temperatur Avg', 'Dew Point (ø C)': 'Dew Point Avg','Humidity (%)': 'Humidity pct' })

rRate.info()
rRate = rRate.rename(columns={'Datum des Erkrankungsbeginns': 'Datum'})

rRate.head(3)
# Entfernen der nicht genutzten Spalten 

weatherGermany = weatherGermany.drop(columns=['Unnamed: 7','Unnamed: 8','Unnamed: 9','Unnamed: 10','Unnamed: 11','Unnamed: 12','Unnamed: 13','Unnamed: 14','Unnamed: 15','Unnamed: 16', 'Unnamed: 17', 'Unnamed: 18', 'Unnamed: 19', 'Unnamed: 20', 'Unnamed: 21', 'Unnamed: 22', 'Unnamed: 23', 'Unnamed: 24', 'Unnamed: 25'])

weatherMuenchen = weatherMuenchen.drop(columns=['Unnamed: 7','Unnamed: 8','Unnamed: 9','Unnamed: 10','Unnamed: 11','Unnamed: 12','Unnamed: 13','Unnamed: 14','Unnamed: 15','Unnamed: 16', 'Unnamed: 17', 'Unnamed: 18', 'Unnamed: 19', 'Unnamed: 20', 'Unnamed: 21', 'Unnamed: 22', 'Unnamed: 23', 'Unnamed: 24', 'Unnamed: 25'])

weatherDresden = weatherDresden.drop(columns=['Unnamed: 7','Unnamed: 8','Unnamed: 9','Unnamed: 10','Unnamed: 11','Unnamed: 12','Unnamed: 13','Unnamed: 14','Unnamed: 15','Unnamed: 16', 'Unnamed: 17', 'Unnamed: 18', 'Unnamed: 19', 'Unnamed: 20', 'Unnamed: 21', 'Unnamed: 22', 'Unnamed: 23', 'Unnamed: 24'])

weatherDuesseldorf = weatherDuesseldorf.drop(columns=['Unnamed: 7','Unnamed: 8','Unnamed: 9','Unnamed: 10','Unnamed: 11','Unnamed: 12','Unnamed: 13','Unnamed: 14','Unnamed: 15','Unnamed: 16', 'Unnamed: 17', 'Unnamed: 18', 'Unnamed: 19', 'Unnamed: 20', 'Unnamed: 21', 'Unnamed: 22', 'Unnamed: 23', 'Unnamed: 24'])

weatherHamburg = weatherHamburg.drop(columns=['Unnamed: 7','Unnamed: 8','Unnamed: 9','Unnamed: 10','Unnamed: 11','Unnamed: 12','Unnamed: 13','Unnamed: 14','Unnamed: 15','Unnamed: 16', 'Unnamed: 17', 'Unnamed: 18', 'Unnamed: 19', 'Unnamed: 20', 'Unnamed: 21', 'Unnamed: 22', 'Unnamed: 23', 'Unnamed: 24'])

rRate.info()

rRate = rRate.drop(columns=['Untere Grenze des 95%-Prädiktionsintervalls der Anzahl Neuerkrankungen (ohne Glä', 'Obere Grenze des 95%-Prädiktionsintervalls der Anzahl Neuerkrankungen (ohne Glät', 'Untere Grenze des 95%-Prädiktionsintervalls der Anzahl Neuerkrankungen', 'Obere Grenze des 95%-Prädiktionsintervalls der Anzahl Neuerkrankungen', 'Untere Grenze des 95%-Prädiktionsintervalls der Reproduktionszahl R', 'Obere Grenze des 95%-Prädiktionsintervalls der Reproduktionszahl R', 'Untere Grenze des 95%-Prädiktionsintervalls des 7-Tage-R Wertes', 'Obere Grenze des 95%-Prädiktionsintervalls des 7-Tage-R Wertes'])

# Wetterdaten und Reproduktionstabelle zusammenfügen 

rRateWeatherG= pd.merge(rRate, weatherGermany, how='inner', on='Datum')

rRateWeatherM = pd.merge(rRate, weatherMuenchen, how='inner', on='Datum')

rRateWeatherD = pd.merge(rRate, weatherDresden, how='inner', on='Datum')

rRateWeatherDu = pd.merge(rRate, weatherDuesseldorf, how='inner', on='Datum')

rRateWeatherH = pd.merge(rRate, weatherHamburg, how='inner', on='Datum')
rRateWeatherG
rRateWeatherM

# Daten nach Luftfeutigkeit gefiltert

periodHumidityG70 = rRateWeatherG[rRateWeatherG['Humidity pct'] < 70.0]

periodHumidityG60 = rRateWeatherG[rRateWeatherG['Humidity pct'] < 60.0]

periodHumidityG50 = rRateWeatherG[rRateWeatherG['Humidity pct'] < 50.0]

periodHumidityG47 = rRateWeatherG[rRateWeatherG['Humidity pct'] < 47.0]

# 1 --> korreliert

# 0 --> kein Zusammenhang

#-1 --> korriliert entegengesetzt
corrG70 =periodHumidityG70.corr()

corrG70
corrG70.plot(kind='scatter', x='Humidity pct',y='Punktschätzer der Reproduktionszahl R', c='blue', title='Streuung der Korrelation - Luftfeuchtigkeit unter 70%').axis([-1,1,-1,1])

corrG60= periodHumidityG60.corr()

corrG60
corrG60.plot(kind='scatter', x='Humidity pct',y='Punktschätzer der Reproduktionszahl R', c='blue', title='Streuung der Korrelation - Luftfeuchtigkeit unter 60%').axis([-1,1,-1,1])



corrG50=periodHumidityG50.corr()

corrG50
corrG50.plot.scatter(x='Humidity pct',y='Punktschätzer der Reproduktionszahl R', c='blue', title='Streuung der Korrelation - Luftfeuchtigkeit unter 50%').axis([-1,1,-1,1])

# zu wenig Werte für 40% Luftfeuchtigkeit deshalb kleinster Wert 47% Luftfeuchtigkeit 

# Tendenz je geringer Luftfeuchtigkeit umso höher die Korrleation 

corrG47 = periodHumidityG47.corr()

corrG47
corrG47.plot.scatter(x='Humidity pct',y='Punktschätzer der Reproduktionszahl R', c='blue', title='Streuung der Korrelation - Luftfeuchtigkeit unter 47%').axis([-1,1,-1,1])
#Darstellung der Korrelation in einer Heatmap (Luftfeuchtigkeit < 47%)

mask = np.zeros_like(corrG47)

mask[np.triu_indices_from(mask)] = True

heatmapG47 = sns.heatmap(corrG47, mask = mask, annot=True, cmap="YlGnBu")

heatmapG47.set_title('Korrelation - Luftfeuchtigkeit unter 47%')
# keine aussagekräftige Darstellung

# periodHumidityG70['Punktschätzer der Reproduktionszahl R'].plot(linestyle='', marker='o')

# periodHumidityG60['Punktschätzer der Reproduktionszahl R'].plot(linestyle='', marker='o')

# periodHumidityG50['Punktschätzer der Reproduktionszahl R'].plot(linestyle='', marker='o')

# periodHumidityG47['Punktschätzer der Reproduktionszahl R'].plot(linestyle='', marker='o')

# keine aussagekräftige Darstellung

# fx = px.scatter(corrG70, x='Humidity pct',y='Punktschätzer der Reproduktionszahl R',color='Humidity pct')

rRateWeatherG.corr()

# Temperaturdaten auf Deutschland bezogen und nicht auf einzelne Städte

periodTemperaturAvgG3 = rRateWeatherG[rRateWeatherG['Temperatur Avg'] < 3]

periodTemperaturAvgG5 = rRateWeatherG[rRateWeatherG['Temperatur Avg'] < 5]

periodTemperaturAvgG7 = rRateWeatherG[rRateWeatherG['Temperatur Avg'] < 7]

periodTemperaturAvgG10 = rRateWeatherG[rRateWeatherG['Temperatur Avg'] < 10]

# periodTemperaturAvgM = rRateWeatherM[rRateWeatherM['Temperatur Avg'] < 5]

# periodTemperaturAvgD = rRateWeatherD[rRateWeatherD['Temperatur Avg'] < 5]

# periodTemperaturAvgDu= rRateWeatherDu[rRateWeatherDu['Temperatur Avg'] < 5]

# Es besteht eine Kausalität zwischen der Temperatur und dem Luftdruck, 

# je geringer die Temperatur umso höher die Korrelation zwischen den Neuerkrankten (ohne Glättung) und dem Luftdruck
temperaturG3 = periodTemperaturAvgG3.corr()

temperaturG3
temperaturG3.plot.scatter(x='Temperatur Avg',y='Punktschätzer der Reproduktionszahl R', c='blue', title='Streuung der Korrelation - Temperatur unter 3°').grid(True)
# Bei einer Temperatur unter 5° kann eine leichte Korrelation mit der Reproduktionszahl festgestellt werden 

temperaturG5= periodTemperaturAvgG5.corr()

temperaturG5
temperaturG5.plot.scatter(x='Temperatur Avg',y='Punktschätzer der Reproduktionszahl R', c='blue', title='Streuung der Korrelation - Temperatur unter 5°').grid(True)
# je höher die Temperatur umso mehr nimmt die Korrelation ab, 

# dies untermauert die Aussage, dass Viren sich im kühlen wohl fühlen 
temperaturG7=periodTemperaturAvgG7.corr()

temperaturG7
temperaturG7.plot.scatter(x='Temperatur Avg',y='Punktschätzer der Reproduktionszahl R', c='blue', title='Streuung der Korrelation - Temperatur unter 7°').grid(True)
temperaturG10 =periodTemperaturAvgG10.corr()

temperaturG10
temperaturG7.plot.scatter(x='Temperatur Avg',y='Punktschätzer der Reproduktionszahl R', c='blue', title='Streuung der Korrelation - Temperatur unter 10°').grid(True)