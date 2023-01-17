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

nrw = pd.read_csv('/kaggle/input/landkreise123/NRW.csv', encoding ='latin1', sep=";")
nrw
nrwcorr = nrw.corr()

nrwcorr['Änderung in % zur Vorwoche']
# Daten nach Luftfeutigkeit gefiltert

hum70nrw = nrw[nrw['humidity avg Woche'] < 70.0]

hum60nrw = nrw[nrw['humidity avg Woche'] < 60.0]

hum55nrw = nrw[nrw['humidity avg Woche'] < 55.0]

hum70nrw.info()
hum60nrw.info()
hum55nrw.info()
hum70nrwcorr = hum70nrw.corr()

hum70nrwcorr['Änderung in % zur Vorwoche']
hum60nrwcorr = hum60nrw.corr()

hum60nrwcorr['Änderung in % zur Vorwoche']
hum55nrwcorr = hum55nrw.corr()

hum55nrwcorr['Änderung in % zur Vorwoche']
hum70nrwcorr.plot(kind='scatter', x='temp avg Woche',y='Änderung in % zur Vorwoche', c='blue', title='Streuung der Korrelation - Luftfeuchtigkeit unter 70%').axis([-1,1,-1,1])
hum60nrwcorr.plot(kind='scatter', x='temp avg Woche',y='Änderung in % zur Vorwoche', c='blue', title='Streuung der Korrelation - Luftfeuchtigkeit unter 60%').axis([-1,1,-1,1])
hum55nrwcorr.plot(kind='scatter', x='temp avg Woche',y='Änderung in % zur Vorwoche', c='blue', title='Streuung der Korrelation - Luftfeuchtigkeit unter 55%').axis([-1,1,-1,1])
#Gefiltert nach Temperatur
temp7nrw = nrw[nrw['temp avg Woche'] < 7]

temp10nrw = nrw[nrw['temp avg Woche'] < 10]

temp15nrw = nrw[nrw['temp avg Woche'] < 15]
temp7nrw.info()
temp7nrwcorr = temp7nrw.corr()

temp7nrwcorr['Änderung in % zur Vorwoche']
temp10nrw.info()
temp10nrwcorr = temp10nrw.corr()

temp10nrwcorr['Änderung in % zur Vorwoche']
temp15nrw.info()
temp15nrwcorr = temp15nrw.corr()

temp15nrwcorr['Änderung in % zur Vorwoche']
temp7nrw.plot(kind='scatter', x='humidity avg Woche',y='Änderung in % zur Vorwoche', c='blue', title='Streuung der Korrelation - Temperatur unter 7 Grad').axis([0,100,-1000,1000])
temp10nrw.plot(kind='scatter', x='humidity avg Woche',y='Änderung in % zur Vorwoche', c='blue', title='Streuung der Korrelation - Temperatur unter 10 Grad').axis([0,100,-1000,1000])
temp15nrw.plot(kind='scatter', x='humidity avg Woche',y='Änderung in % zur Vorwoche', c='blue', title='Streuung der Korrelation - Temperatur unter 15 Grad').axis([0,100,-1000,1000])