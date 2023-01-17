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
from scipy.stats import pearsonr

bw = pd.read_csv('/kaggle/input/wetterbw/Landkreise und Wetter BW.csv', encoding ='latin1', sep=";")
bwA = pd.read_csv('/kaggle/input/bwabsolut/bwAbsolut.csv', encoding ='latin1', sep=";")

#bwA
bw
bw=bw.drop(columns='percipitation avg Woche')
#bwAcorr = bwA.corr()
#bwAcorr['Änderung ']
bwcorr = bw.corr()
bwcorr['Änderung in % zu Vorwoche']
pearsonr(bw['Änderung in % zu Vorwoche'], bw['temp avg Woche'])
# Daten nach Luftfeuchtigkeit gefiltert
#humidity70bwA = bwA[bwA['humidity avg Woche'] < 70] 
#humidity60bwA = bwA[bwA['humidity avg Woche'] < 60]
#humidity55bwA = bwA[bwA['humidity avg Woche'] < 55]
#humidity70bwA = humidity70bwA.corr()
#humidity70bwA['Änderung ']
#humidity60bwA = humidity60bwA.corr()
#humidity60bwA['Änderung ']
#humidity55bwA = humidity55bwA.corr()
#humidity55bwA['Änderung ']
# Daten nach Luftfeuchtigkeit gefiltert
humidity70bw = bw[bw['humidity avg Woche'] < 70] 
humidity60bw = bw[bw['humidity avg Woche'] < 60]
humidity55bw = bw[bw['humidity avg Woche'] < 55]
#je niedriger die Luftfeuchtigkeit umso höher die negative Korrelation mit der Temperatur und den Neuinfektionen
# negative Korrelation besagt in unserem Fall, je niedriger die Temperatur umso höher die Neuinfektionen 
humidity70bw
pearsonr(humidity70bw['Änderung in % zu Vorwoche'],humidity70bw['temp avg Woche'])
humidity70bw = humidity70bw.corr()
humidity70bw['Änderung in % zu Vorwoche']
pearsonr(humidity60bw['Änderung in % zu Vorwoche'], humidity60bw['temp avg Woche'])
humidity60bw = humidity60bw.corr()
humidity60bw['Änderung in % zu Vorwoche']
pearsonr(humidity55bw['Änderung in % zu Vorwoche'], humidity55bw['temp avg Woche'])
humidity55bw = humidity55bw.corr()
humidity55bw['Änderung in % zu Vorwoche']
humidity70bw.plot(kind='scatter', x='temp avg Woche',y='Änderung in % zu Vorwoche', c='blue', title='Streuung der Korrelation - Luftfeuchtigkeit unter 70%').axis([-1,1,-1,1])
humidity60bw.plot(kind='scatter', x='temp avg Woche',y='Änderung in % zu Vorwoche', c='blue', title='Streuung der Korrelation - Luftfeuchtigkeit unter 70%').axis([-1,1,-1,1])
humidity55bw.plot(kind='scatter', x='temp avg Woche',y='Änderung in % zu Vorwoche', c='blue', title='Streuung der Korrelation - Luftfeuchtigkeit unter 70%').axis([-1,1,-1,1])
# Daten nach Temperatur gefiltert
#temp15bwA = bwA[bwA['temp avg Woche'] < 15.0] 
#temp10bwA = bwA[bwA['temp avg Woche'] < 10.0]
#temp5bwA = bwA[bwA['temp avg Woche'] < 5.0]
#temp15bwA = temp15bwA.corr()
#temp15bwA['Änderung ']
#temp10bwA = temp10bwA.corr()
#temp10bwA['Änderung ']
#temp5bwA = temp5bwA.corr()
#temp5bwA['Änderung ']
# Daten nach Temperatur gefiltert
temp15bw = bw[bw['temp avg Woche'] < 15.0] 
temp10bw = bw[bw['temp avg Woche'] < 10.0]
temp5bw = bw[bw['temp avg Woche'] < 5.0]

# 1 --> korreliert
# 0 --> kein Zusammenhang
#-1 --> korriliert entegengesetzt
temp15bw = temp15bw.corr()
temp15bw['Änderung in % zu Vorwoche']
temp10bw = temp10bw.corr()
temp10bw['Änderung in % zu Vorwoche']
temp5bw = temp5bw.corr()
temp5bw['Änderung in % zu Vorwoche']
temp5bw.info()
temp15bw.plot(kind='scatter', x='humidity avg Woche',y='Änderung in % zu Vorwoche', c='blue', title='Streuung der Korrelation - Temperatur unter 15°').axis([-1,1,-1,1])

temp10bw.plot(kind='scatter', x='humidity avg Woche',y='Änderung in % zu Vorwoche', c='blue', title='Streuung der Korrelation - Temperatur unter 10°').axis([-1,1,-1,1])
temp5bw.plot(kind='scatter', x='humidity avg Woche',y='Änderung in % zu Vorwoche', c='blue', title='Streuung der Korrelation - Temperatur unter 5°').axis([-1,1,-1,1])
#tempbw = bw[bw['KW']== 10 | 11 | 12 |13|14]
#tempbw

#temp15bw = tempbw[tempbw['temp avg Woche'] < 15.0] 
#temp10bw = tempbw[tempbw['temp avg Woche'] < 10.0] 
#temp5bw = tempbw[tempbw['temp avg Woche'] < 5.0] 
#temp15bw.corr()
#tempbw2 = bw[bw['Landkreis']== 'Sigmaringen']
#tempbw2 = bw[bw['Landkreis']== 'Baden-Baden']
#tempbw2.corr()