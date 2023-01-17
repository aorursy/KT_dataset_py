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
##Sachsen = pd.read_csv('/kaggle/input/Sachsen_final2.csv',encoding='latin1', sep=";")
##Sachsen1 = pd.read_csv('/kaggle/input/LandkreiseWetterSachsen.csv',encoding='latin1', sep=";")
sh = pd.read_csv('/kaggle/input/sachsenfinal/SachsenWetterFinal.csv',encoding='latin1', sep=";")
sh
shcorr = sh.corr()
shcorr['Ãnderung in % zur Vorwoche']
shcorr
hum70 = sh[sh['humidity avg Woche'] < 70.0]
hum60 = sh[sh['humidity avg Woche'] < 60.0]
hum50 = sh[sh['humidity avg Woche'] < 50.0]
hum55 = sh[sh['humidity avg Woche'] < 55.0]
hum70corr = hum70.corr()
hum70corr['Ãnderung in % zur Vorwoche']
hum60corr = hum60.corr()
hum60corr['Ãnderung in % zur Vorwoche']
hum50corr = hum50.corr()
hum50corr['Ãnderung in % zur Vorwoche']
hum55.info()
hum55corr = hum55.corr()
hum55corr['Ãnderung in % zur Vorwoche']
hum55
hum70corr.plot(kind='scatter', x='temp avg Woche',y='Ãnderung in % zur Vorwoche', c='blue', title='Streuung der Korrelation - Luftfeuchtigkeit unter 70%').axis([-1,1,-1,1])
hum60corr.plot(kind='scatter', x='temp avg Woche',y='Ãnderung in % zur Vorwoche', c='blue', title='Streuung der Korrelation - Luftfeuchtigkeit unter 60%').axis([-1,1,-1,1])
hum55corr.plot(kind='scatter', x='temp avg Woche',y='Ãnderung in % zur Vorwoche', c='blue', title='Streuung der Korrelation - Luftfeuchtigkeit unter 55%').axis([-1,1,-1,1])
hum50corr.plot(kind='scatter', x='temp avg Woche',y='Ãnderung in % zur Vorwoche', c='blue', title='Streuung der Korrelation - Luftfeuchtigkeit unter 50%').axis([-1,1,-1,1])
#Darstellung der Korrelation in einer Heatmap (Luftfeuchtigkeit < 47%)
mask = np.zeros_like(hum55corr)
mask[np.triu_indices_from(mask)] = True
heatmapG55 = sns.heatmap(hum55corr, mask = mask, annot=True, cmap="YlGnBu")
heatmapG55.set_title('Korrelation - Luftfeuchtigkeit unter 55%')
temp12 = sh[sh['temp avg Woche'] < 12]
temp10 = sh[sh['temp avg Woche'] < 10]
temp7 = sh[sh['temp avg Woche'] < 7]
temp12corr = temp12.corr()
temp12corr['Ãnderung in % zur Vorwoche']
temp10corr = temp10.corr()
temp10corr['Ãnderung in % zur Vorwoche']
temp7corr = temp7.corr()
temp7corr['Ãnderung in % zur Vorwoche']
temp12.plot.scatter(x='temp avg Woche',y='Ãnderung in % zur Vorwoche', c='blue', title='Streuung der Korrelation - Temperatur unter 12°').grid(True)
temp10.plot.scatter(x='temp avg Woche',y='Ãnderung in % zur Vorwoche', c='blue', title='Streuung der Korrelation - Temperatur unter 10°').grid(True)
temp7.plot.scatter(x='temp avg Woche',y='Ãnderung in % zur Vorwoche', c='blue', title='Streuung der Korrelation - Temperatur unter 7°').grid(True)
temp15 = sh[sh['temp avg Woche'] < 15]
temp10 = sh[sh['temp avg Woche'] < 10]
temp5 = sh[sh['temp avg Woche'] < 5]
temp15corr = temp15.corr()
temp15corr['Ãnderung in % zur Vorwoche']
temp15.plot.scatter(x='temp avg Woche',y='Ãnderung in % zur Vorwoche', c='blue', title='Streuung der Korrelation - Temperatur unter 15°').grid(True)
temp10corr = temp10.corr()
temp10corr['Ãnderung in % zur Vorwoche']
temp10.plot.scatter(x='temp avg Woche',y='Ãnderung in % zur Vorwoche', c='blue', title='Streuung der Korrelation - Temperatur unter 10°').grid(True)
temp5corr = temp5.corr()
temp5corr['Ãnderung in % zur Vorwoche']
temp5.plot.scatter(x='temp avg Woche',y='Ãnderung in % zur Vorwoche', c='blue', title='Streuung der Korrelation - Temperatur unter 5°').grid(True)