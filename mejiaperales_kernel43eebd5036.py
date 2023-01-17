# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas import ExcelWriter
from pandas import ExcelFile

import matplotlib.pyplot as plt
from scipy import stats

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Create a Pandas Excel writer using XlsxWriter as the engine.
df = pd.read_excel('/kaggle/input/DiversidadAlphaVegetacion.xlsx',index_col='Item')
#Disable warnings
import warnings
warnings.filterwarnings('ignore')

#Matplotlib forms basis for visalization in Python
import matplotlib.pyplot as plt

#We will use the Seaborn library
import seaborn as sns
sns.set()

#Graphics in SVG format are more sharp and legible
%config InlineBackend.figure_format = 'svg'

#Increase the default plot size and set the color scheme
plt.rcParams['figure.figsize'] = 8, 5
plt.rcParams['image.cmap'] = 'viridis'

def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""

    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n

    return x, y

#Check the first dataframe rows
df
#Rows and columns
df.shape
#Info about data
df.info()
#Distribucion de especies por area
especieArea=pd.crosstab(index=df.Especie, columns=df.Area,margins=True, margins_name="Total").sort_values('Total',ascending=False)

especieArea
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(15,15))

sns.heatmap(especieArea.corr(), annot=True, linewidths=.5, ax=ax)
#Correlacion entre especies
especievsespecie=pd.crosstab(index=df.Area, columns=df.Especie,margins=True, margins_name="Total").sort_values('Total',ascending=False)


import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(15,15))
sns.heatmap(especievsespecie.corr(), annot=True, linewidths=.5, ax=ax)
especieArea.plot(figsize=(10, 10))
plt.title(especieArea.columns[0])
especieArea[especieArea.columns[0]].plot(figsize=(8, 6))
plt.show()
plt.title(especieArea.columns[1])
especieArea[especieArea.columns[1]].plot(figsize=(8, 6))
plt.show()
plt.title(especieArea.columns[2])
especieArea[especieArea.columns[2]].plot(figsize=(8, 6))
plt.show()
plt.title(especieArea.columns[3])
especieArea[especieArea.columns[3]].plot(figsize=(8, 6))
plt.show()
plt.title(especieArea.columns[4])
especieArea[especieArea.columns[4]].plot(figsize=(8, 6))
plt.show()
plt.title(especieArea.columns[5])
especieArea[especieArea.columns[5]].plot(figsize=(8, 6))
plt.show()
plt.title(especieArea.columns[6])
especieArea[especieArea.columns[6]].plot(figsize=(8, 6))
plt.show()
plt.title(especieArea.columns[7])
especieArea[especieArea.columns[7]].plot(figsize=(8, 6))
plt.show()
sns.heatmap(pd.crosstab(df.Especie, df.Area),
            cmap="YlGnBu", annot=True, cbar=False)
sns.distplot(especieArea.MIIEpa, fit=stats.norm, kde=False)
print(stats.normaltest(especieArea.MIIEpa))
sns.distplot(especieArea.MIIEpa_Borde, fit=stats.norm, kde=False)
print(stats.normaltest(especieArea.MIIEpa_Borde))
sns.distplot(especieArea.MIICempa, fit=stats.norm, kde=False)
print(stats.normaltest(especieArea.MIICempa_Borde))
sns.distplot(especieArea.MIICempa_Borde, fit=stats.norm, kde=False)
print(stats.normaltest(especieArea.MIICempa_Borde))
sns.distplot(especieArea.MIICrot, fit=stats.norm, kde=False)
print(stats.normaltest(especieArea.MIICrot))
sns.distplot(especieArea.MIICrot_Borde, fit=stats.norm, kde=False)
print(stats.normaltest(especieArea.MIICrot_Borde))
sns.distplot(especieArea.Monocultivo, fit=stats.norm, kde=False)
print(stats.normaltest(especieArea.Monocultivo))
sns.distplot(especieArea.Monocultivo_Borde, fit=stats.norm, kde=False)
print(stats.normaltest(especieArea.Monocultivo_Borde))
#we want to understand the percentage of time each combination occurs (Especie and Area)
#Example 2.04% of total population are Sida Neomexicana on MIIEpa_Borde and 3.4% of population are Spananthe paniculata on MIIEpa
especieAreaNrm=pd.crosstab(df.Especie, df.Area,normalize=True)
especieAreaNrm
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(pd.crosstab(df.Area, df.Especie,normalize=True).round(2),
            cmap="YlGnBu", annot=True, cbar=False, linewidths=.5, ax=ax)
##How the Especies are distributed across Areas
#example This table shows us that 75% of the Amaranthus palmeri (quintonil) are on MIIEpa area and the other  25% is on MIICempa_Borde area 
pd.crosstab(df.Especie, df.Area, normalize='index').round(2)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(pd.crosstab(df.Especie, df.Area,normalize='index').round(2),
            cmap="YlGnBu", annot=True, cbar=False, linewidths=.5, ax=ax)
##how each area is composed
#example This table shows us that MIICempa is compose 10% by Cenchrus cilaris, 3% Chenopodium album, 14% Commelina (4spp.), etc...
pd.crosstab(df.Especie, df.Area, normalize='columns').round(2)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(pd.crosstab(df.Especie, df.Area,normalize='columns').round(2),
            cmap="YlGnBu", annot=True, cbar=False, linewidths=.5, ax=ax)
