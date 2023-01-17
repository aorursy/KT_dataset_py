# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import shapefile as shp



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import geopandas as gpd

import geoplot as gplt

import geoplot.crs as gcrs

from matplotlib.colors import LinearSegmentedColormap

# Any results you write to the current directory are saved as output.
# import shape files using geopandas

shp = gpd.read_file("../input/countries-shape-files/ne_10m_admin_0_countries.shp")

shp.head()
df=pd.read_csv('/kaggle/input/covid19/CovidWB.csv')

df.head(5)

df.corr()
df=pd.read_csv('/kaggle/input/daywisenew//CoronaNewCases.xlsx.csv')

df.head(5)
df.corr()
import matplotlib.pyplot as plt

plt.plot( df.index, 'India', data=df, marker='o', markerfacecolor='blue', markersize=6, color='skyblue', linewidth=4)

plt.plot( df.index, 'Italy', data=df, marker='', color='olive', linewidth=2)

plt.plot( df.index, 'Iran', data=df, marker='', color='olive', linewidth=2, linestyle='dashed')

plt.plot( df.index, 'Spain', data=df, marker='', color='blue', linewidth=2, linestyle='dashed')

plt.plot( df.index, 'UK', data=df, marker='', color='green', linewidth=2, linestyle='dashed')

plt.legend()
