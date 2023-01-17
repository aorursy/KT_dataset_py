# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import requests #import requests library
import json,os,datetime
import csv
from pandas import DataFrame #reading data as tables
from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sns #for beautiful plots
%matplotlib inline
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
nasa_data = pd.read_csv("../input/nasa.csv")
nasa_data.head()
nasa_data.isnull().head()
for col in nasa_data.columns:
    print('Length of column'+" "+col+"=",len(nasa_data[col]))
nasa_data.info()

# we can usually interpolate the missing values, if there are any
nasa_data.interpolate()
nasa_data.sample(1000).head() #random sample of 1000 rows
nasa_data.describe() # description of data
# Make plots larger
plt.rcParams['figure.figsize'] = (20, 9)
sns.distplot(nasa_data['Absolute Magnitude'])
plt.show()
sns.distplot(nasa_data['Est Dia in KM(min)'])
plt.show()
sns.distplot(nasa_data['Est Dia in KM(max)'])
plt.show()
sns.distplot(nasa_data['Est Dia in M(min)'])
plt.show()
sns.distplot(nasa_data['Est Dia in M(max)'])
plt.show()
sns.distplot(nasa_data['Est Dia in Miles(min)'])
plt.show()
sns.distplot(nasa_data['Est Dia in Miles(max)'])
plt.show()
sns.distplot(nasa_data['Est Dia in Miles(max)'])
plt.show()
sns.distplot(nasa_data['Est Dia in Feet(min)'])
plt.show()
sns.distplot(nasa_data['Est Dia in Feet(max)'])
plt.show()
sns.distplot(nasa_data['Relative Velocity km per sec'])
plt.show()
sns.distplot(nasa_data['Relative Velocity km per hr'])
plt.show()
sns.distplot(nasa_data['Miles per hour'])
plt.show()
sns.distplot(nasa_data['Miss Dist.(Astronomical)'])
plt.show()
sns.distplot(nasa_data['Orbit Uncertainity'])
plt.show()
sns.distplot(nasa_data['Minimum Orbit Intersection'])
plt.show()
sns.distplot(nasa_data['Jupiter Tisserand Invariant'])
plt.show()
sns.distplot(nasa_data['Epoch Osculation'])
plt.show()
sns.distplot(nasa_data['Eccentricity'])
plt.show()
sns.distplot(nasa_data['Semi Major Axis'])
plt.show()
sns.distplot(nasa_data['Inclination'])
plt.show()
sns.distplot(nasa_data['Asc Node Longitude'])
plt.show()
sns.distplot(nasa_data['Orbital Period'])
plt.show()

sns.distplot(nasa_data['Perihelion Distance'])
plt.show()
sns.distplot(nasa_data['Aphelion Dist'])
plt.show()
sns.distplot(nasa_data['Perihelion Distance'])
plt.show()
nasa_data.sort_values(by="Close Approach Date", inplace=True)
nasa_data['Close Approach Date'] = pd.to_datetime(nasa_data["Close Approach Date"], format='%Y-%m-%d')

name_count=nasa_data.groupby(['Close Approach Date'],as_index=False).count()

plt.plot(name_count['Close Approach Date'],name_count['Name'])
plt.xlabel("Close Approach Date",fontweight="bold")
plt.ylabel("Count of Asteroids over time",fontweight="bold")
plt.title("Trend of Number of Asteroids",fontweight="bold")
plt.xticks(rotation=30)
mean = name_count['Name'].mean()
plt.axhline(mean, color='r', linestyle='--',label='Mean')
plt.legend()
plt.show()
print()
dfDummy=nasa_data[['Name','Hazardous']]
sns.countplot(x="Hazardous", data=nasa_data,palette="Greens_d")
plt.show()
cmap = sns.diverging_palette(0, 255, sep=1, n=256, as_cmap=True)
correlations = nasa_data[['Neo Reference ID', 'Name', 'Absolute Magnitude', 'Est Dia in KM(min)',
       'Est Dia in KM(max)', 'Est Dia in M(min)', 'Est Dia in M(max)',
       'Est Dia in Miles(min)', 'Est Dia in Miles(max)',
       'Est Dia in Feet(min)', 'Est Dia in Feet(max)', 'Close Approach Date',
       'Epoch Date Close Approach', 'Relative Velocity km per sec',
       'Relative Velocity km per hr', 'Miles per hour',
       'Miss Dist.(Astronomical)', 'Miss Dist.(lunar)',
       'Miss Dist.(kilometers)', 'Miss Dist.(miles)', 'Orbiting Body',
       'Orbit ID', 'Orbit Determination Date', 'Orbit Uncertainity',
       'Minimum Orbit Intersection', 'Jupiter Tisserand Invariant',
       'Epoch Osculation', 'Eccentricity', 'Semi Major Axis', 'Inclination',
       'Asc Node Longitude', 'Orbital Period', 'Perihelion Distance',
       'Perihelion Arg', 'Aphelion Dist', 'Perihelion Time', 'Mean Anomaly',
       'Mean Motion', 'Equinox', 'Hazardous']].corr() 
sns.heatmap(correlations, cmap=cmap)
plt.show()
