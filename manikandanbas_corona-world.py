# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Plotting using plotly

import plotly.express as px

import matplotlib.pyplot as plt

import matplotlib

import matplotlib.dates as mdates

import folium 

# import calmap

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('/kaggle/input/apr21-world-corona/Continent.csv')

data=df.copy()

data1=data.groupby('Continent')['Cases','Deaths','Recovered','Active'].max().reset_index()





labels = list(data['Continent'])

sizes = list(data['Cases'])

color= ['#66b3ff','green','red','purple','brown','blue']

explode = []



for i in labels:

    explode.append(0.05)

plt.figure(figsize= (15,10))

plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=9, explode =explode,colors = color)

plt.title('World COVID-19 Cases',fontsize = 30)

plt.axis('equal')  

plt.tight_layout()



labels = list(data['Continent'])

sizes = list(data['Active'])

color= ['#66b3ff','green','red','purple','brown','blue']

explode = []



for i in labels:

    explode.append(0.05)

plt.figure(figsize= (15,10))

plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=9, explode =explode,colors = color)

plt.title('World COVID-19 Active',fontsize = 30)

plt.axis('equal')  

plt.tight_layout()





labels = list(data['Continent'])

sizes = list(data['Recovered'])

color= ['#66b3ff','green','red','purple','brown','blue']

explode = []



for i in labels:

    explode.append(0.05)

plt.figure(figsize= (15,10))

plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=9, explode =explode,colors = color)

plt.title('World COVID-19 Recovered',fontsize = 30)

plt.axis('equal')  

plt.tight_layout()

labels = list(data['Continent'])

sizes = list(data['Deaths'])

color= ['#66b3ff','green','red','purple','brown','blue']

explode = []



for i in labels:

    explode.append(0.05)

plt.figure(figsize= (15,10))

plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=9, explode =explode,colors = color)

plt.title('World COVID-19 Deaths',fontsize = 30)

plt.axis('equal')  

plt.tight_layout()
