# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df= pd.read_csv("../input/BRAZIL_CITIES.csv", sep=";", decimal=",")
df.head()
df.info()
df.fillna(0., inplace=True)
# Amount of states plus Federal District

df["STATE"].value_counts().shape
# States and cities per state

df["STATE"].value_counts()
# remove zero values

mask1= df["LONG"] != 0

mask2 = df["LAT"] !=0 



x = df[mask1&mask2]["LONG"]

y = df[mask1&mask2]["LAT"]

z = df[mask1&mask2]["IDHM"]

 

# use the scatter function

plt.figure(figsize=(10,10))

plt.title("Cities Latitude and Longitude")

plt.xlabel("Longitude")

plt.ylabel("Latitude")

plt.scatter(x, y, s=z, alpha=1)

plt.show()
# remove zero values

mask1= df["LONG"] != 0

mask2 = df["LAT"] !=0 



x = df[mask1&mask2]["LONG"]

y = df[mask1&mask2]["LAT"]

z = df[mask1&mask2]["ESTIMATED_POP"]

 

# use the scatter function

plt.figure(figsize=(10,10))

plt.title("Population per Latitude and Longitude")

plt.xlabel("Longitude")

plt.ylabel("Latitude")

plt.scatter(x, y, s=z/5000, alpha=1)

plt.show()
# Resident population 2010

df["IBGE_RES_POP"].sum()
# Resident population 2016

df["POP_GDP"].sum()
# Estimated population 2018

df["ESTIMATED_POP"].sum()
# remove zero values

mask1= df["GDP_CAPITA"] != 0

mask2 = df["IDHM"] !=0 

 

# create data

x = df[mask1&mask2]["GDP_CAPITA"]

y = df[mask1&mask2]["IDHM"]

z = df[mask1&mask2]["ESTIMATED_POP"]

 

# use the scatter function

plt.figure(figsize=(15, 8))

plt.title("HDI Human Development Index per Gross Domestic Product per capita per Estimated Population 2018")

plt.xlabel("Gross Domestic Product")

plt.ylabel("HDI Human Development Index")

plt.scatter(x, y, s=z/5000, alpha=0.5)

plt.show()
mask1= df["LONG"] != 0

mask2 = df["LAT"] !=0 

mask3 = df["CATEGORIA_TUR"] != 0.



sns.lmplot( x="LONG", y="LAT", data=df[mask1&mask2&mask3], 

           fit_reg=False, hue='CATEGORIA_TUR', legend=True, scatter_kws={"s": 30},

          height=10)
mask1= df["GDP_CAPITA"] != 0

mask2 = df["BEDS"] !=0 



sns.lmplot( x="GDP_CAPITA", y="BEDS", data=df[mask1&mask2], 

           fit_reg=False, hue='UBER', legend=True, scatter_kws={"s": 30},

          height=7)