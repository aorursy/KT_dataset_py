# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import geopandas as gpd

import geoplot

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
perico = "../input/cocaine-listings/dream_market_cocaine_listings.csv"



coca_data = pd.read_csv(perico)



coca_data.head()



list(coca_data.columns.values)
#estos son todos los vendedores que estan en el dataset

coca_data.vendor_name.unique()
#esta es una muestra de las veces que aparece cada vendedor en las publicaciones 

#en total son 180 vendedores y el que mas se anuncia sale 25 veces

vendedores = coca_data.pivot_table(index=['vendor_name'], aggfunc='size')

v=vendedores.sort_values(ascending=True)





vendedores.values

a=vendedores.sort_values(ascending=True)

a



# grafica de las veces que aparece cada vendedor en las publicaciones  



v.plot(figsize=(35,15))



plt.figure(figsize=(30,10))

plt.xticks(rotation='vertical')

plt.margins(0.5)

sns.barplot (x=v.index, y=a ) 

plt.figure(figsize=(35,10))

plt.xticks(rotation='vertical')

plt.margins(0.4)

sns.barplot (x=coca_data['vendor_name'], y=coca_data['quality'] ) 
coca_data[['quality']].mean()
plt.figure(figsize=(35,10))

plt.xticks(rotation='vertical')

plt.margins(0.4)

sns.barplot (x=coca_data['vendor_name'], y=coca_data['rating'] ) 
coca_data[['rating']].mean()
plt.figure(figsize=(15,10))

sns.scatterplot(x=coca_data['quality'], y=coca_data['cost_per_gram'],hue =coca_data['vendor_name']=="thebestchemist") # Your code here
