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
import seaborn as sns

data = pd.read_csv("../input/london-bike-sharing-dataset/london_merged.csv")

data.head()
import matplotlib.pyplot as plt

data_sample = data.sample(1000)



p = sns.PairGrid(data=data_sample, vars=['t1', 't2', 'hum', 'wind_speed', 'weather_code', 'is_holiday', 'is_weekend','season', 'cnt'])

p.map_diag(plt.hist)

p.map_offdiag(plt.scatter)

#La distribucion de la temperatura, nos da luces sobre como la temperatura de 10 a 20 grados, existe la mayor cantidad de alquileres, cuando la temperatura desciende mucho, de igual manera los alquileres. Por otro lado, si la temperatura aumenta demasiado, igualmente el alquiler cae.
sns.distplot(data["t1"],  color="black")

df=pd.read_csv('..//input/london-bike-sharing-dataset/london_merged.csv',parse_dates=True)

df.plot(x='timestamp',y='season', rot=85,c="red",title='Tiempo de uso de la bicilcleta con respecto a las temperatura')







plt.figure(figsize=[20,10])

plt.title('Correlación entre velocidad del viento y numero de alquileres')

sns.scatterplot(data['wind_speed'],data['cnt'])

plt.figure(figsize=[20,10])

plt.title('Correlación entre humedad y numeros de alquileres')

sns.scatterplot(data['hum'],data['cnt'] )
import matplotlib.pyplot as plt

f, axes = plt.subplots(1,2, figsize=(15, 5))

sns.barplot(x=data['is_holiday'], y=data['cnt'],ax=axes[0], palette="Set1")

sns.barplot(x=data["is_weekend"], y=data['cnt'],ax=axes[1] ,palette="Set1") 
