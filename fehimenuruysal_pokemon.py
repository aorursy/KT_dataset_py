# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # visualization

import seaborn as sns #visualization tool



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/pokemon-challenge/pokemon.csv")
data.info()
data.corr()
f,ax = plt.subplots(figsize=(18,18))

sns.heatmap(data.corr(), annot=True , linewidths = .5 , fmt= '.1f', ax=ax )

plt.show()
data.head(10)
data.columns
series = data["Defense"]

print(type(series))



data_frame = data[["Defense"]]

print(type(data_frame))
filtre = data['Defense']>200 #Dataset içerisinde bulunan pokemonlardan "Defense" özelliği 200 değerinden büyük olanlar koşulu filtreye atanır.

data[filtre] #Datasete filtreleme yapılır.
data[np.logical_and(data["Defense"]>200,data["Attack"]>100)] #Numpy kütüphanesine ait logical_and fonksiyonu ile Defense değeri 200'den büyük ve Attack değeri 130'dan büyük dataları filtreleme
filtre_2 = ((data["Defense"]>200) & (data["Attack"]>100))

data[filtre_2]
for index ,attack in data[["Attack"]][100:106].iterrows():

    print(index , " : " , attack)
data.Speed.plot(kind="line", color="green", label="Speed", linewidth = 1 , alpha=0.5, grid =True ,linestyle=":")

data.Defense.plot(kind="line" , color="blue" , label="Defense" , linewidth=1 , alpha = 0.5 , grid = True , linestyle="-.")

plt.legend(loc="best")

plt.xlabel("x axis")

plt.ylabel("y axis")

plt.title("Line Plot")

plt.show()
data.plot(kind="scatter" , x="Attack" , y="Defense" , alpha="0.5" , color="red")

plt.title("Scatter Plot")

plt.show()
plt.scatter(data.Attack , data.Defense , alpha=0.5)

plt.title("Scatter Plot")

plt.show()
data.Speed.plot(kind="hist",bins=50,figsize=(10,10),grid=True)

plt.xlabel("Speed")

plt.title("Histogram Plot")

plt.show()