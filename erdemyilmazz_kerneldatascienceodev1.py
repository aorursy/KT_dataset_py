# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#string olarak ilk cell' in çıktısını kullanabliriz

data=pd.read_csv("/kaggle/input/videogamesales/vgsales.csv") 
data.columns
data.head(10)
corrData=data.corr()
corrData
type(corrData)
corrData.loc[["Rank", "NA_Sales"],["Rank", "NA_Sales"]]
f,ax = plt.subplots(figsize=(14,5))

sns.heatmap(corrData, annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
plt.plot(data.NA_Sales,data.Rank, color="r")

#plt.xlim(1979,2000)

#plt.ylim(0,18000)

plt.show()
data["NA_Sales"].plot(color = 'r',label = 'Defense',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.show()
#scatter plot çizdirme

plt.scatter(data.JP_Sales, data.Other_Sales, alpha=0.5, color="r")

plt.show()
# histogram plot

# x axis is speed

# y axis frequency yani adetler aslında mesela 150 hızındaki pokemon sayısı yaklaşık 5

data.Global_Sales.plot(kind="hist", bins=100, figsize=(10,10),color="g")

plt.show()

data.Global_Sales.plot(kind="hist", bins=100, figsize=(10,10),color="g")

#plt.xlim(data.Global_Sales.min(),data.Global_Sales.max())

plt.ylim(0,10)

plt.show()

for index,value in data[["Global_Sales"]][0:5].iterrows():

    print(index," : ",value)
i=0

for index,value in data[(data['Global_Sales']>0.57) & (data['Global_Sales']<0.6)].iterrows():

    print(index," : ",value[["Global_Sales"]])

    i=i+1

print(i," adet kayıt var")