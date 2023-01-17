# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import pandas as pd



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

!ls
pokemonData = pd.read_csv("../input/pokemon.csv")
pokemonData.info()
pokemonData.head(10)
pokemonData.corr()
#correlation Of Data Feature

f,ax = plt.subplots(figsize=(18,18))

sns.heatmap(pokemonData.corr(),annot=True,linewidths=.5,fmt = '.1f',ax=ax)

#for hide above text

plt.show()
pokemonData.columns
pokemonData.Speed.plot(kind='line',color='y',label='Speed',linewidth=1, alpha=1, grid=True,linestyle='-.')

pokemonData.Defense.plot(color='r',label='Defense',linewidth=1, alpha=1, grid=True,linestyle='-.')

plt.legend(loc='upper right')

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('Line plot')
pokemonData.plot(kind='scatter',x='Attack',y='Defense',alpha = 0.5, color = 'green')

plt.xlabel('Attack')

plt.ylabel('Defence')

plt.title("Attack and Defense Correlation Scatter Plot")
pokemonData.Defense.plot(kind='hist',bins=50,figsize=(20,10))

#plt.clf()
defence = pokemonData['Attack']>150

pokemonData[defence]
pokemonData[np.logical_and(pokemonData['Defense']>200 , pokemonData['Attack']>50)]
for index,value in pokemonData[['Attack']][0:1].iterrows():

    print(index," : ", value)
avgForSpeed = sum(pokemonData.Speed)/len(pokemonData.Speed) 

pokemonData["Speed_Levels"] = ["high" if i > avgForSpeed else "low" for i in pokemonData.Speed]

pokemonData.loc[:15,["Speed","Speed_Levels"]]
pokemonData.shape
print(pokemonData["Type 2"].value_counts(dropna=False))
pokemonData.boxplot(column='Attack',by='Legendary')

plt.show()
pokemonDataNew = pokemonData.head()

pokemonDataNew
pokemonDataMelted = pd.melt(frame = pokemonDataNew,id_vars = 'Name', value_vars=['Speed','Defense'])

pokemonDataMelted
pokemonDataMelted.pivot(index='Name',columns='variable',values='value')
headData = pokemonData.head()

tailData = pokemonData.tail()

conc_data = pd.concat([headData,tailData],axis=0,ignore_index=True)

conc_data
attack = pokemonData["Attack"].head()

speed = pokemonData["Speed"].head()

conc_data_axis1 = pd.concat([attack,speed],axis=1)

conc_data_axis1
pokemonData["Type 1"] = pokemonData["Type 1"].astype('category')

pokemonData.dtypes
pokemonData.info()
pokemonData["Type 2"].value_counts(dropna=False)
#drop non values

pokemonData["Type 2"].dropna(inplace=True)

#pokemonData["Type 2"].fillna("empty",inplace=True)
assert pokemonData["Type 2"].notnull().all() # is true all datas were  fill with values
data1 = pokemonData.loc[:,["Attack","Defense","Generation"]]

data1.plot()
data1.plot(subplots= True)

plt.show()
fig, axes = plt.subplots(nrows=2,ncols=1)

data1.plot(kind="hist",y="Defense",bins=45,range=(0,250),normed=True,ax=axes[0])

data1.plot(kind="hist",y="Defense",bins=45,range=(0,250),normed=True,ax=axes[1],cumulative=True)

plt.savefig("graph.png")

plt.show()
import warnings

warnings.filterwarnings("ignore")



data2 = pokemonData.head()

date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]

dateTimeObject = pd.to_datetime(date_list)

data2["date"] = dateTimeObject

data2 = data2.set_index("date")

data2
data2.resample("A").mean()
data2.resample("M").mean()
data2.resample("A").first().interpolate("linear")
data2.resample("M").first().interpolate("linear")
datax = pokemonData.set_index("#")

datax.head()
datax.Attack[1]
datax.loc[1,["Attack"]]
datax[["Attack","Speed"]]
print(type(datax["Attack"])) #series

print(type(datax[["Defense"]])) #data frames
pokemonData.loc[1:10,["Attack","Speed"]]
pokemonData.loc[10:1:-1,["Attack","Speed"]]
pokemonData.loc[1:10,"Speed":]
booleanDataFrame = pokemonData.HP > 200

pokemonData[booleanDataFrame]
f1 = pokemonData.HP > 150

f2 = pokemonData.Speed > 25

pokemonData[f1 & f2]
pokemonData.Attack[pokemonData.Speed<15]
pokemonData.HP.apply(lambda x : x*2)
pokemonData["total_power"] = pokemonData.Attack + pokemonData.Defense

pokemonData.head()
dataCopied = pokemonData.copy()

dataCopied.index = range(100,900,1)

dataCopied.head()
dataCopied2 = dataCopied.copy()

dataCopied2 = pokemonData.set_index(["Type 1","Type 2"])

dataCopied2.head(100)