import pandas as pd # Datensets

import numpy as np # Data Manipulation

import os # File System

from IPython.display import Image

from IPython.core.display import HTML 

import matplotlib.pyplot as plt # Library for Plotting

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import seaborn as sns # Library for Plotting

sns.set # make plots look nicer

sns.set_palette("husl")

import warnings

warnings.filterwarnings('ignore')

# Plot inside Notebooks

%matplotlib inline 
data = pd.read_csv("../input/adultdata/adultdata_prep.csv")
data.head()
data.describe()
newData = data.dropna()

newData = newData.drop("Unnamed: 0", axis = 1)

newData = newData[(newData["workclass"] != '?') & (newData["education"] != '?') & (newData["marital.status"] != '?') & (newData["occupation"] != '?') 

                 & (newData["relationship"] != '?') & (newData["race"] != '?') & (newData["sex"] != '?') & (newData["native.country"] != '?') & (newData["income"] != '?')]



newData.head()
hoursPerWeek = newData.groupby("income").mean()

gender = newData.groupby("sex").mean()

gender.head(3)

ageData = newData.groupby("age").mean()

ageData.head(10)



#fig,ax = plt.subplots(figsize=(25,10))

#ax = sns.scatterplot(x="age", y="hours.per.week", data=newData, hue="income")

#ax = sns.barplot(x="income", y="hours.per.week", data=newData)

#ax = sns.barplot(x="sex", y="hours.per.week", data=newData, hue="income")

#ax = sns.scatterplot(x="capital.gain", y="age", data=newData, hue="income")

#ax = sns.lineplot(x="age", y="hours.per.week", data=newData, hue="income")
dataBvA = newData

fig,ax = plt.subplots(figsize=(10,8))

ax = sns.scatterplot(x="education.num", y="hours.per.week", data=dataBvA, hue="income")
dataEvA = newData

fig,ax = plt.subplots(figsize=(10,8))

ax = sns.lineplot(x="age", y="capital.gain", data=dataEvA, hue="sex")

dataESB = newData

dataESB[dataESB["income"] == '>50k']

dataESB = dataESB.groupby("occupation", as_index=False)["fnlwgt"].sum().sort_values(by="fnlwgt", ascending=False).head(10)



fig,ax = plt.subplots(figsize=(20,10))

ax = sns.barplot(x="occupation", y="fnlwgt", data=dataESB)



#dataWaI = newData

#dataESB = dataESB.groupby("occupation", as_index=False).count().sort_values(by="income", ascending=False)
dataBvE = newData

dataBvE = dataBvE[(dataBvE["income"] == '>50K')]

dataBvE = dataBvE.groupby("education.num", as_index=False)["fnlwgt"].sum()



dataBvE1 = newData

dataBvE1 = dataBvE1.groupby("education.num", as_index=False)["fnlwgt"].sum()





dataBvE.loc[-1] = [1.0, 0]

dataBvE.index = dataBvE.index + 1  # shifting index

dataBvE.sort_index(inplace=True)



dataBvE["relativeHighEarners"] = dataBvE["fnlwgt"] / dataBvE1["fnlwgt"] * 100





fig,ax = plt.subplots(figsize=(10,8))

ax = sns.barplot(x="education.num", y="relativeHighEarners", data=dataBvE).set(ylim=(0,100))









dataJaR = newData

dataJaR = dataJaR[(dataJaR["income"] == '>50K')]

dataJaR = dataJaR.groupby(["workclass", "race"], as_index=False)["fnlwgt"].sum()



dataJaR1 = newData

dataJaR1 = dataJaR1.groupby(["workclass", "race"], as_index=False)["fnlwgt"].sum()



dataJaR1

dataJaR.index = dataJaR.index * 2  # shifting index

dataJaR.loc[5] = ["Federal-gov", "Other", 0]

dataJaR.loc[27] = ["Self-emp-inc", "Amer-Indian-Eskimo", 0]

dataJaR.loc[60] = ["Without-Pay", "Asian-Pac-Islander", 0]

dataJaR.loc[61] = ["Without-Pay", "Black", 0]

dataJaR.loc[62] = ["Without-Pay", "White", 0]

dataJaR.sort_index(inplace=True)



dataJaR.index = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]



dataJaR["relativeHighEarners"] = dataJaR["fnlwgt"] / dataJaR1["fnlwgt"] * 100





dataJaR = dataJaR.drop("fnlwgt", axis = 1)



dataJaR.loc[33] = ["Without-Pay", "Amer-Indian-Eskimo", 0]

dataJaR.loc[34] = ["Without-Pay", "Other", 0]

dataJaR.sort_index(inplace=True)







normal_data = dataJaR.pivot(index='race', columns='workclass', values='relativeHighEarners')



fig,ax = plt.subplots(figsize=(5,4))

ax = sns.heatmap(normal_data, vmin=0, vmax =100, cmap="Reds", center=40, cbar_kws={'label': '% with income >50k'})



#Reds




