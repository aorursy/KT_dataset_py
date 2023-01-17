import numpy as np

import pandas as pd 

import seaborn as sbr

import matplotlib.pyplot as plt

import warnings

from mpl_toolkits.mplot3d import Axes3D

import os

print(os.listdir("../input"))
data = pd.read_csv('../input/Pokemon.csv')

data.head()
data.info()
data.columns = data.columns.str.upper().str.replace(' ', '')

data.head()
data.describe()
sbr.jointplot(x="ATTACK",y="DEFENSE",data=data,kind="hex",color="green");

warnings.filterwarnings("ignore")
data.corr()
f,ax=plt.subplots(figsize=(18,18))

sbr.heatmap(data.corr(),annot=True,fmt=".1f",linewidths=.5,cmap="YlGnBu",ax=ax,linecolor="black")

plt.show()

f,ax=plt.subplots(figsize=(15,15))

sbr.violinplot(x=data.GENERATION,y=data.ATTACK,inner=None)

sbr.swarmplot(x=data.GENERATION,y=data.ATTACK,size=5,color="black")

warnings.filterwarnings("ignore")
f,ax=plt.subplots(figsize=(20,20))

sbr.boxplot(x=data.GENERATION,y=data.TOTAL,palette="Set3")

sbr.stripplot(x=data.GENERATION,y=data.TOTAL,hue=data.LEGENDARY,linewidth=1,size=12,jitter=True,edgecolor="black")

warnings.filterwarnings("ignore")
sbr.set(rc={'figure.figsize':(10,10)})

sbr.barplot(x=data.TOTAL,y=data.TYPE1,data=data,palette="Set2",linewidth=1)

warnings.filterwarnings("ignore")
sbr.set(style="whitegrid")

sbr.residplot(x=data.GENERATION,y=data.TOTAL,lowess=True,color="r",order=0.01,robust=True)

warnings.filterwarnings("ignore")
fig=plt.figure()

ax=fig.add_subplot(111,projection="3d")

ax.scatter(data["ATTACK"],data["TOTAL"],data["GENERATION"],c="orange",s=30)

ax.view_init(20,250)

plt.show()
sbr.boxenplot(x="GENERATION",y="ATTACK",hue="LEGENDARY",data=data)

warnings.filterwarnings("ignore")
plt.figure(figsize=(17,10))

sbr.regplot(x="ATTACK",y="TOTAL",data=data)

plt.xlabel("HP",size=20)

plt.ylabel("ATTACK",size=20)

plt.show()
data.head()
plt.figure(figsize=(12,12))

sbr.scatterplot(x="SPEED",y="TOTAL",data=data,hue="GENERATION")

plt.show()