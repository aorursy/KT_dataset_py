import numpy as np                # linear algebra

import pandas as pd               # data frames

import seaborn as sns             # visualizations

import matplotlib.pyplot as plt   # visualizations

import scipy.stats                # statistics

from sklearn import preprocessing

from statistics import mean 

import os

print(os.listdir("../input"))
points=pd.read_csv("../input/points.csv")

serves=pd.read_csv("../input/serves.csv")

rallies=pd.read_csv("../input/rallies.csv")

events=pd.read_csv("../input/events.csv")
points.head()
serves.head()
rallies.head()
events.head()
points[['rallyid','winner']].groupby('winner').count()
df=points.groupby(['winner','serve']).count().iloc[:,:1]

df.columns=['Points Won']

df
df2=points.groupby(['reason']).count().iloc[:,:1]

df2.columns=['Points Won']

df2
df3=points.groupby(['winner','reason']).count().iloc[:,:1]

df3.columns=['Points Won']

df3
f, axes = plt.subplots(1,2, figsize=(15, 5))

sns.countplot(x="reason", data=points, ax=axes[0], palette="Set1")

sns.countplot(x="reason", hue='winner',data=points, ax=axes[1] ,palette="Set1")
sns.catplot(x="reason", hue='winner', col='serve',data=points, kind='count' ,palette="Set1")
sns.distplot(points["strokes"],  color="red")

print("The mean of strokes was: " + str(mean(points["strokes"])))
f, axes = plt.subplots(1,2, figsize=(15, 5))

sns.countplot(x="strokes", hue='serve',data=points, palette="Set1", ax=axes[0])

sns.countplot(x="strokes", hue='winner',data=points ,palette="Set1", ax=axes[1])
print("Segundos de juego: " + str(points.totaltime.sum())) #Segundos de juego

print("Minutos de juego: " + str(points.totaltime.sum()/60)) #Minutos de juego

print("Porcentaje de juego durante el partido (2h 4m): " + str(((points.totaltime.sum()/60)/124)*100) + " %")


sns.distplot(points["totaltime"],  color="red")
f, axes = plt.subplots(1,2, figsize=(15, 5))

sns.scatterplot(x="totaltime", y="strokes", data=points, ax=axes[0])

sns.scatterplot(x="totaltime", y="strokes", hue="winner" ,data=points, ax=axes[1])
df4=serves.groupby(['server']).count().iloc[:,:1]

df4.columns=['Serves']

df4
rallies1=rallies.replace({"__undefined__":"Out/Net (Not Point)"})

df5=rallies1.groupby(['server', 'winner']).count().iloc[:,:1]

df5.columns=['Points Won']

df5

f, axes = plt.subplots(1,2, figsize=(15, 5))

sns.countplot(x="type",data=events, palette="Set1", ax=axes[0])

sns.countplot(x="type", hue="hitter", data=events, palette="Set1", ax=axes[1])
f, axes = plt.subplots(1,2, figsize=(15, 5))

sns.countplot(y="stroke",data=events, palette="Set1", ax=axes[0])



events1=events.replace({"__undefined__":"forehand"})

sns.countplot(y="stroke",data=events1, palette="Set1", ax=axes[1])
f, axes = plt.subplots(1,2, figsize=(15, 5))

sns.countplot(x="stroke",hue="hitter",data=events1, palette="Set1", ax=axes[0])

sns.countplot(x="type", hue="stroke", data=events1, palette="Set1", ax=axes[1])
sns.catplot(x="type", hue='stroke', col='hitter',data=events1, kind='count' ,palette="Set1")
sns.scatterplot(y="hitter_x", x="hitter_y", hue="hitter", data=events)

plt.plot([0,0],[0,10.97], 'k')

plt.plot([23.77,0],[0,0], 'k')

plt.plot([23.77,23.77],[0,10.97], 'k')

plt.plot([23.77,0],[10.97,10.97], 'k')



plt.plot([11.985,11.985],[0,10.97], 'k',linestyle='dashed')



plt.plot([23.77,0],[1.37,1.37], 'k')

plt.plot([23.77,0],[9.6,9.6], 'k')



plt.plot([18.385,18.385],[1.37,9.6], 'k')

plt.plot([5.585,5.585],[1.37,9.6], 'k')



plt.plot([5.585,18.385],[5.485,5.485], 'k')
sns.scatterplot(y="receiver_x", x="receiver_y", hue="receiver", data=events)

plt.plot([0,0],[0,10.97], 'k')

plt.plot([23.77,0],[0,0], 'k')

plt.plot([23.77,23.77],[0,10.97], 'k')

plt.plot([23.77,0],[10.97,10.97], 'k')



plt.plot([11.985,11.985],[0,10.97], 'k',linestyle='dashed')



plt.plot([23.77,0],[1.37,1.37], 'k')

plt.plot([23.77,0],[9.6,9.6], 'k')



plt.plot([18.385,18.385],[1.37,9.6], 'k')

plt.plot([5.585,5.585],[1.37,9.6], 'k')



plt.plot([5.585,18.385],[5.485,5.485], 'k')
sns.scatterplot(y="x", x="y", hue="winner", data=points)

plt.plot([0,0],[0,10.97], 'k')

plt.plot([23.77,0],[0,0], 'k')

plt.plot([23.77,23.77],[0,10.97], 'k')

plt.plot([23.77,0],[10.97,10.97], 'k')



plt.plot([11.985,11.985],[0,10.97], 'k',linestyle='dashed')



plt.plot([23.77,0],[1.37,1.37], 'k')

plt.plot([23.77,0],[9.6,9.6], 'k')



plt.plot([18.385,18.385],[1.37,9.6], 'k')

plt.plot([5.585,5.585],[1.37,9.6], 'k')



plt.plot([5.585,18.385],[5.485,5.485], 'k')
sns.scatterplot(y="x", x="y", hue="server", data=serves)

plt.plot([0,0],[0,10.97], 'k')

plt.plot([23.77,0],[0,0], 'k')

plt.plot([23.77,23.77],[0,10.97], 'k')

plt.plot([23.77,0],[10.97,10.97], 'k')



plt.plot([11.985,11.985],[0,10.97], 'k',linestyle='dashed')



plt.plot([23.77,0],[1.37,1.37], 'k')

plt.plot([23.77,0],[9.6,9.6], 'k')



plt.plot([18.385,18.385],[1.37,9.6], 'k')

plt.plot([5.585,5.585],[1.37,9.6], 'k')



plt.plot([5.585,18.385],[5.485,5.485], 'k')