import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import pandas as pd 

import warnings

warnings.filterwarnings("ignore",category = DeprecationWarning)

warnings.filterwarnings("ignore",category = FutureWarning)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
filename='/kaggle/input/top50spotify2019/top50.csv'

spoti=pd.read_csv(filename,encoding='ISO-8859-1')

spoti.head(10) # It shows first 10 row information in data 
spoti.info()

#It gives information about data.
# Show some statistics about dataset

spoti.describe()
spoti.columns

#It gives what we have columns
gb_genre =spoti.groupby("Genre").sum()

#It classified by genre variable 

gb_genre.head()
#Calculates the number of rows and columns

print(spoti.shape)
#Heatmap

plt.figure(figsize=(10,10))

plt.title('Correlation Map')

ax=sns.heatmap(spoti.corr(),

               linewidth=3.1,

               annot=True,

               center=1)
#Boxplot

#It shows outlier values and value of popularity

sns.boxplot( y = spoti["Popularity"])

plt.show()
#Catplot

#It gives count of genre in spotify top 50 list. 

sns.catplot(y = "Genre", kind = "count",

            palette = "pastel", edgecolor = ".6",

            data = spoti)

plt.show()
plt.figure(figsize=(12,12))

sns.jointplot(x=spoti["Beats.Per.Minute"].values, y=spoti['Popularity'].values, size=10, kind="kde",)

plt.ylabel('Popularity', fontsize=12)

plt.xlabel("Beats.Per.Minute", fontsize=12)

plt.title("Beats.Per.Minute Vs Popularity", fontsize=15);

#The purpose of this graph is to show connection among Beats and Popularity
threshold = sum(spoti.Energy)/len(spoti.Energy)

print(threshold)

spoti["Energy_level"] = ["energized" if i > threshold else "without energy" for i in spoti.Energy]

spoti.loc[:10,["Energy_level","Energy"]]

#This caught my attention to the effect of energy level on music in here and i calcuted it. It classified according to mean of value
plt.figure(figsize=(12,8))

sns.violinplot(x='Loudness..dB..', y='Popularity', data=spoti)

plt.xlabel('Loudness..dB..', fontsize=12)

plt.ylabel('Popularity', fontsize=12)

plt.show()

# I want to show relationship loudness and popularity. From there we can learn to contribution of loudness level to popularity
# Some kind of Histogram Plot

f, ax = plt.subplots(figsize=(10,8))

x = spoti['Loudness..dB..']

ax = sns.distplot(x, bins=10)

plt.show()
sns.pairplot(spoti)

plt.plot()

plt.show()

# It shows all histogram graph with data colums.
sns.lmplot(x="Energy",y="Popularity",data=spoti,size=10,hue="Genre")



plt.plot()

plt.show()

#this graph is so attractive because of different from other. My target in there is to show to Excellence of connection of Energy and Poularity  