# Osman Balli





import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

data = pd.read_csv("../input/top50spotify2019/top50.csv",encoding='ISO-8859-1')

data.head()
data.shape
data.describe()
data.columns
# columns have been renamed and deleted "Unnamed: 0" column

data.rename(columns={"Loudness..dB..":"Loudness",

                     "Valance.":"Valance",

                     "Length.":"Length",

                     "Acousticness..":"Acousticness",

                     "Speechiness.":"Speechiness",

                     "Beats.Per.Minute":"BPM",

                     "Track.Name":"Track",

                     "Artist.Name":"Artist"},inplace=True)

data.drop("Unnamed: 0",axis=1,inplace=True)

print(data.columns)

# checked null values

print(data.isnull().sum())

# Grouped by genre

genre_ = data.groupby('Genre').groups

print(genre_)

# Grouped by genre

genre_ = data.groupby('Genre').sum()

genre_.head()
sns.catplot(y = "Genre", kind = "count",

            palette = "pastel", edgecolor = ".6",

            data = data)
#energy values of the groups are listed

genre_energy=data.groupby("Genre",as_index=False)['Energy'].mean().sort_values("Energy")

print(genre_energy)
#energy values of the groups are plotted

labels=genre_energy["Genre"]

x=np.arange(0,len(genre_energy["Genre"]))

y=genre_energy["Energy"]

plt.plot(x,y,linestyle='solid')

plt.xticks(x, labels, rotation=75)

plt.ylabel("Energy")

plt.xlabel("Genre")

plt.legend("Mean Energy")

plt.title("Ranking of Mean Energy in Genres")

plt.grid(color='r', linestyle='dotted', linewidth=0.5)

plt.show()
columns=["Energy","Length","Acousticness","BPM","Danceability","Loudness","Liveness","Speechiness","Popularity"]

maximum=[]



genre=data.groupby("Genre",as_index=False)[columns].mean()

print(genre)

#The maximum values of the columns and their groups

for i in columns:

    maximum.append(genre[i].agg(np.max))

k=0

for i in range(0,len(genre["Genre"])):

    for j in columns:

        if genre.loc[i,j]==maximum[k]:

            print("maximum "+j+"="+str(maximum[k])+"--->"+genre.loc[i,"Genre"])

        k=k+1

        if k==len(maximum):

            break

    k=0
# 10 most popular songs

popularity = data.sort_values("Popularity",  ascending = False).head(10)[["Popularity","Artist","Genre","Energy"]]

print(popularity)
#correlation between features

#correlation map 

plt.figure(figsize=(9,9))

plt.title('Correlation Map')

ax=sns.heatmap(genre.corr(),

               linewidth=1,

               annot=True,

               center=2)


plt.figure(figsize=(12,8))

sns.violinplot(x='Loudness', y='Energy', data=data)

plt.xlabel('Loudness', fontsize=12)

plt.ylabel('Energy', fontsize=12)
