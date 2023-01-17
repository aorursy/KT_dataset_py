import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)          

import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('../input/top50spotify2019/top50.csv',index_col=0, encoding='ISO-8859-1')

df.head()
df.describe()
artist = df['Artist.Name'].nunique()

print(f'There are {artist} artists in the Spotify top 50.')
top = pd.DataFrame(df['Artist.Name'].value_counts().reset_index().rename(columns = {'index' : 'Artist Name'}))

top.rename(columns={'Artist.Name': 'Total'}, inplace=True)





plt.style.use('default')

plt.figure(figsize=(15,8))





p = sns.barplot(x=top['Total'], y=top['Artist Name'], linewidth=6)

p.axes.set_title("Total Number of Top 50 Songs by Artist",fontsize=20)

p.set_xlabel("Number of Songs",fontsize=15)

p.set_ylabel("Artist Name",fontsize=15)

p.tick_params(labelsize=10)
genre = df['Genre'].nunique()

print(f'There are {genre} different genres in the Spotify top 50.')
plt.style.use('default')

plt.figure(figsize=(15,8))

p = sns.countplot(y='Genre', data = df, linewidth=2, edgecolor='black')

p.axes.set_title("Total Number of Songs by Genre",fontsize=20)

p.set_xlabel("Number of Songs",fontsize=15)

p.set_ylabel("Genre",fontsize=15)

plt.show()
# set figure size

plt.style.use('default')

plt.figure(figsize=(15,8))



# Swarmplot

p = sns.swarmplot(x='Genre', y='Popularity', data=df, s=10)



p.axes.set_title("Popularity by Genre",fontsize=20)

p.set_xlabel("Genre",fontsize=15)

p.set_ylabel("Popularity",fontsize=15)

p.tick_params(labelsize=15)

plt.xticks(rotation=85)
plt.style.use('default')

plt.figure(figsize=(15,8))

p = sns.regplot(x="Energy", y="Popularity", data=df)

p.axes.set_title("Popularity by Energy of the song",fontsize=20)

p.set_xlabel("Energy",fontsize=15)

p.set_ylabel("Popularity",fontsize=15)

p.tick_params(labelsize=15)

plt.style.use('default')



plt.figure(figsize=(15,8))

p = sns.regplot(x="Danceability", y="Popularity", data=df)

p.axes.set_title("Popularity by Danceability of the song",fontsize=20)

p.set_xlabel("Danceability",fontsize=15)

p.set_ylabel("Popularity",fontsize=15)

p.tick_params(labelsize=15)

plt.style.use('default')



plt.figure(figsize=(15,8))

p = sns.regplot(x="Length.", y="Popularity", data=df)

p.axes.set_title("Popularity by Length of the song",fontsize=20)

p.set_xlabel("Length in seconds (approx.)",fontsize=15)

p.set_ylabel("Popularity",fontsize=15)

p.tick_params(labelsize=15)