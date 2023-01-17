import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure, show
df_Netflix = pd.read_csv('/kaggle/input/netflix-shows/netflix_titles.csv')
df_Netflix.head(10)
sns.countplot(x='type', data=df_Netflix, palette='Paired')
figure(figsize=(15,4))

sns.countplot(x='rating', data=df_Netflix, palette='Blues')
Release = df_Netflix[['release_year']]
Release.groupby('release_year').size()
figure(figsize=(25,10))

plt.xticks(rotation=90)

sns.countplot(x= 'release_year', data=Release, palette='Blues')

netflix_date = df_Netflix[['date_added']].dropna()

netflix_date['year'] = netflix_date['date_added'].apply(lambda year : year[-4:])

netflix_date['month'] = netflix_date['date_added'].apply(lambda month : month[0:-9])



month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'][::-1]

df = netflix_date.groupby('year')['month'].value_counts().unstack().fillna(0)[month_order].T
plt.figure(figsize=(7,4), dpi=150)

plt.pcolor(df, cmap='Blues', edgecolors='white', linewidths=2)

plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns, fontsize=7, fontfamily='serif')

plt.yticks(np.arange(0.5, len(df.index), 1), df.index, fontsize=7, fontfamily='serif')



plt.title('Netflix Contents Update', fontsize=10, fontfamily='serif', fontweight='bold', position=(0.23, 1.0+0.02))

cbar = plt.colorbar()



cbar.ax.tick_params(labelsize=6) 

cbar.ax.minorticks_on()

plt.show()