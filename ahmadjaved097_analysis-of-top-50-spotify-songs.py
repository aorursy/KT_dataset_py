import warnings

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# install pywaffle for waffle charts

!pip install pywaffle
import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

from pywaffle import Waffle

import random



%matplotlib inline
# setting plot style for all the plots

plt.style.use('fivethirtyeight')



#accessing all the colors from matplotlib

colors=list(matplotlib.colors.CSS4_COLORS.keys())

df = pd.read_csv('/kaggle/input/top50spotify2019/top50.csv', encoding="ISO-8859-1")

df.head()
#Drop the Unnamed: 0 column



df.drop('Unnamed: 0', inplace=True, axis=1)
# Renaming the columns



df.rename(columns={'Track.Name':'Track Name',

                   'Artist.Name':'Artist Name',

                   'Genre':'Genre',

                   'Beats.Per.Minute':'Beats per Minute',

                   'Energy':'Energy',

                   'Danceability':'Danceability',

                   'Loudness..dB..':'Loudness(dB)',

                   'Liveness':'Liveness',

                   'Valence.':'Valence',

                   'Length.':'Length',

                   'Acousticness..':'Acousticness',

                   'Speechiness.':'Speechiness',

                   'Popularity':'Popularity'}, inplace=True)
# let's see the dataset again

df.head()
df.info()
print('Number of rows in the dataset: ',df.shape[0])

print('Number of columns in the dataset: ',df.shape[1])
df.describe().round(decimals=3)
df['Artist Name'].nunique()
df['Genre'].nunique()
plt.figure(figsize=(8,4))

sns.distplot(df['Beats per Minute'], kde=False, bins=18,color='#3ff073', hist_kws=dict(edgecolor="black", linewidth=1))

plt.show()
minimum_beats_per_min = df[df['Beats per Minute'] == df['Beats per Minute'].min()]

minimum_beats_per_min[['Track Name', 'Artist Name', 'Genre', 'Beats per Minute']].reset_index().drop('index', axis=1)
maximum_beats_per_min = df[df['Beats per Minute'] == df['Beats per Minute'].max()]

maximum_beats_per_min[['Track Name', 'Artist Name', 'Genre', 'Beats per Minute']].reset_index().drop('index', axis=1)
plt.figure(figsize=(8,4))

sns.distplot(df['Energy'], kde=False, bins=15,color='red', hist_kws=dict(edgecolor="k", linewidth=1))

plt.show()
minimum_energy = df[df['Energy'] == df['Energy'].min()]

minimum_energy[['Track Name', 'Artist Name', 'Genre', 'Energy']].reset_index().drop('index', axis=1)
maximum_energy = df[df['Energy'] == df['Energy'].max()]

maximum_energy[['Track Name', 'Artist Name', 'Genre', 'Energy']].reset_index().drop('index', axis=1)
plt.figure(figsize=(8,4))

sns.distplot(df['Danceability'], kde=False, bins=15,color='violet', hist_kws=dict(edgecolor="black", linewidth=1))

plt.show()
maximum_danceability = df[df['Danceability'] == df['Danceability'].max()]

maximum_danceability[['Track Name', 'Artist Name', 'Genre', 'Danceability']].reset_index().drop('index', axis=1)
minimum_danceability = df[df['Danceability'] == df['Danceability'].min()]

minimum_danceability[['Track Name', 'Artist Name', 'Genre', 'Danceability']].reset_index().drop('index', axis=1)
plt.figure(figsize=(8,4))

sns.distplot(df['Loudness(dB)'], kde=False, bins=15,color='aqua', hist_kws=dict(edgecolor="black", linewidth=1))

plt.show()
minimum_loudness = df[df['Loudness(dB)'] == df['Loudness(dB)'].min()]

minimum_loudness[['Track Name', 'Artist Name', 'Genre', 'Loudness(dB)']].reset_index().drop('index', axis=1)
maximum_loudness = df[df['Loudness(dB)'] == df['Loudness(dB)'].max()]

maximum_loudness[['Track Name', 'Artist Name', 'Genre', 'Loudness(dB)']].reset_index().drop('index', axis=1)
plt.figure(figsize=(8,4))

sns.distplot(df['Liveness'], kde=False, bins=15,color='darkorchid', hist_kws=dict(edgecolor="black", linewidth=1))

plt.show()
minimum_Liveness = df[df['Liveness'] == df['Liveness'].min()]

minimum_Liveness[['Track Name', 'Artist Name', 'Genre', 'Liveness']].reset_index().drop('index', axis=1)
maximum_Liveness = df[df['Liveness'] == df['Liveness'].max()]

maximum_Liveness[['Track Name', 'Artist Name', 'Genre', 'Liveness']].reset_index().drop('index', axis=1)
plt.figure(figsize=(8,4))

sns.distplot(df['Valence'], kde=False, bins=15,color='darkgreen', hist_kws=dict(edgecolor="black", linewidth=1))

plt.show()
minimum_Valence = df[df['Valence'] == df['Valence'].min()]

minimum_Valence[['Track Name', 'Artist Name', 'Genre', 'Valence']].reset_index().drop('index', axis=1)
maximum_Valence = df[df['Valence'] == df['Valence'].max()]

maximum_Valence[['Track Name', 'Artist Name', 'Genre', 'Valence']].reset_index().drop('index', axis=1)
plt.figure(figsize=(8,4))

sns.distplot(df['Length'], kde=False, bins=15,color='m', hist_kws=dict(edgecolor="black", linewidth=1))

plt.show()
minimum_Length = df[df['Length'] == df['Length'].min()]

minimum_Length[['Track Name', 'Artist Name', 'Genre', 'Length']].reset_index().drop('index', axis=1)
maximum_Length = df[df['Length'] == df['Length'].max()]

maximum_Length[['Track Name', 'Artist Name', 'Genre', 'Length']].reset_index().drop('index', axis=1)
plt.figure(figsize=(8,4))

sns.distplot(df['Acousticness'], kde=False, bins=15,color='darkblue', hist_kws=dict(edgecolor="black", linewidth=1))

plt.show()
minimum_Acousticness = df[df['Acousticness'] == df['Acousticness'].min()]

minimum_Acousticness[['Track Name', 'Artist Name', 'Genre', 'Acousticness']].reset_index().drop('index', axis=1)
maximum_Acousticness = df[df['Acousticness'] == df['Acousticness'].max()]

maximum_Acousticness[['Track Name', 'Artist Name', 'Genre', 'Acousticness']].reset_index().drop('index', axis=1)
plt.figure(figsize=(8,4))

sns.distplot(df['Popularity'], kde=False, bins=15,color='orange', hist_kws=dict(edgecolor="black", linewidth=1))

plt.show()
minimum_Popularity = df[df['Popularity'] == df['Popularity'].min()]

minimum_Popularity[['Track Name', 'Artist Name', 'Genre', 'Popularity']].reset_index().drop('index', axis=1)
maximum_Popularity = df[df['Popularity'] == df['Popularity'].max()]

maximum_Popularity[['Track Name', 'Artist Name', 'Genre', 'Popularity']].reset_index().drop('index', axis=1)
plt.style.use('fivethirtyeight')

plt.figure(figsize=(16,8))

sns.countplot(x='Genre', data = df, linewidth=2, edgecolor='black')

plt.xticks(rotation=90)

plt.show()
calculated = df.Genre.value_counts()

sns.set_style('darkgrid')

fig = plt.figure(figsize=(13,8),

    FigureClass=Waffle, 

    rows=5, 

    values=list(calculated.values),

    labels=list(calculated.index),

                 legend={'loc': 'upper left', 'bbox_to_anchor': (1.1, 1)},

                 edgecolor='black',

                 colors= random.sample(colors,21),

)
plt.figure(figsize=(12,8))

corr = df.corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



sns.heatmap(corr,mask=mask, annot=True, linewidths=1, cmap='YlGnBu')

plt.show()
sns.set_style('whitegrid')

sns.pairplot(df)

plt.show()
plt.figure(figsize=(20,8))

plt.style.use('fivethirtyeight')

sns.countplot(x=df['Artist Name'],data=df, linewidth=2, edgecolor='black')

plt.title('Number of times an artist appears in the top 50 songs list')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(20,6))

sns.boxplot(x='Genre', y='Energy', data = df, linewidth=2)

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(20,6))

sns.boxplot(x='Genre', y='Beats per Minute', data = df, linewidth=2)

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(20,6))

sns.boxplot(x='Genre', y='Danceability', data = df, linewidth=2)

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(20,6))

sns.boxplot(x='Genre', y='Loudness(dB)', data = df, linewidth=2)

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(20,6))

sns.boxplot(x='Genre', y='Liveness', data = df, linewidth=2)

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(20,6))

sns.boxplot(x='Genre', y='Valence', data = df, linewidth=2)

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(20,5))

sns.boxplot(x='Genre', y='Length', data = df, linewidth=2)

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(20,6))

sns.boxplot(x='Genre', y='Acousticness', data = df, linewidth=2)

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(20,6))

sns.boxplot(x='Genre', y='Speechiness', data = df, linewidth=2)

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(20,6))

sns.boxplot(x='Genre', y='Popularity', data = df, linewidth=2)

plt.xticks(rotation=90)

plt.show()
colors=list(matplotlib.colors.CSS4_COLORS.keys())

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18,10))

axes = axes.flatten()



numeric_cols = list(df.select_dtypes([np.number]).columns[:-1])         #selecting all the numeric columns except Popularity

plt.tight_layout(pad=2)

for i, j in enumerate(numeric_cols):

    axes[i].scatter(x=df[j], y=df['Popularity'], color= random.choice(colors), edgecolor='black')

    axes[i].set_xlabel(j)

    axes[i].set_ylabel('Popularity')