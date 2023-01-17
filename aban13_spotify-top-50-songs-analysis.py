# Data analysis packages

import pandas as pd

import numpy as np



# Visualization libraries

import seaborn as sns

import matplotlib.pyplot as plt

from plotnine import *

%matplotlib inline

sns.set(style='dark')

df = pd.read_csv("../input/top50spotify2019/top50.csv", encoding='ISO-8859-1')
df.head()
df.shape
df.info()
# Dropping 'Unnamed: 0' since it doesn't consist of any relevant information

df.drop(['Unnamed: 0'], axis=1, inplace=True)
df.rename(columns={'Track.Name':'Track_Name', 

                   'Artist.Name':'Artist_Name',

                   'Beats.Per.Minute':'Beats_Per_Minute', 

                   'Loudness..dB..':'Loudness',

                   'Valence.':'Valence', 

                   'Length.':'Length', 

                   'Acousticness..':'Acousticness',

                   'Speechiness.':'Speechiness'}, inplace=True)
df.describe().T
df.Genre.nunique()
df.Genre.value_counts()
plt.style.use('fivethirtyeight')

plt.figure(figsize = (16,10));

sns.countplot(x="Genre", data=df, linewidth=2, edgecolor='black');

plt.ylabel('Number of occurances');

plt.xticks(rotation=45, ha='right');
df.Artist_Name.nunique()
df.Artist_Name.value_counts()
plt.figure(figsize=(20,8))

plt.style.use('fivethirtyeight')

sns.countplot(x=df['Artist_Name'],data=df, linewidth=2, edgecolor='black')

plt.title('Number of times an artist appears in the top 50 songs list')

plt.xticks(rotation=45, ha='right')

plt.show()
top_artists = df.groupby('Artist_Name')

filtered_data = top_artists.filter(lambda x: x['Artist_Name'].value_counts() > 1)
plt.figure(figsize=(20,8))

plt.style.use('fivethirtyeight')

sns.countplot(y=filtered_data['Artist_Name'],data=filtered_data, linewidth=2, edgecolor='black', order=filtered_data["Artist_Name"].value_counts().index)

plt.title('Top Artists of 2019')

plt.xticks(rotation=45, ha='right')

plt.show()
values = df.Liveness.value_counts()

indexes = values.index



fig = plt.figure(figsize=(15, 8))

sns.barplot(indexes, values,linewidth=2, edgecolor='black')



plt.ylabel('Number of occurances')

plt.xlabel('Liveness')
minimum_Liveness = df[df.Liveness == df.Liveness.min()]

minimum_Liveness[['Track_Name', 'Artist_Name', 'Genre', 'Liveness']]
maximum_Liveness = df[df.Liveness == df.Liveness.max()]

maximum_Liveness[['Track_Name', 'Artist_Name', 'Genre', 'Liveness']]
plt.figure(figsize=(8,4))

sns.distplot(df.Valence, kde=False, bins=15,color='blue', hist_kws=dict(edgecolor="black", linewidth=1))

plt.show()
minimum_Valence = df[df.Valence == df.Valence.min()]

minimum_Valence[['Track_Name', 'Artist_Name', 'Genre', 'Valence']]
maximum_Valence = df[df.Valence == df.Valence.max()]

maximum_Valence[['Track_Name', 'Artist_Name', 'Genre', 'Valence']]
plt.figure(figsize=(8,4))

sns.distplot(df['Length'], kde=False, bins=15,color='green', hist_kws=dict(edgecolor="black", linewidth=1))

plt.show()
plt.figure(figsize=(8,4))

sns.distplot(df['Loudness'], kde=False, bins=15,color='red', hist_kws=dict(edgecolor="black", linewidth=1))

plt.show()
plt.figure(figsize=(8,4))

sns.distplot(df['Danceability'], kde=False, bins=15,color='violet', hist_kws=dict(edgecolor="black", linewidth=1))

plt.show()
plt.figure(figsize=(8,4))

sns.distplot(df['Energy'], kde=False, bins=15,color='#F06292', hist_kws=dict(edgecolor="k", linewidth=1))

plt.show()
plt.figure(figsize=(8,4))

sns.distplot(df['Beats_Per_Minute'], kde=False, bins=18,color='#E67E22', hist_kws=dict(edgecolor="black", linewidth=1))

plt.show()
correlations = df.corr()



fig = plt.figure(figsize=(12, 8))

sns.heatmap(correlations, annot=True, linewidths=1, cmap='YlGnBu', center=1)

plt.show()
sns.set_style('whitegrid')

sns.pairplot(df)

plt.show()
fig = plt.figure(figsize=(8, 6))

sns.regplot(x='Energy', y='Loudness', data=df)

plt.show()
sns.catplot(x = "Loudness", y = "Energy", kind = "box", data = df)

plt.show()
sns.jointplot(x="Beats_Per_Minute", y="Speechiness", data=df, kind="kde");