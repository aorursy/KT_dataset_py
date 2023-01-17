# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from scipy import stats

import plotly.express as px



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv(r'/kaggle/input/top50spotify2019/top50.csv',encoding='ISO-8859-1')

data.head()
print("Number of datapoints in the data : ",data.shape[0])

print("Number of features in the data : ",data.shape[1])

print("Features :",data.columns.values)
data.info()
data.describe()
data.dtypes
print("Number of duplicate values : ",data.duplicated().sum())
plt.figure(figsize=(20,20))

sns.pairplot(data)

plt.show()
plt.figure(figsize=(10,8))

sns.countplot('Genre',data=data)

plt.xticks(rotation=90)

plt.title("Number of songs by genres")

plt.show()
plt.figure(figsize=(15,8))

sns.countplot('Artist.Name',data=data)

plt.xticks(rotation=90)

plt.title("Number of songs by each atrist")

plt.show()
energy_genre = (data.groupby("Genre",as_index=False)['Energy'].mean().sort_values("Energy")).reset_index(drop=True)

print(energy_genre)



plt.figure(figsize=(10,8))

plt.plot(energy_genre['Genre'],energy_genre['Energy'],color='green')

plt.grid()

plt.xlabel('Genre')

plt.ylabel('Energy')

plt.xticks(rotation=90)

plt.show()



energy_artistwise = (data.groupby("Artist.Name",as_index=False)['Energy'].mean().sort_values("Energy")).reset_index(drop=True)

print(energy_artistwise)



plt.figure(figsize=(10,8))

plt.plot(energy_artistwise['Artist.Name'],energy_artistwise['Energy'],color='green')

plt.grid()

plt.xlabel('Artist Names')

plt.ylabel('Energy')

plt.xticks(rotation=90)

plt.show()
loudness_genre = (data.groupby("Genre",as_index=False)['Loudness..dB..'].mean().sort_values("Loudness..dB..")).reset_index(drop=True)

print(loudness_genre)



plt.figure(figsize=(10,8))

plt.plot(loudness_genre['Genre'],loudness_genre['Loudness..dB..'],color='green')

plt.grid()

plt.xlabel('Genre')

plt.ylabel('Loudness')

plt.xticks(rotation=90)

plt.show()



loudness_artistwise = (data.groupby("Artist.Name",as_index=False)['Loudness..dB..'].mean().sort_values('Loudness..dB..')).reset_index(drop=True)

print(loudness_artistwise)



plt.figure(figsize=(10,8))

plt.plot(loudness_artistwise['Artist.Name'],loudness_artistwise['Loudness..dB..'],color='green')

plt.grid()

plt.xlabel('Artist Names')

plt.ylabel('Loudness')

plt.xticks(rotation=90)

plt.show()
#with the original data

fig,ax = plt.subplots(4,2,figsize=(20,20))

sns.distplot(data['Beats.Per.Minute'],color='red',ax=ax[0][0],bins=10)

sns.distplot(data['Energy'],ax=ax[0][1],color='violet',bins=10)

sns.distplot(data['Loudness..dB..'],ax=ax[1][0],color='yellow',bins=10)

sns.distplot(data['Liveness'],ax=ax[1][1],color='blue',bins=10)

sns.distplot(data['Acousticness..'],ax=ax[2][0],color='green',bins=10)

sns.distplot(data['Length.'],ax=ax[2][1],color='black',bins=10)

sns.distplot(data['Popularity'],ax=ax[3][0],color='purple',bins=10)

sns.distplot(data['Speechiness.'],ax=ax[3][1],color='orange',bins=10)

plt.show()
skew = data.skew()

skew
#skewed data

#using box-cox transformation from scipy-stats

bpm = np.asarray(data['Beats.Per.Minute'].values)

bpm_transformed = stats.boxcox(bpm)[0]

energy = np.asarray(data['Energy'].values)

energy_transformed = stats.boxcox(energy)[0]

liveness = np.asarray(data['Liveness'].values)

liveness_transformed = stats.boxcox(liveness)[0]

Acousticness = np.asarray(data['Acousticness..'].values)

Acousticness_transformed = stats.boxcox(Acousticness)[0]

Length = np.asarray(data['Length.'].values)

Length_transformed = stats.boxcox(Length)[0]

Popularity = np.asarray(data['Popularity'].values)

Popularity_transformed = stats.boxcox(Popularity)[0]
Popularity_transformed 
fig,ax = plt.subplots(2,2,figsize=(20,20))

sns.distplot(bpm_transformed,color='red',ax=ax[0][0],bins=10)

sns.distplot(energy_transformed,ax=ax[0][1],color='violet',bins=10)

sns.distplot(liveness_transformed,ax=ax[1][0],color='yellow',bins=10)

sns.distplot(Acousticness_transformed,ax=ax[1][1],color='blue',bins=10)

plt.show()
data = data.drop("Unnamed: 0",axis=1)  #removing the unwanted column
data.head()
plt.figure(figsize=(20,20))

sns.heatmap(data.corr(),fmt=".3f",annot=True,cmap='YlOrBr')

plt.show()
plt.subplots(figsize=(15,8))

sns.boxplot(x='Artist.Name',y='Popularity',data=data)

plt.title("Artist name with respect to Popularity of the song")

plt.xticks(rotation=90)

plt.show()



plt.subplots(figsize=(15,8))

sns.boxplot(x='Artist.Name',y='Loudness..dB..',data=data)

plt.title("Artist name with respect to Loudness of the song")

plt.xticks(rotation=90)

plt.show()



plt.subplots(figsize=(15,8))

sns.boxplot(x='Artist.Name',y='Energy',data=data)

plt.title("Artist name with respect to Energy of the song")

plt.xticks(rotation=90)

plt.show()



plt.subplots(figsize=(15,8))

sns.boxplot(x='Artist.Name',y='Liveness',data=data)

plt.title("Artist name with respect to Liveness of the song")

plt.xticks(rotation=90)

plt.show()



plt.subplots(figsize=(15,8))

sns.boxplot(x='Artist.Name',y='Length.',data=data)

plt.title("Artist name with respect to Length of the song")

plt.xticks(rotation=90)

plt.show()
plt.subplots(figsize=(15,8))

sns.boxplot(x='Genre',y='Popularity',data=data)

plt.title("Genre with respect to Popularity of the song")

plt.xticks(rotation=90)

plt.show()



plt.subplots(figsize=(15,8))

sns.boxplot(x='Genre',y='Loudness..dB..',data=data)

plt.title("Genre with respect to Loudness of the song")

plt.xticks(rotation=90)

plt.show()



plt.subplots(figsize=(15,8))

sns.boxplot(x='Genre',y='Energy',data=data)

plt.title("Genre with respect to Energy of the song")

plt.xticks(rotation=90)

plt.show()





plt.subplots(figsize=(15,8))

sns.boxplot(x='Genre',y='Liveness',data=data)

plt.title("Genre with respect to Liveness of the song")

plt.xticks(rotation=90)

plt.show()



plt.subplots(figsize=(15,8))

sns.boxplot(x='Genre',y='Length.',data=data)

plt.title("Genre with respect to Length of the song")

plt.xticks(rotation=90)

plt.show()
fig = px.histogram(data,x='Danceability',y='Popularity')

fig.show()
fig = px.histogram(data,x='Speechiness.',y='Popularity')

fig.show()
genres = np.unique(data['Genre'])

genres
fig = px.pie(data,names='Artist.Name',values='Popularity', title='Artists with the percentage of songs in the data with popularity of songs of each artist')

fig.show()
fig = px.pie(data,values ='Popularity',names='Genre', title='Genre with Popularity',template='seaborn')

fig.update_traces(rotation=90,pull=0.02,textinfo="percent+label",marker=dict(line=dict(color='#000000', width=2)))

fig.show()