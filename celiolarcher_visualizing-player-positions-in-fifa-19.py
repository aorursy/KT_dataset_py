# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



dataset=pd.read_csv("../input/data.csv")

# Any results you write to the current directory are saved as output.
dataset['Value']=dataset['Value'].str.replace("€","")

dataset['Value']=dataset['Value'].replace(r'[K|M]','',regex=True).astype(float)*dataset['Value'].str.extract(r'[\d\.]+([KM]+)', expand=False).fillna(1).replace(['K','M'], [10**3, 10**6]).astype(int)

dataset=dataset.fillna(0)
from sklearn.cluster import KMeans,DBSCAN

from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer



def grouping_playes(data, n_clusters, attr=['Crossing','Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling','Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',

       'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower','Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression','Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',

       'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',

       'GKKicking', 'GKPositioning', 'GKReflexes']):

    

    subset=data[attr]

    subset=Normalizer().fit_transform(subset)

    subset=MinMaxScaler().fit_transform(subset)

    model=KMeans(n_clusters=n_clusters, random_state=42)

    model.fit(subset)

    return model.predict(subset)



dataset['cat']=pd.Series(grouping_playes(dataset,5)).apply(lambda x:"Class {}".format(x)).values



best_players=dataset[dataset['Overall']>88]



_, ax=plt.subplots(figsize=(15,12))

fig=sns.scatterplot(x='Overall',y='Value',data=best_players,hue='cat',ax=ax)



for x,y,val in zip(best_players['Overall'],best_players['Value'],best_players["Name"]):

    fig.text(x+0.02, y, str(val))



plt.show()
def plot_players(data_in, attr=['Crossing','Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling','Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',

       'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower','Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression','Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',       'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',

       'GKKicking', 'GKPositioning', 'GKReflexes']):



    from sklearn.manifold import TSNE



    subset=data_in[attr]

    subset=Normalizer().fit_transform(subset)

    subset=StandardScaler().fit_transform(subset)

    model=TSNE(n_components=2,random_state=42)

    subset=model.fit_transform(subset)

    

    ax=plt.subplots(figsize=(15,10))

    fig=sns.scatterplot(subset[:,0],subset[:,1],hue=data_in['cat'])

    

    for x,y,val in zip(subset[:,0],subset[:,1],data_in["Name"]):

        fig.text(x+0.02, y, str(val))



    plt.show()

    return 

    



plot_players(best_players)
best_players_85=dataset[dataset['Overall']>85]

plot_players(best_players_85)
CF=dataset[dataset['Position'].isin(['CF','RS','LS','ST','RF','LF'])]

CF['cat']=pd.Series(grouping_playes(CF,4)).apply(lambda x:"Class {}".format(x)).values



CF=CF[CF['Overall']>80]



plot_players(CF)