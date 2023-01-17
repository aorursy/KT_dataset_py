import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

plt.rcParams['axes.titlesize'] = 18
stats = pd.read_csv('../input/Seasons_Stats.csv')

stats.info()
stats.head()
stats = stats[['Year',

                'Player',

                'Pos',

                'Age',

                'G',

                'GS',

                'MP',

                'PER',

                'TS%',

                '3PAr',

                'FTr',

                'ORB%',

                'DRB%',

                'TRB%',

                'AST%',

                'STL%',

                'BLK%',

                'TOV%',

                'USG%',

                'OWS',

                'DWS',

                'WS',

                'WS/48',

                'OBPM',

                'DBPM',

                'BPM',

                'VORP',

                'FG',

                'FGA',

                'FG%',

                '3P',

                '3PA',

                '3P%',

                '2P',

                '2PA',

                '2P%',

                'eFG%',

                'FT',

                'FTA',

                'FT%',

                'ORB',

                'DRB',

                'TRB',

                'AST',

                'STL',

                'BLK',

                'TOV',

                'PF',

                'PTS']]

stats.head()
categorical = ['Player', 'Pos']

numerical = []

for c in stats.columns:

    if c not in categorical:

        numerical.append(c)
fig, ax = plt.subplots(1,1, figsize = (10,10))

heatmap = ax.imshow(stats.corr(),

                   cmap='terrain',

                   interpolation='nearest')

plt.xticks(range(stats.corr().shape[1]), stats.corr().columns, rotation=90)

plt.yticks(range(stats.corr().shape[1]), stats.corr().index)



plt.colorbar(heatmap, shrink=0.8);
from sklearn.decomposition import PCA

from sklearn.preprocessing import MinMaxScaler
def create_biplot(df, title):

    pca = PCA()

    scaler = MinMaxScaler()



    data_for_scaling = df.fillna(0)

    data_for_scaling.drop(['Year', 'Age'], axis=1, inplace=True)



    scaled_data = scaler.fit_transform(data_for_scaling)

    transformed_data = pca.fit_transform(scaled_data)



    pc1 = pca.components_[0] # first principle component

    pc2 = pca.components_[1] # second principle component



    fig, ax = plt.subplots(1,1, figsize=(10,10))



    for i in range(len(data_for_scaling.columns)):

        plt.arrow(0, 0,

            pc1[i]*max(transformed_data[:,0]),

            pc2[i]*max(transformed_data[:,1]),

            color='r',

            width=0.005,

            head_width=0.05)

        plt.text(pc1[i]*max(transformed_data[:,0])*1.2,

            pc2[i]*max(transformed_data[:,1])*1.2,

            data_for_scaling.columns[i], color='b')

    plt.scatter(transformed_data[:,0], transformed_data[:,1], c='g', edgecolors=None, alpha=0.1)

    plt.xlim([-1.5, 1.5])

    plt.ylim([-1.5, 1.5])

    plt.xlabel('First Principle Component')

    plt.ylabel('Second Principle Component');

    plt.title(title)

    

    return
create_biplot(stats[numerical], 'Biplot of the Entire Dataset')
sixties_stats = stats[(stats['Year'] >= 1960) & (stats['Year'] < 1970)].copy()

create_biplot(sixties_stats[numerical], 'Biplot of \'Sixties NBA')
seventies_stats = stats[(stats['Year'] >= 1970) & (stats['Year'] < 1980)].copy()

create_biplot(seventies_stats[numerical], 'Biplot of \'Seventies NBA')
eighties_stats = stats[(stats['Year'] >= 1980) & (stats['Year'] < 1990)].copy()

create_biplot(eighties_stats[numerical], 'Biplot of \'Eighties NBA')
nineties_stats = stats[(stats['Year'] >= 1990) & (stats['Year'] < 2000)].copy()

create_biplot(nineties_stats[numerical], 'Biplot of \'Nineties NBA')
noughties_stats = stats[(stats['Year'] >= 2000) & (stats['Year'] < 2010)].copy()

create_biplot(noughties_stats[numerical], 'Biplot of Noughties NBA')
modern_stats = stats[stats['Year'] >= 2000].copy()

create_biplot(modern_stats[numerical], 'Biplot of Modern NBA')