import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import os

import seaborn as sns

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

plt.style.use('fivethirtyeight')
path = "../input/"

os.chdir(path)

filenames = os.listdir(path)

df = pd.DataFrame()

for filename in sorted(filenames):

    try:

        read_filename = '../input/' + filename

        temp = pd.read_csv(read_filename,encoding='utf8')

        frame = [df,temp]

        df = pd.concat(frame)

    except UnicodeDecodeError:

        pass
df['Aces'] = df.l_ace + df.w_ace



plt.bar(1,np.mean(df.Aces[df.surface == 'Hard']))

plt.bar(2,np.mean(df.Aces[df.surface == 'Grass']), color = 'g')

plt.bar(3,np.mean(df.Aces[df.surface == 'Clay']), color ='r')

plt.ylabel('Aces per Match')

plt.xticks([1,2,3], ['Hard','Grass','Clay'])

plt.title('More Aces on Grass')
df['loser_1st_rate'] = np.true_divide(df['l_1stIn'], df['l_svpt'])

df['winner_1st_rate'] = np.true_divide(df['w_1stIn'], df['w_svpt'])

df['first_serve_rate'] = (df['loser_1st_rate'] + df['winner_1st_rate'])/2



plt.bar(1,100*np.mean(df.first_serve_rate[df.surface == 'Hard']))

plt.bar(2,100*np.mean(df.first_serve_rate[df.surface == 'Grass']), color = 'g')

plt.bar(3,100*np.mean(df.first_serve_rate[df.surface == 'Clay']), color ='r')

plt.ylabel('First Serve In [%]')

plt.ylim([50,70])

plt.xticks([1,2,3], ['Hard','Grass','Clay'])

plt.title('% of 1st serves in')



plt.figure()

df['loser_1st_taken'] =  np.true_divide(df['l_1stWon'], df['l_1stIn'])

df['winner_1st_taken'] =  np.true_divide(df['w_1stWon'], df['w_1stIn'])

df['first_taken'] = (df['loser_1st_taken'] + df['winner_1st_taken'])/2



plt.bar(1,100*np.mean(df.first_taken[df.surface == 'Hard']))

plt.bar(2,100*np.mean(df.first_taken[df.surface == 'Grass']), color = 'g')

plt.bar(3,100*np.mean(df.first_taken[df.surface == 'Clay']), color ='r')

plt.ylabel('First Serve point taken')

plt.title('% of first serve points taken')

plt.xticks([1,2,3], ['Hard','Grass','Clay'])

plt.ylim([50,70])
df['aces_per_serve_w'] = np.true_divide(df.w_ace,df.w_svpt)

df['aces_per_serve_l'] = np.true_divide(df.l_ace,df.l_svpt)

df['aces_per_serve'] = (df['aces_per_serve_w'] + df['aces_per_serve_l'])/2



plt.bar(1,np.mean(df['aces_per_serve'][df.surface == 'Hard']))

plt.bar(2,np.mean(df['aces_per_serve'][df.surface == 'Grass']), color = 'g')

plt.bar(3,np.mean(df['aces_per_serve'][df.surface == 'Clay']), color = 'r')

plt.ylabel('Aces Per Serve')

plt.title('Aces Per Serve')

plt.xticks([1,2,3], ['Hard','Grass','Clay'])
winners = list(np.unique(df.winner_name))

losers = list(np.unique(df.loser_name))



all_players = winners + losers

players = np.unique(all_players)



players_df = pd.DataFrame()

players_df['Name'] = players

players_df['Wins'] = players_df.Name.apply(lambda x: len(df[df.winner_name == x]))

players_df['Losses'] = players_df.Name.apply(lambda x: len(df[df.loser_name == x]))

players_df['PCT'] = np.true_divide(players_df.Wins,players_df.Wins + players_df.Losses)

players_df['Games'] = players_df.Wins + players_df.Losses



surfaces = ['Hard','Grass','Clay','Carpet']

for surface in surfaces:

    players_df[surface + '_wins'] = players_df.Name.apply(lambda x: len(df[(df.winner_name == x) & (df.surface == surface)]))

    players_df[surface + '_losses'] = players_df.Name.apply(lambda x: len(df[(df.loser_name == x) & (df.surface == surface)]))

    players_df[surface + 'PCT'] = np.true_divide(players_df[surface + '_wins'],players_df[surface + '_losses'] + players_df[surface + '_wins'])

    

serious_players = players_df[players_df.Games>40]

serious_players['Height'] = serious_players.Name.apply(lambda x: list(df.winner_ht[df.winner_name == x])[0])

serious_players['Best_Rank'] = serious_players.Name.apply(lambda x: min(df.winner_rank[df.winner_name == x]))

serious_players['Win_Aces'] = serious_players.Name.apply(lambda x: np.mean(df.w_ace[df.winner_name == x]))

serious_players['Lose_Aces'] = serious_players.Name.apply(lambda x: np.mean(df.l_ace[df.loser_name == x]))

serious_players['Aces'] = (serious_players['Win_Aces']*serious_players['Wins'] + serious_players['Lose_Aces']*serious_players['Losses'])/serious_players['Games']

serious_players = serious_players[np.isnan(serious_players.GrassPCT) == 0]

serious_players = serious_players[np.isnan(serious_players.ClayPCT) == 0]

kmeans_df = serious_players[['HardPCT','GrassPCT','ClayPCT']][np.isnan(serious_players.GrassPCT) == 0]



kmeans = KMeans(n_clusters = 6, random_state = 0).fit(kmeans_df)

kmeans.cluster_centers_



serious_players['label'] = kmeans.labels_

print(['Hard','Grass','Clay'])

for i,label in enumerate(kmeans.cluster_centers_):

    print(label)

    print(np.mean(serious_players.Height[serious_players.label == i]))





for i,cluster in enumerate(kmeans.cluster_centers_):

    plt.bar(i+1-0.25,100*cluster[0], width = 0.25, color = 'b')

    plt.bar(i+1,100*cluster[1],width = 0.25, color = 'g')

    plt.bar(i+1+0.25,100*cluster[2],width = 0.25, color = 'r')



plt.legend(['Hard','Grass','Clay'], loc = 2, fontsize = 10)

plt.ylabel('Winning Percentage')

plt.xlabel('Cluster')

plt.title('6 Clusters of players')



pca = PCA(2)

pca.fit(kmeans_df)

pca_df = pca.transform(kmeans_df)

pca_df = pd.DataFrame(pca_df)

pca_df['label'] = kmeans.labels_



plt.figure()

legend = []



for i,label in enumerate(kmeans.cluster_centers_):

    plt.plot(pca_df[pca_df.columns[0]][pca_df.label == i],pca_df[pca_df.columns[1]][pca_df.label == i],'o')

    legend.append('Cluster ' + str(i+1))

    

plt.legend(legend)

plt.xlabel('PCA 1')

plt.ylabel('PCA 2')

plt.title('PCA 2D view')