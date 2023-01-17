import numpy as np

import pandas as pd



df = pd.read_csv('../input/israeli-elections-2015-2013/votes per booth 2019b.csv', encoding='iso-8859-8')

df.head()
new_supervision_fields = ['ברזל', 'סמל ועדה', 'שופט', 'ריכוז']

metadata_columns = ['בזב', 'מצביעים', 'פסולים', 'כשרים']



df.drop(new_supervision_fields+metadata_columns+['סמל ישוב'], axis=1, inplace=True)



df.set_index(['שם ישוב', 'קלפי'], inplace=True)

df.head()
print('dropping parties with zero votes: ' + str(df.columns[df.sum()==0]))

df.drop(df.columns[df.sum()==0], axis=1, inplace=True)



print('dropping ballots with zero legit votes: ' +str(df.index[df.sum(axis=1)==0]))

df.drop(df.index[df.sum(axis=1)==0], axis=0, inplace=True)
party_sizes = df.sum().div(df.sum().sum())

print(party_sizes.sort_values(ascending=False).round(2))

major_parties = party_sizes.index[party_sizes>0.01].to_list()
import sklearn

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

import matplotlib.pyplot as plt



normalized_votes=df.div(df.sum(axis=1), axis=0) 



pca=PCA(n_components=2)

K=7; 

k_means = KMeans(n_clusters=K, max_iter=100).fit(normalized_votes)



transformed_votes=pca.fit_transform(normalized_votes)

cluster_centers = pca.transform(k_means.cluster_centers_)
df_for_plot = pd.DataFrame(transformed_votes, columns=['x','y']).join(pd.Series(k_means.labels_, name='label'))

ax = df_for_plot.plot.scatter('x','y',c='label', figsize=(16,12), cmap='jet', style='.', s=3, grid=True)



pd.DataFrame(cluster_centers).plot.scatter(0,1,ax=ax, c='black', s=60, grid=True)



for cluster_label, cluster_center in enumerate(cluster_centers):

    plt.text(cluster_center[0], cluster_center[1] ,r' Cluster '+str(cluster_label), fontsize=16)
cluster_sizes = pd.DataFrame.from_dict({'votes': df.sum(axis=1), 'cluster': k_means.labels_}).groupby('cluster').sum()

cluster_sizes = cluster_sizes.div(cluster_sizes.sum()).round(2)



clusters_df = pd.DataFrame(columns = df.columns, data=k_means.cluster_centers_)

clusters_df['size'] = cluster_sizes

clusters_df[['size'] + major_parties].round(2)
import networkx as nx

from community import best_partition # --> http://perso.crans.org/aynaud/communities/



G=nx.Graph(normalized_votes.corr()>0)

G=nx.relabel_nodes(G,dict(zip(G.nodes(),[x[::-1] for x in normalized_votes.columns])))

communities=best_partition(G)

num_communities = len(set(communities.values()))



community_colors={x: x/(num_communities-1) for x in range(num_communities)}

node_coloring=[community_colors[communities[node]] for node in G.nodes()]



nx.pos=nx.fruchterman_reingold_layout(G, dim=2, k=None, pos=None, fixed=None, iterations=5000, weight='weight', scale=1.0, center=None)



nx.draw_networkx(G, cmap=plt.get_cmap('jet'), with_labels=True, node_color=node_coloring,font_size=16)

plt.title('Party correlation network and community partition (all parties)', fontsize=22)

fig = plt.gcf()

fig.set_size_inches(18.5, 10.5)
major_party_votes=normalized_votes[major_parties]



G=nx.Graph(major_party_votes.corr()>0)

G=nx.relabel_nodes(G,dict(zip(G.nodes(),[x[::-1] for x in major_parties])))

communities=best_partition(G)



community_colors={0:0,1:0.2,2:0.5,3:0.7,4:0.9}

node_coloring=[community_colors[communities[node]] for node in G.nodes()]



nx.pos=nx.fruchterman_reingold_layout(G, dim=2, k=None, pos=None, fixed=None, iterations=5000, weight='weight', scale=1.0, center=None)



nx.draw_networkx(G, cmap=plt.get_cmap('jet'), with_labels=True, node_color=node_coloring,font_size=30)

plt.title('Party correlation network and community partition (major parties)', fontsize=22)

fig = plt.gcf()

fig.set_size_inches(18.5, 10.5)