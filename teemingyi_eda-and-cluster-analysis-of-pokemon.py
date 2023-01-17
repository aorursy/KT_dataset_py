import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
df = pd.read_csv('../input/pokemon.csv')

print(df.shape)

df.info()
df.head()
df.Total.describe()
sns.distplot(df.Total)

plt.show()
sns.distplot(df.HP)

plt.show()
sns.distplot(df.Attack)

plt.show()
sns.distplot(df.Defense)

plt.show()
sns.distplot(df['Sp. Atk'])

plt.show()
sns.distplot(df['Sp. Def'])

plt.show()
sns.distplot(df['Speed'])

plt.show()
df[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']].describe()    
for pkm_type in df['Type 1'].unique():

    df[pkm_type] = df[['Type 1', 'Type 2']].apply(lambda x: 1 if pkm_type in x.values else 0, axis = 1)
def summarise_stats(stats):

    

    summary_stats = []

    for i, pkm_type in enumerate(df['Type 1'].unique()):

        temp = df.loc[df[pkm_type]==1,stats].describe()

        temp.name = pkm_type

        summary_stats.append(temp)



    total_stats_summary = pd.concat(summary_stats, axis=1)

    

    print('Top types by average:')

    print(total_stats_summary.loc['mean'].sort_values(ascending=False).head(3))

    print(total_stats_summary.loc['50%'].sort_values(ascending=False).head(3))

        

    print('\nBottom types by average:')

    print(total_stats_summary.loc['mean'].sort_values(ascending=False).tail(3))

    print(total_stats_summary.loc['50%'].sort_values(ascending=False).tail(3))

    

    print('\nBiggest variance:')

    print(total_stats_summary.loc['std'].sort_values(ascending=False).head(3))

    

    plt_type=total_stats_summary.loc['mean'].sort_values(ascending=False).head(3).index.tolist()+total_stats_summary.loc['mean'].sort_values(ascending=False).tail(3).index.tolist()

    

    for i, pkm_type in enumerate(plt_type):

        sns.distplot(df.loc[df[pkm_type]==1,stats], label=pkm_type, hist=False)

    
summarise_stats('Total')
summarise_stats('HP')
summarise_stats('Attack')
summarise_stats('Defense')
summarise_stats('Sp. Atk')
summarise_stats('Sp. Def')
summarise_stats('Speed')
sns.pairplot(df[['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed']])

plt.show()
from sklearn.cluster import KMeans

from yellowbrick.cluster import KElbowVisualizer



# Instantiate the clustering model and visualizer

model = KMeans(max_iter=1000, random_state=42)

visualizer = KElbowVisualizer(model, k=(2,10), metric='silhouette')



features = ['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed']
X = df[features].values
visualizer.fit(X)    # Fit the data to the visualizer

visualizer.poof()    # Draw/show/poof the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X1 = scaler.fit_transform(X)
visualizer = KElbowVisualizer(model, k=(2,10), metric='silhouette')

visualizer.fit(X1)    # Fit the data to the visualizer

visualizer.poof()    # Draw/show/poof the data
model_kmeans = KMeans(n_clusters=3, max_iter=1000, random_state=42)
df['kmeans_group'] = model_kmeans.fit_predict(X)
cluster_center = pd.DataFrame(model_kmeans.cluster_centers_)

cluster_center.columns = features

cluster_center['total'] = cluster_center.sum(axis=1)



cluster_center['ordered_label'] = cluster_center.total.rank().astype(int)

cluster_center.sort_values(by='ordered_label').set_index('ordered_label')
relabel = cluster_center.ordered_label.to_dict()

df.kmeans_group = df.kmeans_group.map(lambda x: relabel[x])
df.kmeans_group.value_counts()
df.loc[df.kmeans_group==1].sample(10, random_state=42)
df.loc[df.kmeans_group==2].sample(10, random_state=42)
df.loc[df.kmeans_group==3].sample(10, random_state=42)
def bin_value(some_series, bins=3):

    cumsum_series = some_series.value_counts().sort_index().cumsum()

    limits = [len(some_series)/bins*i for i in range(1, bins)]

    right_edge = [abs(cumsum_series-i).idxmin() for i in limits]

    return [sum([x>i for i in right_edge]) for x in some_series]
X2 = df[features].apply(bin_value)
visualizer = KElbowVisualizer(model, k=(2,10), metric='silhouette')

visualizer.fit(X2)    # Fit the data to the visualizer

visualizer.poof()    # Draw/show/poof the data
model_kmeans_bins = KMeans(n_clusters=3, max_iter=1000, random_state=42)
df['kmeans_bin_group'] = model_kmeans_bins.fit_predict(X2)
cluster_center_bins = pd.DataFrame(model_kmeans_bins.cluster_centers_)

cluster_center_bins.columns = features

cluster_center_bins['total'] = cluster_center_bins.sum(axis=1)

cluster_center_bins['ordered_label'] = cluster_center_bins.total.rank().astype(int)



cluster_center_bins.sort_values(by='ordered_label').set_index('ordered_label')
relabel_bins = cluster_center_bins.ordered_label.to_dict()

df.kmeans_bin_group = df.kmeans_bin_group.map(lambda x: relabel_bins[x])
df.kmeans_bin_group.value_counts().sort_index()
df.kmeans_group.value_counts().sort_index()
df['group_combi'] = df.iloc[:,-2:].astype(str).apply(lambda x: ''.join(x), axis=1)
group_count = df.group_combi.value_counts().sort_index()

group_count
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score, adjusted_rand_score, confusion_matrix
print(adjusted_mutual_info_score(df['kmeans_bin_group'], df['kmeans_group']))

print(adjusted_rand_score(df['kmeans_bin_group'], df['kmeans_group']))
confusion_matrix(df['kmeans_bin_group'], df['kmeans_group'])
df.loc[df.group_combi=='13']
df.loc[df.group_combi=='31']
X2_4 = df[features].apply(bin_value, bins=4)
visualizer = KElbowVisualizer(model, k=(2,10), metric='silhouette')

visualizer.fit(X2_4)    # Fit the data to the visualizer

visualizer.poof()    # Draw/show/poof the data
model_kmeans_4bins = KMeans(n_clusters=3, max_iter=1000, random_state=42)

df['kmeans_4bins_group'] = model_kmeans_4bins.fit_predict(X2_4)



cluster_center_4bins = pd.DataFrame(model_kmeans_4bins.cluster_centers_)

cluster_center_4bins.columns = features

cluster_center_4bins['total'] = cluster_center_4bins.sum(axis=1)

cluster_center_4bins['ordered_label'] = cluster_center_4bins.total.rank().astype(int)



cluster_center_4bins.sort_values(by='ordered_label').set_index('ordered_label')
relabel_4bins = cluster_center_4bins.ordered_label.to_dict()

df.kmeans_4bins_group = df.kmeans_4bins_group.map(lambda x: relabel_4bins[x])
df.kmeans_4bins_group.value_counts().sort_index()
print(adjusted_mutual_info_score(df['kmeans_4bins_group'], df['kmeans_group']))

print(adjusted_rand_score(df['kmeans_4bins_group'], df['kmeans_group']))
confusion_matrix(df['kmeans_4bins_group'], df['kmeans_group'])
df.loc[(df.kmeans_group==1)&(df.kmeans_4bins_group==3)]
df.loc[(df.kmeans_group==3)&(df.kmeans_4bins_group==1)]
X2_5 = df[features].apply(bin_value, bins=5)
def build_cluster_bins(X, bins=3):

    

    model = KMeans(n_clusters=3, max_iter=1000, random_state=42)

    df['kmeans_{}bins_group'.format(bins)] = model.fit_predict(X)



    cluster_center_df = pd.DataFrame(model.cluster_centers_)

    cluster_center_df.columns = features

    cluster_center_df['total'] = cluster_center_df.sum(axis=1)

    cluster_center_df['ordered_label'] = cluster_center_df.total.rank().astype(int)

    relabel = cluster_center_df.ordered_label.to_dict()

    df['kmeans_{}bins_group'.format(bins)] = df['kmeans_{}bins_group'.format(bins)].map(lambda x: relabel[x])

    return cluster_center_df.sort_values(by='ordered_label').set_index('ordered_label')
visualizer = KElbowVisualizer(model, k=(2,10), metric='silhouette')

visualizer.fit(X2_5)    # Fit the data to the visualizer

visualizer.poof()    # Draw/show/poof the data
build_cluster_bins(X2_5, bins=5)
print(adjusted_mutual_info_score(df['kmeans_5bins_group'], df['kmeans_group']))

print(adjusted_rand_score(df['kmeans_5bins_group'], df['kmeans_group']))

confusion_matrix(df['kmeans_5bins_group'], df['kmeans_group'])
df.loc[(df.kmeans_group==1)&(df.kmeans_5bins_group==3)]
df.loc[(df.kmeans_group==3)&(df.kmeans_5bins_group==1)]
X2_10 = df[features].apply(bin_value, bins=10)
visualizer = KElbowVisualizer(model, k=(2,10), metric='silhouette')

visualizer.fit(X2_10)    # Fit the data to the visualizer

visualizer.poof()    # Draw/show/poof the data
build_cluster_bins(X2_10, bins=10)
print(adjusted_mutual_info_score(df['kmeans_10bins_group'], df['kmeans_group']))

print(adjusted_rand_score(df['kmeans_10bins_group'], df['kmeans_group']))

confusion_matrix(df['kmeans_10bins_group'], df['kmeans_group'])
df.loc[(df.kmeans_group==1)&(df.kmeans_10bins_group==3)]
df.loc[(df.kmeans_group==3)&(df.kmeans_10bins_group==1)]
df['nunique_group'] = df[[x for x in df.columns if 'bin' in x]].apply(pd.Series.nunique, axis=1)



df.nunique_group.value_counts()
df.loc[df.nunique_group==3]
for x in features:

    visualizer = KElbowVisualizer(model, k=(2,10), metric='silhouette')

    visualizer.fit(df[x].values.reshape(-1,1))    # Fit the data to the visualizer

    print(x)

    visualizer.poof()    # Draw/show/poof the data
num_bins = [5,7,5,3,7,7]

bins_required = {x:num_bins[i] for i, x in enumerate(features)}
bins_stats = {}

stats_cluster_centers = {}

for x in features:

    model_stats = KMeans(n_clusters=bins_required[x], random_state=42)

    bins_stats[x] = model_stats.fit_predict(df[x].values.reshape(-1,1))

    stats_cluster_centers[x] = model_stats.cluster_centers_
stats_relabel = {}

for x in features:

    df1 = pd.DataFrame({x:stats_cluster_centers[x].flatten()})

    df1['ordered_label'] = df1.rank().astype(int)

    print('\n',df1.set_index('ordered_label').sort_index())

    stats_relabel[x] = df1.ordered_label.to_dict()
stats_relabel
temp = pd.DataFrame(bins_stats)

for x in features:

    temp[x] = temp[x].map(lambda y: stats_relabel[x][y])

    

temp.head()
for x in features:

    print('\n',temp[x].value_counts().sort_index())
X3 = temp
visualizer = KElbowVisualizer(model, k=(2,10), metric='silhouette')

visualizer.fit(X3)    # Fit the data to the visualizer

visualizer.poof()    # Draw/show/poof the data
model_kmeans_bins_km = KMeans(n_clusters=3, max_iter=1000, random_state=42)



df['kmeans_kmbin_group'] = model_kmeans_bins_km.fit_predict(X3)



cluster_center_bins_km = pd.DataFrame(model_kmeans_bins_km.cluster_centers_)

cluster_center_bins_km.columns = features

cluster_center_bins_km['total'] = cluster_center_bins_km.sum(axis=1)

cluster_center_bins_km['ordered_label'] = cluster_center_bins_km.total.rank().astype(int)



cluster_center_bins_km.sort_values(by='ordered_label').set_index('ordered_label')
relabel_bins_km = cluster_center_bins_km.ordered_label.to_dict()

df.kmeans_kmbin_group = df.kmeans_kmbin_group.map(lambda x: relabel_bins_km[x])
df.kmeans_kmbin_group.value_counts().sort_index()
print(adjusted_mutual_info_score(df['kmeans_kmbin_group'], df['kmeans_group']))

print(adjusted_rand_score(df['kmeans_kmbin_group'], df['kmeans_group']))
confusion_matrix(df['kmeans_kmbin_group'], df['kmeans_group'])
def build_cluster_n(X, n_clust=3):

    

    model = KMeans(n_clusters=n_clust, max_iter=1000, random_state=42)

    df['kmeans_{}_group'.format(n_clust)] = model.fit_predict(X)



    cluster_center_df = pd.DataFrame(model.cluster_centers_)

    cluster_center_df.columns = features

    cluster_center_df['total'] = cluster_center_df.sum(axis=1)

    cluster_center_df['ordered_label'] = cluster_center_df.total.rank().astype(int)

    relabel = cluster_center_df.ordered_label.to_dict()

    df['kmeans_{}_group'.format(n_clust)] = df['kmeans_{}_group'.format(n_clust)].map(lambda x: relabel[x])

    return cluster_center_df.sort_values(by='ordered_label').set_index('ordered_label')
build_cluster_n(X)
result = {}

for n_clust in [9, 11, 14, 17, 19]:

    result[n_clust] = build_cluster_n(X, n_clust=n_clust)
from sklearn.metrics import silhouette_samples
for n_clust in [9, 11, 14, 17, 19]:

    df['silhouette_{}'.format(n_clust)] = silhouette_samples(X, df['kmeans_{}_group'.format(n_clust)])
df.groupby('kmeans_9_group').silhouette_9.mean()
df.loc[df.kmeans_9_group==9,['Name','silhouette_9']].sort_values(by='silhouette_9')