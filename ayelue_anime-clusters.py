import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.utils import shuffle

from sklearn.cluster import MiniBatchKMeans

from sklearn.metrics import silhouette_score

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

import plotly as py

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot



%matplotlib inline



init_notebook_mode(connected=True)
df_anime = pd.read_csv('../input/anime-recommendations-database/anime.csv')
df_anime
df_rating = pd.read_csv('../input/anime-recommendations-database/rating.csv')
df_rating
df_anime.isna().sum(), df_anime.isnull().sum()
df_anime = df_anime.dropna()
df_anime.isna().sum(), df_anime.isnull().sum()
df_rating.isna().sum(), df_rating.isnull().sum()
unknown_index = df_anime[df_anime['episodes']=='Unknown'].index.to_list()
df_anime.loc[unknown_index,'episodes'] = 0

df_anime[df_anime['episodes']==0]
df_anime['episodes'] = df_anime['episodes'].astype('int')
df_anime['episodes'].describe()
df_anime.loc[unknown_index,'episodes'] = df_anime['episodes'].median()
df_anime['episodes'].describe()
df_anime
len(df_anime['anime_id'].unique())
#first of all get all genres 

all_genres = ''

for genre in df_anime['genre'].to_list():

    all_genres += str(genre) + ', '
all_genres = all_genres.split(',')
all_genres = list(map(lambda x: x.strip() ,all_genres))
all_genres = set(all_genres)
all_genres.remove('')
for genre_name in all_genres:

    list_code = list(map(lambda x: 1 if x.find(genre_name)+1 else 0,df_anime['genre'].to_list()))

    df_anime.loc[:,'genre_%s'%genre_name] =list_code 
df_anime = df_anime.drop('genre',axis=1)
count_genres = {genre: df_anime['genre_%s'%genre].sum() for genre in all_genres}
count_genres = {key: value for key, value in sorted(count_genres.items(),key=lambda x: x[1],reverse=True)}
x = list(count_genres.keys())[:10]

y = [count_genres[key] for key in x]
plt.figure(figsize=(10,10))

sns.barplot(x=x,y=y)


rate_genres = [(genre,df_anime[df_anime['genre_%s'%genre]==1]['rating'].mean()) for genre in all_genres]
rate_genres = sorted(rate_genres, key=lambda x: x[1], reverse=True)
len(rate_genres)
plt.figure(figsize=(11,10))

plt.xlabel('Genre')

plt.ylabel('Mean rating')

sns.barplot(x=list(map(lambda x: x[0],rate_genres[:10])), y=list(map(lambda x: x[1],rate_genres[:10])))
values = df_anime['type'].value_counts()
labels = values.index.to_list()

values = values.to_list()

values,labels
plt.figure(figsize=(10,10))

plt.pie(values, labels=labels,autopct='%1.1f%%')

_ = plt.legend(labels)
rate_type = [(type_name, df_anime[df_anime['type']==type_name]['rating'].mean()) for type_name in df_anime['type'].unique()]
rate_type = sorted(rate_type, key=lambda x: x[1],reverse=True)
plt.figure(figsize=(10,10))

plt.xlabel('type')

plt.ylabel('mean rating')

sns.barplot(x=list(map(lambda x: x[0],rate_type)), y=list(map(lambda x: x[1],rate_type)))
sns.violinplot(df_anime['rating'])
df_anime['rating'].describe()
top5 = df_anime.sort_values(by=['members'], ascending=False)[:5]

down5 = df_anime.sort_values(by=['members'], ascending=False)[-5:]
plt.figure(figsize=(10,10))

a = sns.barplot(x=top5['name'],y=top5['members'])

_ = plt.xticks(a.get_xticks(), rotation=90)
plt.figure(figsize=(10,10))

a = sns.barplot(x=top5['name'],y=top5['rating'])

_ = plt.xticks(a.get_xticks(), rotation=90)
plt.figure(figsize=(10,10))

a = sns.barplot(x=down5['name'],y=down5['members'])

_ = plt.xticks(a.get_xticks(), rotation=90)
plt.figure(figsize=(10,10))

a = sns.barplot(x=down5['name'],y=down5['rating'])

_ = plt.xticks(a.get_xticks(), rotation=90)
df_rating
#find number of unique user

len(df_rating['user_id'].unique())
#find number of unique anime

len(df_rating['anime_id'].unique())
df_rating['mean_rating'] = df_rating.groupby('user_id')['rating'].transform('mean')

df_rating
a = df_rating[df_rating['rating']>=df_rating['mean_rating']].apply(lambda x: 1,axis=1)
index_liked = a.index.to_list()
df_rating_liked = df_rating.iloc[index_liked,:]


df_rating_liked = df_rating_liked.drop(['rating','mean_rating'], axis=1)
df_rating_liked
anime_index = {df_anime.loc[idx,'anime_id']:idx for idx in df_anime.index}
df_anime_clusterize = df_anime.drop(['name','anime_id'],axis=1)
df_anime_clusterize = pd.get_dummies(df_anime_clusterize)
num_cols= df_anime_clusterize[['episodes','rating','members']]

cat_cols = df_anime_clusterize.drop(['episodes','rating','members'], axis=1)
scaler = StandardScaler()
num_cols = pd.DataFrame(scaler.fit_transform(num_cols))
num_cols.columns = ['episodes_scale','rating_scale','members_scale']
df_anime_clusterize = pd.concat([num_cols, cat_cols], axis=1, join='inner')
scores = []

inertia_list = np.empty(11)



for i in range(2,11):

    print(i)

    kmeans = MiniBatchKMeans(n_clusters=i, batch_size=50)

    kmeans.fit(df_anime_clusterize)

    inertia_list[i] = kmeans.inertia_

    scores.append(silhouette_score(df_anime_clusterize, kmeans.labels_))




plt.plot(range(0,11),inertia_list,'-o')

plt.xlabel('Number of cluster')

plt.axvline(x=4, color='blue', linestyle='--')

plt.ylabel('Inertia')

plt.show()







plt.plot(range(2,11), scores);

plt.title('Results KMeans')

plt.xlabel('n_clusters');

plt.axvline(x=4, color='blue', linestyle='--')

plt.ylabel('Silhouette Score');

plt.show()



kmeans =  MiniBatchKMeans(n_clusters=4,batch_size=40)

kmeans = kmeans.fit(df_anime_clusterize)

clusters = kmeans.predict(df_anime_clusterize)

df_anime_clusterize['cluster'] = clusters

df_anime_clusterize['cluster'].value_counts()
plot_df = pd.DataFrame(np.array(df_anime_clusterize.sample(4000)))

plot_df.columns = df_anime_clusterize.columns
perplexity = 30
tsne_2d = TSNE(n_components=2, perplexity=perplexity)



tsne_3d = TSNE(n_components=3, perplexity=perplexity)
TCs_2d = pd.DataFrame(tsne_2d.fit_transform(plot_df.drop(["cluster"], axis=1)))

TCs_3d = pd.DataFrame(tsne_3d.fit_transform(plot_df.drop(["cluster"], axis=1)))
TCs_2d.columns = ["TC1_2d","TC2_2d"]



TCs_3d.columns = ["TC1_3d","TC2_3d","TC3_3d"]
plot_df = pd.concat([plot_df,TCs_2d,TCs_3d], axis=1, join='inner')
plot_df["1d_y"] = 0
clusters = {}

for cluster_label in plot_df['cluster'].unique():

    clusters[cluster_label] = plot_df[plot_df["cluster"] == cluster_label]
data = []

for key in clusters.keys():

    data.append(go.Scatter(

                    x = clusters[key]["TC1_2d"],

                    y = clusters[key]["TC2_2d"],

                    mode = "markers",

                    name = "Cluster %s"%key,

                    text = None))



title = "Visualizing Clusters in Two Dimensions Using T-SNE (perplexity=" + str(perplexity) + ")"



layout = dict(title = title,

              xaxis= dict(title= 'TC1',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'TC2',ticklen= 5,zeroline= False)

             )



fig = dict(data = data, layout = layout)



iplot(fig)
data = []

for key in clusters.keys():

    data.append(go.Scatter3d(

                    x = clusters[key]["TC1_3d"],

                    y = clusters[key]["TC2_3d"],

                    z = clusters[key]["TC3_3d"],

                    mode = "markers",

                    name = "Cluster %s"%key,

                    text = None))





title = "Visualizing Clusters in Three Dimensions Using T-SNE (perplexity=" + str(perplexity) + ")"



layout = dict(title = title,

              xaxis= dict(title= 'TC1',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'TC2',ticklen= 5,zeroline= False)

             )

plt.figure(figsize=(20,20))

fig = dict(data = data, layout = layout)



iplot(fig)
anime_clusters = {i: [] for i in range(4)}

for anime_id, c_pred in zip(df_anime['anime_id'], df_anime_clusterize['cluster']):

    anime_clusters[c_pred] +=[anime_id]
def find_user_centroid(data):

    data = data[data['cluster']==data['cluster'].mode()[0]]

    data = data.drop(['user_id','anime_id','cluster'], axis=1,errors='ignore')

    return pd.DataFrame(data.mean(axis=0)).T
data = df_rating_liked[:100000]

grouped = data.groupby('user_id')
train_data = {'user_id': [],'anime_id': []}

test_data = {'user_id': [],'anime_id': []}

for name,group in grouped:

    if len(group)>1:

        

        train, test = train_test_split(group['anime_id'],test_size=0.2,random_state=42)



        train_data['user_id']+=[name for _ in range(len(train))]

        train_data['anime_id']+= list(train)



        test_data['user_id']+=[name for _ in range(len(test))]

        test_data['anime_id']+= list(test)

    

    
len(train_data['user_id']),len(test_data['user_id'])
df_train = pd.DataFrame(train_data)

df_test = pd.DataFrame(test_data)

df_train = df_train.join(df_anime_clusterize, how='inner')
train_centroids = pd.DataFrame(columns = ['user_id']+list(df_anime_clusterize.columns))



for name,group in df_train.groupby('user_id'):

    user_centroid = find_user_centroid(group)

    user_centroid['user_id'] = name

    user_centroid['cluster'] = group['cluster'].mode()[0]

    train_centroids = train_centroids.append(user_centroid,ignore_index=True)

#     print(group['cluster'])

train_centroids
result = {}

for user_id in train_centroids['user_id'][:10]:

    print('User id %s'%user_id)

    user = train_centroids[train_centroids['user_id']==user_id]

    result_dist = []

   

    for anime_id in anime_clusters[user['cluster'].iloc[0]]:

        #iterate by all points in cluster. find 10 closer points to user centroid

        

        anime_point = df_anime_clusterize.loc[anime_index[anime_id],:].drop('cluster').to_numpy()

        

        result_dist.append((anime_id, np.linalg.norm(user.drop(['cluster','user_id'],axis=1)-anime_point)))

        

    

    result[user_id] = sorted(result_dist,key=lambda x: x[1])[:10]
test_data = pd.DataFrame(test_data)
error_recom = {}

for user_id in list(result.keys())[:10]:

    test_centroid = find_user_centroid(test_data[test_data['user_id']==user_id].join(df_anime_clusterize,how='inner'))

    index = list(map(lambda x: anime_index[x[0]],result[user_id]))

    result_centroid = find_user_centroid(df_anime_clusterize.iloc[index,:])

    error_recom[user_id]  = np.linalg.norm(test_centroid-result_centroid)
error_recom