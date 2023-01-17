import numpy as np 
import scipy.spatial.distance as dist
import pandas as pd 
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

plt.figure(figsize=(8,8))
plt.rcParams.update({'font.size': 18})

lables =['L','S']
colours =['r','g']
# Килограммы, Метры
data = np.array([
    [80,1.8],
    [55,1.6]
])
point = np.array([67.5,1.9])

for i, d in enumerate(data):
    plt.scatter(d[0], d[1],color=colours[i],label=f'L2 distance to {lables[i]} = {dist.euclidean(d,point):.4f}')
    plt.annotate(lables[i], (d[0], d[1]),**{'size':20})

plt.scatter(point[0],point[1])
plt.annotate('?', (point[0], point[1]),**{'size':20})
# plt.xlim(50,90) 
# plt.ylim(1.5,2) 
plt.title('Расстояния до центройдов классов размеров (L,S)')
plt.xlabel('Килограммы')
plt.ylabel('Метры')
plt.legend(loc='lower right')
plt.show()
plt.figure(figsize=(8,8))
plt.rcParams.update({'font.size': 18})

scaler = MinMaxScaler()

s_data = scaler.fit_transform(data)
s_point = scaler.transform(point.reshape(-1, 2)).reshape(2)

for i, d in enumerate(s_data):
    plt.scatter(d[0], d[1],color=colours[i],label=f'L2 distance to {lables[i]} = {dist.euclidean(d,s_point):.4}')
    plt.annotate(lables[i], (d[0], d[1]),**{'size':20})

plt.scatter(s_point[0],s_point[1])
plt.annotate('?', (s_point[0], s_point[1]),**{'size':20})
# plt.xlim(50,90) 
# plt.ylim(1.5,2) 
plt.title('Расстояния до центройдов классов размеров (L,S). Scaled')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(loc='lower right')
plt.show()
from sklearn import datasets
#Iris Dataset
iris = datasets.load_iris()
data_titles = ['Petal length','Petal Width','Sepal Length','Sepal Width']
df=pd.DataFrame(iris['data'],columns=data_titles)
df.head()
df.describe()
plt.figure(figsize=(10,8))
plt.rcParams.update({'font.size': 18})
df.boxplot()
plt.show()
scaler = StandardScaler()
df[data_titles] = scaler.fit_transform(df[data_titles])
df.head()
df.describe()
plt.figure(figsize=(10,8))
plt.rcParams.update({'font.size': 18})
df.boxplot()
plt.show()
from sklearn.cluster import KMeans

kmeans_results = []
distances_sums = []
MAX_K = 10
k_values = list(range(1, MAX_K))

for k in k_values:
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(df)
    kmeans_results.append(kmeans.labels_)
    # Sum of squared distances of samples to their closest cluster center.     
    distances_sums.append(kmeans.inertia_)

plt.figure(figsize=(10,8))
plt.rcParams.update({'font.size': 18})
plt.plot(k_values, distances_sums)
plt.xlabel("Количество кластеров")
plt.ylabel("Sum of squared distances")
plt.show()
np.unique(iris['target']).shape[0]
k_means_labels = kmeans_results[2]
df['target'] = k_means_labels
g = sns.pairplot(df,hue='target')
g.fig.suptitle("Результат классификации K-means", y=1.08)
plt.show()
df['target'] = [iris['target_names'][x] for x in iris['target']]
g = sns.pairplot(df,hue='target')
g.fig.suptitle("Настоящие классы",y=1.08)
plt.show()
dist_data = []
X = df[data_titles].to_numpy()
for i in range(X.shape[0]):
    for j in range(i+1,X.shape[0]):
        d = dist.euclidean(X[i],X[j])
        dist_data.append(d)


plt.figure(figsize=(10,8))
plt.rcParams.update({'font.size': 18})

sns.distplot(dist_data);
plt.axvline(1.2,color='r')
plt.xlabel('Density')
plt.ylabel('Euclidean distance')
plt.grid(True)
plt.show()
from  sklearn.neighbors import KDTree

plt.figure(figsize=(10,8))
plt.rcParams.update({'font.size': 18})

k_samples = [5,7,10,13,15]

kd_tree = KDTree(X)

for k_sample in k_samples:
    dist, _ = kd_tree.query(X, k=k_sample, return_distance=True, sort_results=True)  
    # Remove distance to self point
    dist = np.delete(dist, 0, 1)
    sorted_m_dist = np.sort(dist.mean(axis=1))
    plt.plot(np.arange(len(sorted_m_dist)),sorted_m_dist,label=f"k={k_sample}")
    
plt.xlabel('Sample number')
plt.ylabel('Mean distance')
plt.legend()
plt.grid(True)
plt.show()
from sklearn.cluster import DBSCAN

clustering = DBSCAN(eps=0.75, min_samples=5).fit(df[data_titles])
dbscan_labels =clustering.labels_

df['target'] = dbscan_labels
g = sns.pairplot(df,hue='target')
g.fig.suptitle("Результат классификации DBSCAN", y=1.08)
plt.show()
df['target'] = [iris['target_names'][x] for x in iris['target']]
g = sns.pairplot(df,hue='target')
g.fig.suptitle("Настоящие классы",y=1.08)
plt.show()
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=3)
gmm.fit(df[data_titles])
gmm_labels = gmm.predict(df[data_titles])

df['target'] = gmm_labels
g = sns.pairplot(df,hue='target')
g.fig.suptitle("Результат классификации GaussianMixture", y=1.08)
plt.show()
df['target'] = [iris['target_names'][x] for x in iris['target']]
g = sns.pairplot(df,hue='target')
g.fig.suptitle("Настоящие классы",y=1.08)
plt.show()
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score

data = {
    'Метод': [
        'K-means',
        'DBSCAN',
        'GaussianMixture'
    ], 
    'Davies–Bouldin Index': [
        davies_bouldin_score(df[data_titles],k_means_labels),
        davies_bouldin_score(df[data_titles],dbscan_labels),
        davies_bouldin_score(df[data_titles],gmm_labels)
    ],
    'Silhouette score': [
        silhouette_score(df[data_titles],k_means_labels),
        silhouette_score(df[data_titles],dbscan_labels),
        silhouette_score(df[data_titles],gmm_labels)
    ]
}
pd.DataFrame.from_dict(data)
db_index_values = []
silhouette_score_values = []
for x in kmeans_results:
    try:
        db_index_values.append(davies_bouldin_score(df[data_titles],x))
    except:
        db_index_values.append(None)
    try:
        silhouette_score_values.append(silhouette_score(df[data_titles],x))
    except:
        silhouette_score_values.append(None)
    
data = {'K': k_values, 'Davies–Bouldin Index': db_index_values,'Silhouette score':silhouette_score_values}
pd.DataFrame.from_dict(data)
