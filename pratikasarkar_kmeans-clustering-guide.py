import numpy as np

import pandas as pd

import seaborn as sns
df = pd.DataFrame({'ID': ['s1','s2','s3','s4','s5','s6','s7','s8','s9'],

                   'X1': [3,6,4.5,6.1,8.8,9.1,10,11.8,13],

                   'X2': [5,5.2,6,8,7,7.9,5.8,4.2,5.7]})

df
sns.scatterplot(df['X1'],df['X2'])
c1 = ( 4.5 , 6 )

c2 = ( 8.8 , 7 )

print('Group1')

print('Iteration1')

df_dist = pd.DataFrame({'ID': ['s1','s2','s3','s4','s5','s6','s7','s8','s9'],

                        'Dist_c1':np.sqrt(((df['X1']-c1[0])**2) + ((df['X2']-c1[1])**2)),

                        'Dist_c2':np.sqrt(((df['X1']-c2[0])**2) + ((df['X2']-c2[1])**2)),

                        'cluster':['c1','c1','c1','c1','c2','c2','c2','c2','c2']})

print('Centroids : ',c1,c2)

print(df_dist)

print()

print('Iteration2')

c1 = (df.loc[list(df_dist[df_dist['cluster'] == 'c1'].index),:]['X1'].mean(),df.loc[list(df_dist[df_dist['cluster'] == 'c1'].index),:]['X2'].mean())

c2 = (df.loc[list(df_dist[df_dist['cluster'] == 'c2'].index),:]['X1'].mean(),df.loc[list(df_dist[df_dist['cluster'] == 'c2'].index),:]['X2'].mean())

print('Centroids : ',c1,c2)

df_dist = pd.DataFrame({'ID': ['s1','s2','s3','s4','s5','s6','s7','s8','s9'],

                        'Dist_c1':np.sqrt(((df['X1']-c1[0])**2) + ((df['X2']-c1[1])**2)),

                        'Dist_c2':np.sqrt(((df['X1']-c2[0])**2) + ((df['X2']-c2[1])**2)),

                        'cluster':['c1','c1','c1','c1','c2','c2','c2','c2','c2']})

print(df_dist)

print()

print('Iteration3')

c1 = (df.loc[list(df_dist[df_dist['cluster'] == 'c1'].index),:]['X1'].mean(),df.loc[list(df_dist[df_dist['cluster'] == 'c1'].index),:]['X2'].mean())

c2 = (df.loc[list(df_dist[df_dist['cluster'] == 'c2'].index),:]['X1'].mean(),df.loc[list(df_dist[df_dist['cluster'] == 'c2'].index),:]['X2'].mean())

print('Centroids : ',c1,c2)

print('As centroid value does not change after Iteration 2, the algorithm stops here...')

Inertia_c1 = np.sum((np.sqrt(((df.loc[list(df_dist[df_dist['cluster'] == 'c1'].index),:]['X1'] - c1[0])**2) + ((df.loc[list(df_dist[df_dist['cluster'] == 'c1'].index),:]['X2'] - c1[1])**2)))**2)

Inertia_c2 = np.sum((np.sqrt(((df.loc[list(df_dist[df_dist['cluster'] == 'c2'].index),:]['X1'] - c2[0])**2) + ((df.loc[list(df_dist[df_dist['cluster'] == 'c2'].index),:]['X2'] - c2[1])**2)))**2)

total_inertia = Inertia_c1 + Inertia_c2

print('C1 Inertia : ',Inertia_c1)

print('C2 Inertia : ',Inertia_c2)

print("Total Inertia : ",total_inertia)
c1 = ( 6 , 5.2 )

c2 = ( 6.1 , 8 )

print('Group2')

print('Iteration1')

df_dist = pd.DataFrame({'ID': ['s1','s2','s3','s4','s5','s6','s7','s8','s9'],

                        'Dist_c1':np.sqrt(((df['X1']-c1[0])**2) + ((df['X2']-c1[1])**2)),

                        'Dist_c2':np.sqrt(((df['X1']-c2[0])**2) + ((df['X2']-c2[1])**2)),

                        'cluster':['c1','c1','c1','c2','c2','c2','c1','c1','c1']})

print('Centroids : ',c1,c2)

print(df_dist)

print()

print('Iteration2')

c1 = (df.loc[list(df_dist[df_dist['cluster'] == 'c1'].index),:]['X1'].mean(),df.loc[list(df_dist[df_dist['cluster'] == 'c1'].index),:]['X2'].mean())

c2 = (df.loc[list(df_dist[df_dist['cluster'] == 'c2'].index),:]['X1'].mean(),df.loc[list(df_dist[df_dist['cluster'] == 'c2'].index),:]['X2'].mean())

print('Centroids : ',c1,c2)

df_dist = pd.DataFrame({'ID': ['s1','s2','s3','s4','s5','s6','s7','s8','s9'],

                        'Dist_c1':np.sqrt(((df['X1']-c1[0])**2) + ((df['X2']-c1[1])**2)),

                        'Dist_c2':np.sqrt(((df['X1']-c2[0])**2) + ((df['X2']-c2[1])**2)),

                        'cluster':['c1','c1','c1','c2','c2','c2','c1','c1','c1']})

print(df_dist)

print()

print('Iteration3')

c1 = (df.loc[list(df_dist[df_dist['cluster'] == 'c1'].index),:]['X1'].mean(),df.loc[list(df_dist[df_dist['cluster'] == 'c1'].index),:]['X2'].mean())

c2 = (df.loc[list(df_dist[df_dist['cluster'] == 'c2'].index),:]['X1'].mean(),df.loc[list(df_dist[df_dist['cluster'] == 'c2'].index),:]['X2'].mean())

print('Centroids : ',c1,c2)

print('As centroid value does not change after Iteration 2, the algorithm stops here...')

Inertia_c1 = np.sum((np.sqrt(((df.loc[list(df_dist[df_dist['cluster'] == 'c1'].index),:]['X1'] - c1[0])**2) + ((df.loc[list(df_dist[df_dist['cluster'] == 'c1'].index),:]['X2'] - c1[1])**2)))**2)

Inertia_c2 = np.sum((np.sqrt(((df.loc[list(df_dist[df_dist['cluster'] == 'c2'].index),:]['X1'] - c2[0])**2) + ((df.loc[list(df_dist[df_dist['cluster'] == 'c2'].index),:]['X2'] - c2[1])**2)))**2)

total_inertia = Inertia_c1 + Inertia_c2

print('C1 Inertia : ',Inertia_c1)

print('C2 Inertia : ',Inertia_c2)

print("Total Inertia : ",total_inertia)
c1 = ( 4.5 , 6 )

c2 = ( 10 , 5.8 )

print('Group3')

print('Iteration1')

df_dist = pd.DataFrame({'ID': ['s1','s2','s3','s4','s5','s6','s7','s8','s9'],

                        'Dist_c1':np.sqrt(((df['X1']-c1[0])**2) + ((df['X2']-c1[1])**2)),

                        'Dist_c2':np.sqrt(((df['X1']-c2[0])**2) + ((df['X2']-c2[1])**2)),

                        'cluster':['c1','c1','c1','c1','c2','c2','c2','c2','c2']})

print('Centroids : ',c1,c2)

print(df_dist)

print()

print('Iteration2')

c1 = (df.loc[list(df_dist[df_dist['cluster'] == 'c1'].index),:]['X1'].mean(),df.loc[list(df_dist[df_dist['cluster'] == 'c1'].index),:]['X2'].mean())

c2 = (df.loc[list(df_dist[df_dist['cluster'] == 'c2'].index),:]['X1'].mean(),df.loc[list(df_dist[df_dist['cluster'] == 'c2'].index),:]['X2'].mean())

print('Centroids : ',c1,c2)

df_dist = pd.DataFrame({'ID': ['s1','s2','s3','s4','s5','s6','s7','s8','s9'],

                        'Dist_c1':np.sqrt(((df['X1']-c1[0])**2) + ((df['X2']-c1[1])**2)),

                        'Dist_c2':np.sqrt(((df['X1']-c2[0])**2) + ((df['X2']-c2[1])**2)),

                        'cluster':['c1','c1','c1','c1','c2','c2','c2','c2','c2']})

print(df_dist)

print()

print('Iteration3')

c1 = (df.loc[list(df_dist[df_dist['cluster'] == 'c1'].index),:]['X1'].mean(),df.loc[list(df_dist[df_dist['cluster'] == 'c1'].index),:]['X2'].mean())

c2 = (df.loc[list(df_dist[df_dist['cluster'] == 'c2'].index),:]['X1'].mean(),df.loc[list(df_dist[df_dist['cluster'] == 'c2'].index),:]['X2'].mean())

print('Centroids : ',c1,c2)

print('As centroid value does not change after Iteration 2, the algorithm stops here...')

Inertia_c1 = np.sum((np.sqrt(((df.loc[list(df_dist[df_dist['cluster'] == 'c1'].index),:]['X1'] - c1[0])**2) + ((df.loc[list(df_dist[df_dist['cluster'] == 'c1'].index),:]['X2'] - c1[1])**2)))**2)

Inertia_c2 = np.sum((np.sqrt(((df.loc[list(df_dist[df_dist['cluster'] == 'c2'].index),:]['X1'] - c2[0])**2) + ((df.loc[list(df_dist[df_dist['cluster'] == 'c2'].index),:]['X2'] - c2[1])**2)))**2)

total_inertia = Inertia_c1 + Inertia_c2

print('C1 Inertia : ',Inertia_c1)

print('C2 Inertia : ',Inertia_c2)

print("Total Inertia : ",total_inertia)
c1 = ( 6.1 , 8 )

c2 = ( 6 , 5.2 )

print('Group4')

print('Iteration1')

df_dist = pd.DataFrame({'ID': ['s1','s2','s3','s4','s5','s6','s7','s8','s9'],

                        'Dist_c1':np.sqrt(((df['X1']-c1[0])**2) + ((df['X2']-c1[1])**2)),

                        'Dist_c2':np.sqrt(((df['X1']-c2[0])**2) + ((df['X2']-c2[1])**2)),

                        'cluster':['c2','c2','c2','c1','c1','c1','c2','c2','c2']})

print('Centroids : ',c1,c2)

print(df_dist)

print()

print('Iteration2')

c1 = (df.loc[list(df_dist[df_dist['cluster'] == 'c1'].index),:]['X1'].mean(),df.loc[list(df_dist[df_dist['cluster'] == 'c1'].index),:]['X2'].mean())

c2 = (df.loc[list(df_dist[df_dist['cluster'] == 'c2'].index),:]['X1'].mean(),df.loc[list(df_dist[df_dist['cluster'] == 'c2'].index),:]['X2'].mean())

print('Centroids : ',c1,c2)

df_dist = pd.DataFrame({'ID': ['s1','s2','s3','s4','s5','s6','s7','s8','s9'],

                        'Dist_c1':np.sqrt(((df['X1']-c1[0])**2) + ((df['X2']-c1[1])**2)),

                        'Dist_c2':np.sqrt(((df['X1']-c2[0])**2) + ((df['X2']-c2[1])**2)),

                        'cluster':['c2','c2','c2','c1','c1','c1','c2','c2','c2']})

print(df_dist)

print()

print('Iteration3')

c1 = (df.loc[list(df_dist[df_dist['cluster'] == 'c1'].index),:]['X1'].mean(),df.loc[list(df_dist[df_dist['cluster'] == 'c1'].index),:]['X2'].mean())

c2 = (df.loc[list(df_dist[df_dist['cluster'] == 'c2'].index),:]['X1'].mean(),df.loc[list(df_dist[df_dist['cluster'] == 'c2'].index),:]['X2'].mean())

print('Centroids : ',c1,c2)

print('As centroid value does not change after Iteration 2, the algorithm stops here...')

Inertia_c1 = np.sum((np.sqrt(((df.loc[list(df_dist[df_dist['cluster'] == 'c1'].index),:]['X1'] - c1[0])**2) + ((df.loc[list(df_dist[df_dist['cluster'] == 'c1'].index),:]['X2'] - c1[1])**2)))**2)

Inertia_c2 = np.sum((np.sqrt(((df.loc[list(df_dist[df_dist['cluster'] == 'c2'].index),:]['X1'] - c2[0])**2) + ((df.loc[list(df_dist[df_dist['cluster'] == 'c2'].index),:]['X2'] - c2[1])**2)))**2)

total_inertia = Inertia_c1 + Inertia_c2

print('C1 Inertia : ',Inertia_c1)

print('C2 Inertia : ',Inertia_c2)

print("Total Inertia : ",total_inertia)
c1 = ( 8.8 , 7 )

c2 = ( 11.8 , 4.2 )

print('Group5')

print('Iteration1')

df_dist = pd.DataFrame({'ID': ['s1','s2','s3','s4','s5','s6','s7','s8','s9'],

                        'Dist_c1':np.sqrt(((df['X1']-c1[0])**2) + ((df['X2']-c1[1])**2)),

                        'Dist_c2':np.sqrt(((df['X1']-c2[0])**2) + ((df['X2']-c2[1])**2)),

                        'cluster':['c1','c1','c1','c1','c1','c1','c1','c2','c2']})

print('Centroids : ',c1,c2)

print(df_dist)

print()

print('Iteration2')

c1 = (df.loc[list(df_dist[df_dist['cluster'] == 'c1'].index),:]['X1'].mean(),df.loc[list(df_dist[df_dist['cluster'] == 'c1'].index),:]['X2'].mean())

c2 = (df.loc[list(df_dist[df_dist['cluster'] == 'c2'].index),:]['X1'].mean(),df.loc[list(df_dist[df_dist['cluster'] == 'c2'].index),:]['X2'].mean())

print('Centroids : ',c1,c2)

df_dist = pd.DataFrame({'ID': ['s1','s2','s3','s4','s5','s6','s7','s8','s9'],

                        'Dist_c1':np.sqrt(((df['X1']-c1[0])**2) + ((df['X2']-c1[1])**2)),

                        'Dist_c2':np.sqrt(((df['X1']-c2[0])**2) + ((df['X2']-c2[1])**2)),

                        'cluster':['c1','c1','c1','c1','c1','c1','c2','c2','c2']})

print(df_dist)

print()

print('Iteration3')

c1 = (df.loc[list(df_dist[df_dist['cluster'] == 'c1'].index),:]['X1'].mean(),df.loc[list(df_dist[df_dist['cluster'] == 'c1'].index),:]['X2'].mean())

c2 = (df.loc[list(df_dist[df_dist['cluster'] == 'c2'].index),:]['X1'].mean(),df.loc[list(df_dist[df_dist['cluster'] == 'c2'].index),:]['X2'].mean())

print('Centroids : ',c1,c2)

df_dist = pd.DataFrame({'ID': ['s1','s2','s3','s4','s5','s6','s7','s8','s9'],

                        'Dist_c1':np.sqrt(((df['X1']-c1[0])**2) + ((df['X2']-c1[1])**2)),

                        'Dist_c2':np.sqrt(((df['X1']-c2[0])**2) + ((df['X2']-c2[1])**2)),

                        'cluster':['c1','c1','c1','c1','c1','c1','c2','c2','c2']})

print(df_dist)

print()

print('Iteration4')

c1 = (df.loc[list(df_dist[df_dist['cluster'] == 'c1'].index),:]['X1'].mean(),df.loc[list(df_dist[df_dist['cluster'] == 'c1'].index),:]['X2'].mean())

c2 = (df.loc[list(df_dist[df_dist['cluster'] == 'c2'].index),:]['X1'].mean(),df.loc[list(df_dist[df_dist['cluster'] == 'c2'].index),:]['X2'].mean())

print('Centroids : ',c1,c2)

print('As centroid value does not change after Iteration 3, the algorithm stops here...')

Inertia_c1 = np.sum((np.sqrt(((df.loc[list(df_dist[df_dist['cluster'] == 'c1'].index),:]['X1'] - c1[0])**2) + ((df.loc[list(df_dist[df_dist['cluster'] == 'c1'].index),:]['X2'] - c1[1])**2)))**2)

Inertia_c2 = np.sum((np.sqrt(((df.loc[list(df_dist[df_dist['cluster'] == 'c2'].index),:]['X1'] - c2[0])**2) + ((df.loc[list(df_dist[df_dist['cluster'] == 'c2'].index),:]['X2'] - c2[1])**2)))**2)

total_inertia = Inertia_c1 + Inertia_c2

print('C1 Inertia : ',Inertia_c1)

print('C2 Inertia : ',Inertia_c2)

print("Total Inertia : ",total_inertia)
c1 = ( 6 , 5.2 )

c2 = ( 11.8 , 4.2 )

print('Group6')

print('Iteration1')

df_dist = pd.DataFrame({'ID': ['s1','s2','s3','s4','s5','s6','s7','s8','s9'],

                        'Dist_c1':np.sqrt(((df['X1']-c1[0])**2) + ((df['X2']-c1[1])**2)),

                        'Dist_c2':np.sqrt(((df['X1']-c2[0])**2) + ((df['X2']-c2[1])**2)),

                        'cluster':['c1','c1','c1','c1','c1','c1','c2','c2','c2']})

print('Centroids : ',c1,c2)

print(df_dist)

print()

print('Iteration2')

c1 = (df.loc[list(df_dist[df_dist['cluster'] == 'c1'].index),:]['X1'].mean(),df.loc[list(df_dist[df_dist['cluster'] == 'c1'].index),:]['X2'].mean())

c2 = (df.loc[list(df_dist[df_dist['cluster'] == 'c2'].index),:]['X1'].mean(),df.loc[list(df_dist[df_dist['cluster'] == 'c2'].index),:]['X2'].mean())

print('Centroids : ',c1,c2)

df_dist = pd.DataFrame({'ID': ['s1','s2','s3','s4','s5','s6','s7','s8','s9'],

                        'Dist_c1':np.sqrt(((df['X1']-c1[0])**2) + ((df['X2']-c1[1])**2)),

                        'Dist_c2':np.sqrt(((df['X1']-c2[0])**2) + ((df['X2']-c2[1])**2)),

                        'cluster':['c1','c1','c1','c1','c1','c1','c2','c2','c2']})

print(df_dist)

print()

print('Iteration3')

c1 = (df.loc[list(df_dist[df_dist['cluster'] == 'c1'].index),:]['X1'].mean(),df.loc[list(df_dist[df_dist['cluster'] == 'c1'].index),:]['X2'].mean())

c2 = (df.loc[list(df_dist[df_dist['cluster'] == 'c2'].index),:]['X1'].mean(),df.loc[list(df_dist[df_dist['cluster'] == 'c2'].index),:]['X2'].mean())

print('Centroids : ',c1,c2)

print('As centroid value does not change after Iteration 2, the algorithm stops here...')

Inertia_c1 = np.sum((np.sqrt(((df.loc[list(df_dist[df_dist['cluster'] == 'c1'].index),:]['X1'] - c1[0])**2) + ((df.loc[list(df_dist[df_dist['cluster'] == 'c1'].index),:]['X2'] - c1[1])**2)))**2)

Inertia_c2 = np.sum((np.sqrt(((df.loc[list(df_dist[df_dist['cluster'] == 'c2'].index),:]['X1'] - c2[0])**2) + ((df.loc[list(df_dist[df_dist['cluster'] == 'c2'].index),:]['X2'] - c2[1])**2)))**2)

total_inertia = Inertia_c1 + Inertia_c2

print('C1 Inertia : ',Inertia_c1)

print('C2 Inertia : ',Inertia_c2)

print("Total Inertia : ",total_inertia)
df2 = pd.read_csv('../input/abalone-dataset/abalone.csv')
df2.head()
df2.shape
df2.drop('Sex',axis = 1,inplace = True)
sns.pairplot(df2,diag_kind = 'kde')
from scipy.stats import zscore

df_scaled = df2.apply(zscore)

df_scaled.head()
from sklearn.cluster import KMeans
cluster_range = range(1,15)

cluster_errors = []

for num_clusters in cluster_range:

  model = KMeans(num_clusters)

  model.fit(df_scaled)

  cluster_errors.append(model.inertia_)
import matplotlib.pyplot as plt
clusters_df = pd.DataFrame({'clusters':cluster_range,

                            'inertia': cluster_errors})

clusters_df
plt.figure(figsize = (12,6))

plt.plot(clusters_df['clusters'],clusters_df['inertia'],marker = 'o')

plt.xlabel('k')

plt.ylabel('Inertia')
kmeans = KMeans(n_clusters = 3,n_init = 15,random_state = 2)

kmeans.fit(df_scaled)
centroids = kmeans.cluster_centers_

centroids
centroid_df = pd.DataFrame(centroids,columns = list(df_scaled.columns))

centroid_df
df_scaled['labels'] = list(kmeans.labels_)

df_scaled
sns.pairplot(df_scaled,hue = 'labels')
sns.set(style = 'ticks',color_codes=True)



df2 = pd.read_csv('../input/matplotlib-datasets/iris_dataset.csv')



df2
df2.drop('species',axis = 1,inplace = True)



df_scaled = df2.apply(zscore)



sns.pairplot(df_scaled,diag_kind='kde')
clusters_range = range(1,15)

inertia = []

for num_clust in clusters_range:

  model = KMeans(n_clusters = num_clust,random_state = 2)

  model.fit(df_scaled)

  inertia.append(model.inertia_)



plt.plot(cluster_range,inertia,marker = 'o')
kmeans = KMeans(n_clusters=3,random_state=2)

kmeans.fit(df_scaled)

df2['class'] = kmeans.labels_

df2
sns.pairplot(df2,hue = 'class')
from mpl_toolkits.mplot3d import Axes3D



fig = plt.figure(figsize=(10,8))

ax = Axes3D(fig,rect = [0,0,1,1],elev = 10,azim = 120)

labels = kmeans.labels_

ax.scatter(df_scaled.iloc[:,0],df_scaled.iloc[:,2],df_scaled.iloc[:,3],c = labels.astype(np.float),edgecolor = 'k')

ax.w_xaxis.set_ticklabels([])

ax.w_yaxis.set_ticklabels([])

ax.w_zaxis.set_ticklabels([])

ax.set_xlabel('Sepal Length')

ax.set_ylabel('Petal Length')

ax.set_zlabel('Petal Width')

ax.set_title('3D plot for KMeans Clustering')