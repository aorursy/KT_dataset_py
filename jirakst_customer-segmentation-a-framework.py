# Basic

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 

from matplotlib import cm

import seaborn as sns 

import plotly as py

import plotly.graph_objs as go



# Tools

from sklearn.preprocessing import LabelEncoder



# Cluster Visualization

from scipy.cluster.hierarchy import dendrogram, ward

from sklearn.decomposition import PCA as PCA

from sklearn.manifold import TSNE

from sklearn.metrics import silhouette_samples



# Cluster Algorithms

from sklearn.cluster import KMeans

from sklearn.cluster import AgglomerativeClustering as AggClus

import scipy.cluster.hierarchy as sch



# Defaults

import warnings

import os

warnings.filterwarnings("ignore")

py.offline.init_notebook_mode(connected = True)

%matplotlib inline

#plt.rcParams['figure.figsize'] = (16, 9)

#plt.style.use('ggplot')
print(os.listdir("../input"))
df = pd.read_csv('../input/Mall_Customers.csv')
df.shape
df.columns
Id = df['CustomerID']
df = df.drop(['CustomerID'], axis=1)
df.info()
df.head()
df.describe().transpose()
sns.countplot(y = 'Gender' , data = df)
sns.pairplot(df, hue="Gender")
plt.figure(1, figsize = (25,25) )

n = 0



for x in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']:

    for y in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']:

        if( x != y ):

            n = n + 1

            plt.subplot(3, 3, n)

            plt.subplots_adjust(hspace = 0.5, wspace = 0.5)

            for gender in ['Male', 'Female']:

                sns.regplot(x = x, y = y, data = df[ df['Gender'] == gender ], label = gender)

                plt.title("{} vs {} wrt Gender".format(x,y))

                

plt.legend()

plt.show()
plt.figure(1 , figsize = (15 , 7))

n = 0 

for cols in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:

    n += 1 

    plt.subplot(1 , 3 , n)

    plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)

    sns.violinplot(x = cols , y = 'Gender' , data = df , palette = 'vlag')

    sns.swarmplot(x = cols , y = 'Gender' , data = df)

    plt.ylabel('Gender' if n == 1 else '')

    plt.title('Boxplots & Swarmplots' if n == 2 else '')

plt.show()
fig = plt.figure(figsize=(15,15))

ax1 = fig.add_subplot(3,3,1)

sns.barplot(y='Age',x='Gender', data=df);

ax2 = fig.add_subplot(3,3,2)

sns.barplot(y='Annual Income (k$)',x='Gender', data=df);

ax3 = fig.add_subplot(3,3,3)

sns.barplot(y='Spending Score (1-100)',x='Gender', data=df);
# Option 1

'''le = LabelEncoder()

df['Gender'] = le.fit_transform(df['Gender'])'''
# Option 2

df = df.drop(['Gender'], axis=1)
# Annual Income and spending Score

X2 = df[['Annual Income (k$)' , 'Spending Score (1-100)']].iloc[: , :].values

inertia = []

for n in range(1 , 11):

    algorithm = (KMeans(n_clusters = n ,init='k-means++', n_init = 10 ,max_iter=300, 

                        tol=0.0001,  random_state= 111  , algorithm='elkan') )

    algorithm.fit(X2)

    inertia.append(algorithm.inertia_)
plt.figure(1 , figsize = (15 ,6))

plt.plot(np.arange(1 , 11) , inertia , 'o')

plt.plot(np.arange(1 , 11) , inertia , '-' , alpha = 0.5)

plt.xlabel('Number of Clusters') , plt.ylabel('Inertia')

plt.show()
algorithm = (KMeans(n_clusters = 5 ,init='k-means++', n_init = 10 ,max_iter=300, 

                        tol=0.0001,  random_state= 111  , algorithm='elkan') )

algorithm.fit(X2)

labels2 = algorithm.labels_

centroids2 = algorithm.cluster_centers_
h = 0.02

x_min, x_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1

y_min, y_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z2 = algorithm.predict(np.c_[xx.ravel(), yy.ravel()]) 
plt.figure(1 , figsize = (15 , 7) )

plt.clf()

Z2 = Z2.reshape(xx.shape)

plt.imshow(Z2 , interpolation='nearest', 

           extent=(xx.min(), xx.max(), yy.min(), yy.max()),

           cmap = plt.cm.Pastel2, aspect = 'auto', origin='lower')



plt.scatter( x = 'Annual Income (k$)' ,y = 'Spending Score (1-100)' , data = df , c = labels2 , 

            s = 200 )

plt.scatter(x = centroids2[: , 0] , y =  centroids2[: , 1] , s = 300 , c = 'red' , alpha = 0.5)

plt.ylabel('Spending Score (1-100)') , plt.xlabel('Annual Income (k$)')

plt.show()
X = df.iloc[: , :].values

inertia = []

for n in range(1 , 11):

    algorithm = (KMeans(n_clusters = n ,init='k-means++', n_init = 10 ,max_iter=300, 

                        tol=0.0001,  random_state= 111  , algorithm='elkan') )

    algorithm.fit(X)

    inertia.append(algorithm.inertia_)
plt.figure(1 , figsize = (15 ,6))

plt.plot(np.arange(1 , 11) , inertia , 'o')

plt.plot(np.arange(1 , 11) , inertia , '-' , alpha = 0.5)

plt.xlabel('Number of Clusters') , plt.ylabel('Inertia')

plt.show()
algorithm = (KMeans(n_clusters = 6 ,init='k-means++', n_init = 10 ,max_iter=300, 

                        tol=0.0001,  random_state= 111  , algorithm='elkan') )

algorithm.fit(X)

labels = algorithm.labels_

centroids = algorithm.cluster_centers_
df['label'] =  labels

trace1 = go.Scatter3d(

    x= df['Age'],

    y= df['Spending Score (1-100)'],

    z= df['Annual Income (k$)'],

    mode='markers',

     marker=dict(

        color = df['label'], 

        size= 20,

        line=dict(

            color= df['label'],

            width= 12

        ),

        opacity=0.8

     )

)

data = [trace1]

layout = go.Layout(

#     margin=dict(

#         l=0,

#         r=0,

#         b=0,

#         t=0

#     )

    title= 'Clusters',

    scene = dict(

            xaxis = dict(title  = 'Age'),

            yaxis = dict(title  = 'Spending Score'),

            zaxis = dict(title  = 'Annual Income')

        )

)

fig = go.Figure(data=data, layout=layout)

py.offline.iplot(fig)
pca = PCA(n_components=2)

pca.fit(df)

Xpca = pca.transform(df)

sns.set()

plt.figure(figsize=(8,8))

plt.scatter(Xpca[:,0],Xpca[:,1], c='Red')

plt.show()
tsn = TSNE()

res_tsne = tsn.fit_transform(df)

plt.figure(figsize=(8,8))

plt.scatter(res_tsne[:,0],res_tsne[:,1]);
fig = plt.figure(figsize=(20,20))

ax1 = fig.add_subplot(3,3,1)

sns.scatterplot(x=res_tsne[:,0],y=res_tsne[:,1],s=100, hue=df['Spending Score (1-100)']);

ax2 = fig.add_subplot(3,3,2)

sns.scatterplot(x=res_tsne[:,0],y=res_tsne[:,1],s=100, hue=df['Annual Income (k$)']);

ax3 = fig.add_subplot(3,3,3)

sns.scatterplot(x=res_tsne[:,0],y=res_tsne[:,1],s=100, hue=df['Age']);
sns.set(style='white')

plt.figure(figsize=(10,7))

link = ward(res_tsne)

dendrogram(link)

ax = plt.gca()

bounds = ax.get_xbound()

ax.plot(bounds, [30,30],'--', c='k')

ax.plot(bounds,'--', c='k')

plt.show()
clus_mod = AggClus(n_clusters=6)

assign = clus_mod.fit_predict(df)

plt.figure(figsize=(8,8))

sns.set(style='darkgrid',palette='muted')

cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)

sns.scatterplot(x=res_tsne[:,0],y=res_tsne[:,1],s=100, hue=assign, palette='Set1');
def clust_sill(num):

    fig = plt.figure(figsize=(25,20))

    ax1 = fig.add_subplot(3,3,1)



    clus_mod = AggClus(n_clusters=num)

    assign = clus_mod.fit_predict(df)

    sns.set(style='darkgrid',palette='muted')

    cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)

    sns.scatterplot(x=res_tsne[:,0],y=res_tsne[:,1],s=100, hue=assign, palette='copper');

    cluster_labels=np.unique(assign)

    n_clusters = len(np.unique(assign))

    silhouette_vals = silhouette_samples(res_tsne, assign, metric='euclidean')



    y_ax_lower, y_ax_upper = 0, 0

    yticks = []

    ax2 = fig.add_subplot(3,3,2)

    for i , c in enumerate(cluster_labels):

        c_silhouette_vals = silhouette_vals[assign==c]

        c_silhouette_vals.sort()

        y_ax_upper += len(c_silhouette_vals)

        color = cm.jet(float(i) / n_clusters)

        plt.barh(range(y_ax_lower,y_ax_upper),

                c_silhouette_vals,height=1.0,edgecolor='none',color=color)

        yticks.append((y_ax_lower+y_ax_upper) / 2)

        y_ax_lower += len(c_silhouette_vals)

    silhouette_avg = np.mean(silhouette_vals)

    

    plt.title(str(num)+ ' Clusters')

    plt.axvline(silhouette_avg,color="red",linestyle= "--")

    plt.yticks(yticks , cluster_labels + 1)

    plt.ylabel ('Cluster')

    plt.xlabel('Silhouette coefficient')
clust_sill(3)

clust_sill(4)

clust_sill(5)

clust_sill(6)

clust_sill(7)
cluster_labels=np.unique(assign)

n_clusters = len(np.unique(assign))

silhouette_vals = silhouette_samples(res_tsne, assign, metric='euclidean')

y_ax_lower, y_ax_upper = 0, 0

yticks = []

plt.figure(figsize=(10,8))

for i , c in enumerate(cluster_labels):

        c_silhouette_vals = silhouette_vals[assign==c]

        c_silhouette_vals.sort()

        y_ax_upper += len(c_silhouette_vals)

        color = cm.jet(float(i) / n_clusters)

        plt.barh(range(y_ax_lower,y_ax_upper),

                c_silhouette_vals,height=1.0,edgecolor='none',color=color)

        yticks.append((y_ax_lower+y_ax_upper) / 2)

        y_ax_lower += len(c_silhouette_vals)

silhouette_avg = np.mean(silhouette_vals)



plt.axvline(silhouette_avg,color="red",linestyle= "--")

plt.yticks(yticks , cluster_labels + 1)

plt.ylabel ('Cluster')

plt.xlabel('Silhouette coefficient')
df['predict'] = pd.DataFrame(assign)
df.head(3)
sns.boxplot(y='Spending Score (1-100)',x='predict',data=df);
model = pd.DataFrame()

model['age'] = df['Age'].groupby(df['predict']).median()

model['annual income'] = df['Annual Income (k$)'].groupby(df['predict']).median()

model['spending score'] = df['Spending Score (1-100)'].groupby(df['predict']).median()

model.reset_index(inplace=True)
model