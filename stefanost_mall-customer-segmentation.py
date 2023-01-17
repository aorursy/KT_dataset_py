import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df=pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')

df.head(3)
df.info()
df.drop(columns='CustomerID', inplace=True)
#distribution of categorical binary variable: Gender

plt.figure(figsize=(12,4));

plt.subplot(1,2,1);

sns.countplot(df['Gender']);

plt.title('gender value_counts');

plt.subplot(1,2,2);

df['Gender'].value_counts().plot(kind='pie',autopct='%.1f%%');

plt.title('gender proportion');
#one-hot encoding of Gender

df['Male']=pd.get_dummies(df['Gender'],drop_first=True)

df.drop(columns='Gender',inplace=True)

df.head(3)
sns.pairplot(df);
sns.scatterplot(x='Spending Score (1-100)', y='Annual Income (k$)',

               data=df, hue='Male');

plt.legend(loc=[1.1,0.7]);
sns.scatterplot(x='Spending Score (1-100)', y='Annual Income (k$)',

               data=df, hue='Age');

plt.legend(loc=[1.1,0.7]);
sns.heatmap(df.corr(), annot=True, fmt='1.1f');
from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_samples

from matplotlib import cm
x=df[['Annual Income (k$)','Spending Score (1-100)']].values

inertia=[]



for i in range(1,11):

    km=KMeans(n_clusters=i,random_state=33)

    km.fit(x)

    inertia.append(km.inertia_)

    

sns.lineplot(range(1,11),inertia);
km=KMeans(n_clusters=5, random_state=33)

clusters=km.fit_predict(x)



plt.figure(figsize=(7,5));

sns.scatterplot(x[:,0], x[:,1], hue =clusters);

plt.title('Clusters');

plt.legend(loc=(1.05,0.7));



plt.figure(figsize=(7,5));

sns.scatterplot(x[:,0], x[:,1], hue =clusters);

sns.scatterplot(km.cluster_centers_[:,0], km.cluster_centers_[:,1], color='red');

plt.title('Clusters plus Cluster Centroids');

plt.legend(loc=(1.05,0.7));



#Graphing Silhouette

#code from Sebastian Raschka's book 'Python Machine Learning'

labels=np.unique(clusters)

n_clusters=labels.shape[0]

sils=silhouette_samples(x,clusters,metric='euclidean')

y_ax_lower, y_ax_upper=0, 0

yticks=[]

plt.figure(figsize=(6,5))

for i,c in enumerate(labels):

    cluster_sil=sils[clusters==c]

    cluster_sil.sort()

    y_ax_upper +=len(cluster_sil)

    color=cm.jet(float(i)/n_clusters)

    plt.barh(range(y_ax_lower, y_ax_upper),

            cluster_sil, height=1.0,

            edgecolor='none', color=color)

    yticks.append((y_ax_lower+y_ax_upper)/2.)

    y_ax_lower+=len(cluster_sil)

silhouette_avg=np.mean(sils)

plt.axvline(silhouette_avg,color='red', linestyle='--')

plt.title('Silhouette Graph');

plt.yticks(yticks,labels+1)

plt.ylabel('Cluster')

plt.xlabel('Silhouette coefficient')

plt.show();
#initializations

x=df[['Age','Spending Score (1-100)']].values

inertia=[]



#kmeans with various cluster-numbers

for i in range (1,11):

    km=KMeans(n_clusters=i, random_state=33)

    km.fit(x)

    inertia.append(km.inertia_)



#plot inertia

ind=np.arange(1,11)

sns.lineplot(ind,inertia);
km=KMeans(n_clusters=4,random_state=33)

clusters=km.fit_predict(x)



plt.figure(figsize=(7,5));

sns.scatterplot(x[:,1],x[:,0],hue=clusters);

plt.title('Clusters');

plt.legend(loc=(1.05,0.7));



plt.figure(figsize=(7,5));

sns.scatterplot(x[:,1],x[:,0],hue=clusters);

sns.scatterplot(km.cluster_centers_[:,1],km.cluster_centers_[:,0], color='red')

plt.title('Clusters plus Cluster Centers');

plt.legend(loc=(1.05,0.7));



#Graphing Silhouette

labels=np.unique(clusters)

n_clusters=labels.shape[0]

sils=silhouette_samples(x,clusters,metric='euclidean')

y_ax_lower, y_ax_upper=0, 0

yticks=[]

plt.figure(figsize=(6,5))

for i,c in enumerate(labels):

    cluster_sil=sils[clusters==c]

    cluster_sil.sort()

    y_ax_upper +=len(cluster_sil)

    color=cm.jet(float(i)/n_clusters)

    plt.barh(range(y_ax_lower, y_ax_upper),

            cluster_sil, height=1.0,

            edgecolor='none', color=color)

    yticks.append((y_ax_lower+y_ax_upper)/2.)

    y_ax_lower+=len(cluster_sil)

silhouette_avg=np.mean(sils)

plt.axvline(silhouette_avg,color='red', linestyle='--')

plt.yticks(yticks,labels+1)

plt.ylabel('Cluster')

plt.xlabel('Silhouette coefficient')

plt.title('Silhouette Graph');

plt.show();
x=df.values

inertia=[]



for i in range(1,11):

    km=KMeans(n_clusters=i,random_state=33)

    km.fit(x)

    inertia.append(km.inertia_)

    

sns.lineplot(range(1,11),inertia);
km=KMeans(n_clusters=6,random_state=33)

clusters=km.fit_predict(x)



#Graphing Silhouette

labels=np.unique(clusters)

n_clusters=labels.shape[0]

sils=silhouette_samples(x,clusters,metric='euclidean')

y_ax_lower, y_ax_upper=0, 0

yticks=[]

plt.figure(figsize=(6,5))

for i,c in enumerate(labels):

    cluster_sil=sils[clusters==c]

    cluster_sil.sort()

    y_ax_upper +=len(cluster_sil)

    color=cm.jet(float(i)/n_clusters)

    plt.barh(range(y_ax_lower, y_ax_upper),

            cluster_sil, height=1.0,

            edgecolor='none', color=color)

    yticks.append((y_ax_lower+y_ax_upper)/2.)

    y_ax_lower+=len(cluster_sil)

silhouette_avg=np.mean(sils)

plt.axvline(silhouette_avg,color='red', linestyle='--')

plt.yticks(yticks,labels+1)

plt.ylabel('Cluster')

plt.xlabel('Silhouette coefficient')

plt.title('Silhouette Graph');

plt.show();
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

x=sc.fit_transform(df.values)

inertia=[]



for i in range(1,11):

    km=KMeans(n_clusters=i,random_state=33)

    km.fit(x)

    inertia.append(km.inertia_)

    

sns.lineplot(range(1,11),inertia);
#six clusters

km=KMeans(n_clusters=6,random_state=33)

clusters=km.fit_predict(x)



#Graphing Silhouette

labels=np.unique(clusters)

n_clusters=labels.shape[0]

sils=silhouette_samples(x,clusters,metric='euclidean')

y_ax_lower, y_ax_upper=0, 0

yticks=[]

plt.figure(figsize=(6,5))

for i,c in enumerate(labels):

    cluster_sil=sils[clusters==c]

    cluster_sil.sort()

    y_ax_upper +=len(cluster_sil)

    color=cm.jet(float(i)/n_clusters)

    plt.barh(range(y_ax_lower, y_ax_upper),

            cluster_sil, height=1.0,

            edgecolor='none', color=color)

    yticks.append((y_ax_lower+y_ax_upper)/2.)

    y_ax_lower+=len(cluster_sil)

silhouette_avg=np.mean(sils)

plt.axvline(silhouette_avg,color='red', linestyle='--')

plt.yticks(yticks,labels+1)

plt.ylabel('Cluster')

plt.xlabel('Silhouette coefficient')

plt.show();
#seven clusters

km=KMeans(n_clusters=7,random_state=33)

clusters=km.fit_predict(x)



#Graphing Silhouette

labels=np.unique(clusters)

n_clusters=labels.shape[0]

sils=silhouette_samples(x,clusters,metric='euclidean')

y_ax_lower, y_ax_upper=0, 0

yticks=[]

plt.figure(figsize=(6,5))

for i,c in enumerate(labels):

    cluster_sil=sils[clusters==c]

    cluster_sil.sort()

    y_ax_upper +=len(cluster_sil)

    color=cm.jet(float(i)/n_clusters)

    plt.barh(range(y_ax_lower, y_ax_upper),

            cluster_sil, height=1.0,

            edgecolor='none', color=color)

    yticks.append((y_ax_lower+y_ax_upper)/2.)

    y_ax_lower+=len(cluster_sil)

silhouette_avg=np.mean(sils)

plt.axvline(silhouette_avg,color='red', linestyle='--')

plt.yticks(yticks,labels+1)

plt.ylabel('Cluster')

plt.xlabel('Silhouette coefficient')

plt.show();
#eight clusters

km=KMeans(n_clusters=8,random_state=33)

clusters=km.fit_predict(x)



#Graphing Silhouette

labels=np.unique(clusters)

n_clusters=labels.shape[0]

sils=silhouette_samples(x,clusters,metric='euclidean')

y_ax_lower, y_ax_upper=0, 0

yticks=[]

plt.figure(figsize=(6,5))

for i,c in enumerate(labels):

    cluster_sil=sils[clusters==c]

    cluster_sil.sort()

    y_ax_upper +=len(cluster_sil)

    color=cm.jet(float(i)/n_clusters)

    plt.barh(range(y_ax_lower, y_ax_upper),

            cluster_sil, height=1.0,

            edgecolor='none', color=color)

    yticks.append((y_ax_lower+y_ax_upper)/2.)

    y_ax_lower+=len(cluster_sil)

silhouette_avg=np.mean(sils)

plt.axvline(silhouette_avg,color='red', linestyle='--')

plt.yticks(yticks,labels+1)

plt.ylabel('Cluster')

plt.xlabel('Silhouette coefficient')

plt.show();
from sklearn.decomposition import PCA
pca=PCA(n_components=2)

x=pca.fit_transform(df.values)

sns.scatterplot(x[:,0],x[:,1]);



plt.figure()

sns.scatterplot(x[:,0],x[:,1],hue=df['Spending Score (1-100)']);

plt.legend(loc=(1.01,0.64));



plt.figure()

sns.scatterplot(x[:,0],x[:,1],hue=df['Annual Income (k$)']);

plt.legend(loc=(1.01,0.64));



plt.figure()

sns.scatterplot(x[:,0],x[:,1],hue=df['Age']);

plt.legend(loc=(1.01,0.57));



plt.figure()

sns.scatterplot(x[:,0],x[:,1],hue=df['Male']);

plt.legend(loc=(1.01,0.64));
#elbow graph to determine optimal n_clusters

inertia=[]

for i in range(1,11):

    km=KMeans(n_clusters=i,random_state=33)

    km.fit(x)

    inertia.append(km.inertia_)

    

sns.lineplot(range(1,11),inertia);
km=KMeans(n_clusters=5, random_state=33)

clusters=km.fit_predict(x)



plt.figure(figsize=(7,5));

sns.scatterplot(x[:,0], x[:,1], hue =clusters);

plt.title('Clusters');

plt.legend(loc=(1.05,0.7));



plt.figure(figsize=(7,5));

sns.scatterplot(x[:,0], x[:,1], hue =clusters);

sns.scatterplot(km.cluster_centers_[:,0], km.cluster_centers_[:,1], color='red')

plt.title('Clusters plus Cluster Centers');

plt.legend(loc=(1.05,0.7));



#Graphing Silhouette

labels=np.unique(clusters)

n_clusters=labels.shape[0]

sils=silhouette_samples(x,clusters,metric='euclidean')

y_ax_lower, y_ax_upper=0, 0

yticks=[]

plt.figure(figsize=(6,5))

for i,c in enumerate(labels):

    cluster_sil=sils[clusters==c]

    cluster_sil.sort()

    y_ax_upper +=len(cluster_sil)

    color=cm.jet(float(i)/n_clusters)

    plt.barh(range(y_ax_lower, y_ax_upper),

            cluster_sil, height=1.0,

            edgecolor='none', color=color)

    yticks.append((y_ax_lower+y_ax_upper)/2.)

    y_ax_lower+=len(cluster_sil)

silhouette_avg=np.mean(sils)

plt.axvline(silhouette_avg,color='red', linestyle='--')

plt.yticks(yticks,labels+1)

plt.ylabel('Cluster')

plt.xlabel('Silhouette coefficient')

plt.title('Silhouette Graph');

plt.show();
pca=PCA(n_components=2)

sc=StandardScaler()

x=pca.fit_transform(sc.fit_transform(df.values))

sns.scatterplot(x[:,0],x[:,1]);



plt.figure()

sns.scatterplot(x[:,0],x[:,1],hue=df['Spending Score (1-100)']);

plt.legend(loc=(1.01,0.64));



plt.figure()

sns.scatterplot(x[:,0],x[:,1],hue=df['Annual Income (k$)']);

plt.legend(loc=(1.01,0.64));



plt.figure()

sns.scatterplot(x[:,0],x[:,1],hue=df['Age']);

plt.legend(loc=(1.01,0.57));



plt.figure()

sns.scatterplot(x[:,0],x[:,1],hue=df['Male']);

plt.legend(loc=(1.01,0.64));
#elbow graph to determine optimal n_clusters

inertia=[]

for i in range(1,11):

    km=KMeans(n_clusters=i,random_state=33)

    km.fit(x)

    inertia.append(km.inertia_)

    

sns.lineplot(range(1,11),inertia);
km=KMeans(n_clusters=4, random_state=33)

clusters=km.fit_predict(x)



plt.figure(figsize=(7,5));

sns.scatterplot(x[:,0], x[:,1], hue =clusters);

plt.title('Clusters');

plt.legend(loc=(1.05,0.7));



plt.figure(figsize=(7,5));

sns.scatterplot(x[:,0], x[:,1], hue =clusters);

sns.scatterplot(km.cluster_centers_[:,0], km.cluster_centers_[:,1], color='red')

plt.title('Clusters plus Cluster Centers');

plt.legend(loc=(1.05,0.7));



#Graphing Silhouette

labels=np.unique(clusters)

n_clusters=labels.shape[0]

sils=silhouette_samples(x,clusters,metric='euclidean')

y_ax_lower, y_ax_upper=0, 0

yticks=[]

plt.figure(figsize=(6,5))

for i,c in enumerate(labels):

    cluster_sil=sils[clusters==c]

    cluster_sil.sort()

    y_ax_upper +=len(cluster_sil)

    color=cm.jet(float(i)/n_clusters)

    plt.barh(range(y_ax_lower, y_ax_upper),

            cluster_sil, height=1.0,

            edgecolor='none', color=color)

    yticks.append((y_ax_lower+y_ax_upper)/2.)

    y_ax_lower+=len(cluster_sil)

silhouette_avg=np.mean(sils)

plt.axvline(silhouette_avg,color='red', linestyle='--')

plt.yticks(yticks,labels+1)

plt.ylabel('Cluster')

plt.xlabel('Silhouette coefficient')

plt.title('Silhouette Graph');

plt.show();
km=KMeans(n_clusters=6, random_state=33)

clusters=km.fit_predict(x)



plt.figure(figsize=(7,5));

sns.scatterplot(x[:,0], x[:,1], hue =clusters);

plt.title('Clusters');

plt.legend(loc=(1.05,0.7));



plt.figure(figsize=(7,5));

sns.scatterplot(x[:,0], x[:,1], hue =clusters);

sns.scatterplot(km.cluster_centers_[:,0], km.cluster_centers_[:,1], color='red')

plt.title('Clusters plus Cluster Centers');

plt.legend(loc=(1.05,0.7));



#Graphing Silhouette

labels=np.unique(clusters)

n_clusters=labels.shape[0]

sils=silhouette_samples(x,clusters,metric='euclidean')

y_ax_lower, y_ax_upper=0, 0

yticks=[]

plt.figure(figsize=(6,5))

for i,c in enumerate(labels):

    cluster_sil=sils[clusters==c]

    cluster_sil.sort()

    y_ax_upper +=len(cluster_sil)

    color=cm.jet(float(i)/n_clusters)

    plt.barh(range(y_ax_lower, y_ax_upper),

            cluster_sil, height=1.0,

            edgecolor='none', color=color)

    yticks.append((y_ax_lower+y_ax_upper)/2.)

    y_ax_lower+=len(cluster_sil)

silhouette_avg=np.mean(sils)

plt.axvline(silhouette_avg,color='red', linestyle='--')

plt.yticks(yticks,labels+1)

plt.ylabel('Cluster')

plt.xlabel('Silhouette coefficient')

plt.title('Silhouette Graph');

plt.show();
from scipy.cluster.hierarchy import dendrogram

from scipy.cluster.hierarchy import linkage

from sklearn.cluster import AgglomerativeClustering as agglo
x=df[['Annual Income (k$)','Spending Score (1-100)']].values

dend=dendrogram(linkage(x, method='ward'))

plt.show();
ag=agglo(n_clusters=5, affinity='euclidean',linkage='complete')

clusters=ag.fit_predict(x)



plt.figure(figsize=(7,5));

sns.scatterplot(x[:,0], x[:,1], hue =clusters);

plt.title('Clusters');

plt.legend(loc=(1.05,0.7));





#Graphing Silhouette

labels=np.unique(clusters)

n_clusters=labels.shape[0]

sils=silhouette_samples(x,clusters,metric='euclidean')

y_ax_lower, y_ax_upper=0, 0

yticks=[]

plt.figure(figsize=(6,5))

for i,c in enumerate(labels):

    cluster_sil=sils[clusters==c]

    cluster_sil.sort()

    y_ax_upper +=len(cluster_sil)

    color=cm.jet(float(i)/n_clusters)

    plt.barh(range(y_ax_lower, y_ax_upper),

            cluster_sil, height=1.0,

            edgecolor='none', color=color)

    yticks.append((y_ax_lower+y_ax_upper)/2.)

    y_ax_lower+=len(cluster_sil)

silhouette_avg=np.mean(sils)

plt.axvline(silhouette_avg,color='red', linestyle='--')

plt.yticks(yticks,labels+1)

plt.ylabel('Cluster')

plt.xlabel('Silhouette coefficient')

plt.title('Silhouette Graph');

plt.show();
x=df[['Age','Spending Score (1-100)']].values

den=dendrogram(linkage(x,method='ward'))

plt.show();
#four clusters



ag=agglo(n_clusters=4, affinity='euclidean',linkage='complete')

clusters=ag.fit_predict(x)



plt.figure(figsize=(7,5));

sns.scatterplot(x[:,1], x[:,0], hue =clusters);

plt.title('Clusters');

plt.legend(loc=(1.05,0.7));





#Graphing Silhouette

labels=np.unique(clusters)

n_clusters=labels.shape[0]

sils=silhouette_samples(x,clusters,metric='euclidean')

y_ax_lower, y_ax_upper=0, 0

yticks=[]

plt.figure(figsize=(6,5))

for i,c in enumerate(labels):

    cluster_sil=sils[clusters==c]

    cluster_sil.sort()

    y_ax_upper +=len(cluster_sil)

    color=cm.jet(float(i)/n_clusters)

    plt.barh(range(y_ax_lower, y_ax_upper),

            cluster_sil, height=1.0,

            edgecolor='none', color=color)

    yticks.append((y_ax_lower+y_ax_upper)/2.)

    y_ax_lower+=len(cluster_sil)

silhouette_avg=np.mean(sils)

plt.axvline(silhouette_avg,color='red', linestyle='--')

plt.yticks(yticks,labels+1)

plt.ylabel('Cluster')

plt.xlabel('Silhouette coefficient')

plt.title('Silhouette Graph');

plt.show();
#three clusters

ag=agglo(n_clusters=3, affinity='euclidean',linkage='complete')

clusters=ag.fit_predict(x)



plt.figure(figsize=(7,5));

sns.scatterplot(x[:,1], x[:,0], hue =clusters);

plt.title('Clusters');

plt.legend(loc=(1.05,0.7));





#Graphing Silhouette

labels=np.unique(clusters)

n_clusters=labels.shape[0]

sils=silhouette_samples(x,clusters,metric='euclidean')

y_ax_lower, y_ax_upper=0, 0

yticks=[]

plt.figure(figsize=(6,5))

for i,c in enumerate(labels):

    cluster_sil=sils[clusters==c]

    cluster_sil.sort()

    y_ax_upper +=len(cluster_sil)

    color=cm.jet(float(i)/n_clusters)

    plt.barh(range(y_ax_lower, y_ax_upper),

            cluster_sil, height=1.0,

            edgecolor='none', color=color)

    yticks.append((y_ax_lower+y_ax_upper)/2.)

    y_ax_lower+=len(cluster_sil)

silhouette_avg=np.mean(sils)

plt.axvline(silhouette_avg,color='red', linestyle='--')

plt.yticks(yticks,labels+1)

plt.ylabel('Cluster')

plt.xlabel('Silhouette coefficient')

plt.title('Silhouette Graph');

plt.show();
x=df.values

den=dendrogram(linkage(x,method='ward'))

plt.show();
#the dendrogram suggests six clusters

ag=agglo(n_clusters=6, affinity='euclidean',linkage='complete')

clusters=ag.fit_predict(x)



#Graphing Silhouette

labels=np.unique(clusters)

n_clusters=labels.shape[0]

sils=silhouette_samples(x,clusters,metric='euclidean')

y_ax_lower, y_ax_upper=0, 0

yticks=[]

plt.figure(figsize=(6,5))

for i,c in enumerate(labels):

    cluster_sil=sils[clusters==c]

    cluster_sil.sort()

    y_ax_upper +=len(cluster_sil)

    color=cm.jet(float(i)/n_clusters)

    plt.barh(range(y_ax_lower, y_ax_upper),

            cluster_sil, height=1.0,

            edgecolor='none', color=color)

    yticks.append((y_ax_lower+y_ax_upper)/2.)

    y_ax_lower+=len(cluster_sil)

silhouette_avg=np.mean(sils)

plt.axvline(silhouette_avg,color='red', linestyle='--')

plt.yticks(yticks,labels+1)

plt.ylabel('Cluster')

plt.xlabel('Silhouette coefficient')

plt.show();
pca=PCA(n_components=2)

x=pca.fit_transform(df.values)

dend=dendrogram(linkage(x,method='ward'))

plt.show();
#dendrogram suggests five clusters

ag=agglo(n_clusters=5, linkage='complete')

clusters=ag.fit_predict(x)



plt.figure(figsize=(7,5));

sns.scatterplot(x[:,0], x[:,1], hue =clusters);

plt.title('Clusters');

plt.legend(loc=(1.05,0.7));



#Graphing Silhouette

labels=np.unique(clusters)

n_clusters=labels.shape[0]

sils=silhouette_samples(x,clusters,metric='euclidean')

y_ax_lower, y_ax_upper=0, 0

yticks=[]

plt.figure(figsize=(6,5))

for i,c in enumerate(labels):

    cluster_sil=sils[clusters==c]

    cluster_sil.sort()

    y_ax_upper +=len(cluster_sil)

    color=cm.jet(float(i)/n_clusters)

    plt.barh(range(y_ax_lower, y_ax_upper),

            cluster_sil, height=1.0,

            edgecolor='none', color=color)

    yticks.append((y_ax_lower+y_ax_upper)/2.)

    y_ax_lower+=len(cluster_sil)

silhouette_avg=np.mean(sils)

plt.axvline(silhouette_avg,color='red', linestyle='--')

plt.yticks(yticks,labels+1)

plt.ylabel('Cluster')

plt.xlabel('Silhouette coefficient')

plt.title('Silhouette Graph');

plt.show();
sc=StandardScaler()

pca=PCA(n_components=2)

x=pca.fit_transform(sc.fit_transform(df.values))

dend=dendrogram(linkage(x,method='ward'))

plt.show();
#dendrogram suggests four clusters

ag=agglo(n_clusters=4, linkage='complete')

clusters=ag.fit_predict(x)



plt.figure(figsize=(7,5));

sns.scatterplot(x[:,0], x[:,1], hue =clusters);

plt.title('Clusters');

plt.legend(loc=(1.05,0.7));



#Graphing Silhouette

labels=np.unique(clusters)

n_clusters=labels.shape[0]

sils=silhouette_samples(x,clusters,metric='euclidean')

y_ax_lower, y_ax_upper=0, 0

yticks=[]

plt.figure(figsize=(6,5))

for i,c in enumerate(labels):

    cluster_sil=sils[clusters==c]

    cluster_sil.sort()

    y_ax_upper +=len(cluster_sil)

    color=cm.jet(float(i)/n_clusters)

    plt.barh(range(y_ax_lower, y_ax_upper),

            cluster_sil, height=1.0,

            edgecolor='none', color=color)

    yticks.append((y_ax_lower+y_ax_upper)/2.)

    y_ax_lower+=len(cluster_sil)

silhouette_avg=np.mean(sils)

plt.axvline(silhouette_avg,color='red', linestyle='--')

plt.yticks(yticks,labels+1)

plt.ylabel('Cluster')

plt.xlabel('Silhouette coefficient')

plt.title('Silhouette Graph');

plt.show();