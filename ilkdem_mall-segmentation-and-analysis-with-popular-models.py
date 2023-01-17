!conda install -c conda-forge --yes hdbscan
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN,MeanShift,estimate_bandwidth,AffinityPropagation,SpectralClustering

from sklearn.mixture import GaussianMixture

from scipy.cluster.hierarchy import dendrogram,linkage,ward,complete,single,fcluster

from sklearn.metrics import silhouette_score

from sklearn.manifold import TSNE

import umap

import hdbscan

import warnings

warnings.filterwarnings("ignore")
def kmeans_cluster_decision(X,point,min_clusters=2,max_clusters=10):

    inertia = []

    score = []

    ticks = []

    for cluster in range(min_clusters,max_clusters):

        model = KMeans(n_clusters=cluster)

        model.fit(X)

        inertia.append(model.inertia_)

        score.append(silhouette_score(X,model.labels_))

        ticks.append(cluster)



    fig,ax = plt.subplots(1,2,figsize=(18,4))

    sns.lineplot(x=ticks,y=inertia,ax=ax[0]).set_title("Inertia vs Clusters")

    ax[0].set_xticks(ticks)

    ax[0].set_xticklabels(ticks)

    ax[0].plot(point,inertia[point - min_clusters],marker='x',c='r',markersize=6)

    sns.lineplot(x=ticks,y=score,ax=ax[1]).set_title("Sil. Score vs Clusters")

    ax[1].set_xticks(ticks)

    ax[1].set_xticklabels(ticks)

    ax[1].plot(point,score[point - min_clusters],marker='x',c='r',markersize=6)

    plt.show()



def draw_analysis_graphs(labels):

    fig,ax=plt.subplots(2,3,figsize=(18,14))

    sns.scatterplot(x='age',y='income',data=dfdata,hue=labels,palette='Paired',ax=ax[0][0])

    sns.scatterplot(x='age',y='spending',data=dfdata,hue=labels,palette='Paired',ax=ax[0][1])

    sns.scatterplot(x='spending',y='income',data=dfdata,hue=labels,palette='Paired',ax=ax[0][2])

    sns.swarmplot(x=labels,y='age',data=dfdata,hue=labels,palette='Paired',ax=ax[1][0])

    sns.swarmplot(x=labels,y='spending',data=dfdata,hue=labels,palette='Paired',ax=ax[1][1])

    sns.swarmplot(x=labels,y='income',data=dfdata,hue=labels,palette='Paired',ax=ax[1][2])

    plt.show()



def scaler(X):

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    return X_scaled
dfdata = pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')

dfdata.head()
dfdata.describe()
dfdata.columns = ['id','gender','age','income','spending']

dfdata.info()
plt.figure(figsize=(10,2))

sns.countplot(y = 'gender',data = dfdata)

plt.show()
plt.figure(figsize=(10,10))

sns.pairplot(data = dfdata[['age','income','spending','gender']],hue='gender',diag_kind='kde')

plt.show()
plt.figure(figsize=(18,6))

sns.countplot(x='age',data=dfdata,hue='gender')

plt.title("Age vs Gender Distribution")

plt.show()
plt.figure(figsize=(18,6))

sns.countplot(x='income',data=dfdata,hue='gender')

plt.title("Income vs gender distribution")

plt.show()
plt.figure(figsize=(18,6))

sns.countplot(x='spending',data=dfdata,hue='gender')

plt.title("Spending score vs gender distribution")

plt.show()
X_scaled = scaler(dfdata[['income','spending','age']])

kmeans_cluster_decision(X_scaled,6,2,10)
model = KMeans(n_clusters=6)

labels = model.fit_predict(X_scaled)

print("Inertia : ",round(model.inertia_,2),"Silhouette Score : ",round(silhouette_score(X_scaled,labels),2))

draw_analysis_graphs(labels)
X_scaled_ai = scaler(dfdata[['age','income']])

X_scaled_as = scaler(dfdata[['age','spending']])

X_scaled_is = scaler(dfdata[['income','spending']])

kmeans_cluster_decision(X_scaled_ai,3,2,20)
model = KMeans(n_clusters=3)

labels_ai = model.fit_predict(X_scaled_ai)

print("Inertia : ",round(model.inertia_,2),"Silhouette Score : ",round(silhouette_score(X_scaled_ai,labels_ai),2))

draw_analysis_graphs(labels_ai)
kmeans_cluster_decision(X_scaled_as,6,2,20)
model = KMeans(n_clusters=6)

labels_as = model.fit_predict(X_scaled_as)

print("Inertia : ",round(model.inertia_,2),"Silhouette Score : ",round(silhouette_score(X_scaled_as,labels_as),2))

draw_analysis_graphs(labels_as)
kmeans_cluster_decision(X_scaled_is,5,2,20)
model = KMeans(n_clusters=5)

labels_is = model.fit_predict(X_scaled_is)

print("Inertia : ",round(model.inertia_,2),"Silhouette Score : ",round(silhouette_score(X_scaled_is,labels_is),2))

draw_analysis_graphs(labels_is)
score = []

xticks = []

for i in range(3,10):

    model = GaussianMixture(n_components=i).fit(X_scaled_is)

    labels = model.predict(X_scaled_is)

    xticks.append("i:" + str(i) + "c:" + str(len(np.unique(labels))))

    score.append(silhouette_score(X_scaled_is,labels))

plt.figure(figsize=(18,4))

sns.lineplot(x=range(len(score)),y=score)

plt.xticks(range(len(score)),xticks,rotation=45)

plt.show()
model = GaussianMixture(n_components=5).fit(X_scaled_is)

labels_gm = model.predict(X_scaled_is)

print("Silhouette Score : ",round(silhouette_score(X_scaled_is,labels_gm),2))

draw_analysis_graphs(labels_gm)
pd.crosstab(labels_is,labels_gm)
links = linkage(X_scaled_is,method='ward')

plt.figure(figsize=(20,6))

dendrogram(links,leaf_rotation=90,leaf_font_size=6)

plt.show()
score = []

for i in range(1,15):

    labels_dendro = fcluster(links,t=i,criterion='distance')

    score.append(silhouette_score(X_scaled_is,labels_dendro))

plt.figure(figsize=(18,5))

sns.lineplot(x=range(len(score)),y=score)

plt.show()
labels_dendro = fcluster(links,t=5,criterion = 'distance')

print("Silhouette Score : ",round(silhouette_score(X_scaled_is,labels_dendro),2))

draw_analysis_graphs(labels_dendro)
score = []

for i in range(2,15):

    labels_agg = AgglomerativeClustering(n_clusters = i).fit(X_scaled_is)

    score.append(silhouette_score(X_scaled_is,labels_agg.labels_))

plt.figure(figsize=(18,5))

sns.lineplot(x=range(len(score)),y=score)

plt.show()
model = AgglomerativeClustering(n_clusters=5).fit(X_scaled_is)

labels_agg = model.labels_

print("Silhouette Score : ",round(silhouette_score(X_scaled_is,labels_agg),2))

draw_analysis_graphs(labels_agg)
fig,ax=plt.subplots(3,3,figsize=(24,15))

for i,eps in enumerate([round(x*0.1,1) for x in range(4,7)]):

    for j,sample in enumerate(range(6,12,2)):

        model = DBSCAN(eps=eps,min_samples=sample).fit(X_scaled_is)

        labels = model.labels_

        sns.scatterplot(x = 'spending',y='income',hue=labels,data=dfdata,ax=ax[j][i],palette='Paired')

        title = "eps: " + str(eps) + " min_samples:" + str(sample)

        ax[j,i].set_title(title)

plt.show()
score = []

xlabels = []

for i,eps in enumerate([round(x*0.1,1) for x in range(3,9)]):

    for j,sample in enumerate(range(6,14)):

        model = DBSCAN(eps=eps,min_samples=sample).fit(X_scaled_is)

        labels = model.labels_

        if len(np.unique(labels)) > 1:

            score.append(silhouette_score(X_scaled_is,labels))

        else:

            score.append(0)

        xlabels.append('e: ' + str(eps) + ' s: ' + str(sample))

plt.figure(figsize=(18,5))

sns.lineplot(x=range(len(score)),y=score)

plt.xticks(range(len(xlabels)),xlabels,rotation = 90)

plt.show()
score = []

for i in [x*0.01 for x in range(1,17)]:

    bandwidth = estimate_bandwidth(X_scaled_is, quantile=i)

    model = MeanShift(bandwidth).fit(X_scaled_is)

    score.append(silhouette_score(X_scaled_is,model.labels_))

plt.figure(figsize=(18,5))

sns.lineplot(x=range(len(score)),y=score)

plt.show()
bandwidth = estimate_bandwidth(X_scaled_is, quantile=0.15)

model = MeanShift(bandwidth).fit(X_scaled_is)

labels_ms = model.labels_

print("Silhouette Score : ",round(silhouette_score(X_scaled_is,labels_ms),2))

draw_analysis_graphs(labels_ms)
selected_columns = ['age','income','spending','genderlabel']

dfdata['genderlabel'] = dfdata['gender'].replace(['Male','Female'],[0,1])

X_scaled_all = pd.DataFrame(scaler(dfdata[selected_columns]),columns = selected_columns)
score = []

for i in range(1,5):

    pca = PCA(n_components = i).fit(X_scaled_all)

    score.append(np.sum(pca.explained_variance_ratio_))

plt.figure(figsize=(10,5))

sns.lineplot(x=range(len(score)),y=score)

plt.xticks(range(len(score)),range(1,5))

plt.show()
score = []

pca_vars = []

for i in range(1,5):

    pca = PCA(n_components = i)

    X_pca = pca.fit_transform(X_scaled_all)

    for j in range(3,7):

        model = KMeans(n_clusters=j).fit(X_pca)

        score.append(silhouette_score(X_pca,model.labels_))

        pca_vars.append("pca:" + str(i) + "c:" + str(j))

plt.figure(figsize=(18,6))

sns.lineplot(x=range(len(score)),y=score)

plt.xticks(ticks=range(len(score)),labels=pca_vars,rotation=90)

plt.show()
pca = PCA(n_components = 3).fit(X_scaled_all)

X_pca = pca.transform(X_scaled_all)

sns.heatmap(pca.components_,annot=True,fmt='.1g')

plt.xticks(range(X_scaled_all.shape[1]),X_scaled_all.columns.values,rotation=45)

plt.show()
model = KMeans(n_clusters=6).fit(X_pca)

labels_pca = model.labels_

score.append(silhouette_score(X_pca,labels_pca))

print("Silhouette Score : ",round(silhouette_score(X_pca,labels_pca),2))

draw_analysis_graphs(labels_pca)
X_scaled = scaler(dfdata[['income','spending','age']])

tsne = TSNE(n_components=2,perplexity=15,learning_rate=450)

tsne_result=tsne.fit_transform(X_scaled)
sns.scatterplot(tsne_result[:,0],tsne_result[:,1])

plt.show()
fig,ax = plt.subplots(1,4,figsize=(24,6))

sns.scatterplot(tsne_result[:,0],tsne_result[:,1],hue=labels_is,palette = 'Paired',ax = ax[0])

sns.scatterplot(tsne_result[:,0],tsne_result[:,1],hue=labels_dendro,palette = 'Paired',ax = ax[1])

sns.scatterplot(tsne_result[:,0],tsne_result[:,1],hue=labels_agg,palette = 'Paired',ax = ax[2])

sns.scatterplot(tsne_result[:,0],tsne_result[:,1],hue=labels_ms,palette = 'Paired',ax = ax[3])

plt.show()
resultset = ['Kmeans','Dendro','Agg','MSA']

dfResult = pd.DataFrame(zip(labels_is,labels_dendro,labels_agg,labels_ms),columns=resultset)

for i in range(len(resultset)):

    for j in range(i + 1,len(resultset)):

        print("Results for " + resultset[i] + " and " + resultset[j])

        print(pd.crosstab(dfResult.iloc[:,i],dfResult.iloc[:,j]))
X_scaled = scaler(dfdata[['income','spending','age']])

umap_data = umap.UMAP(n_neighbors=20).fit_transform(X_scaled)

model = hdbscan.HDBSCAN(min_cluster_size=5).fit(umap_data)

labels_hdbscan = model.labels_

print("Silhouette Score : ",round(silhouette_score(X_scaled,labels_hdbscan),2))

fig,ax = plt.subplots(1,2,figsize=(18,5))

sns.scatterplot(tsne_result[:,0],tsne_result[:,1],hue=labels_hdbscan,palette = 'Paired',ax=ax[0])

sns.scatterplot(umap_data[:,0],umap_data[:,1],hue=labels_hdbscan,palette = 'Paired',ax=ax[1])

plt.show()
pd.crosstab(labels_is,labels_hdbscan)
score = []

for i in [round(x * 0.1,1) for x in range(5,10)]:

    model = AffinityPropagation(damping = i,random_state=12).fit(X_scaled_is)

    score.append(silhouette_score(X_scaled_is,model.labels_))

plt.figure(figsize=(18,5))

sns.lineplot(x=range(len(score)),y=score)

plt.xticks(range(len(score)),[round(x * 0.1,1) for x in range(5,10)])

plt.show()
model = AffinityPropagation(damping = 0.7,random_state=12).fit(X_scaled_is)

labels_ap = model.labels_

print("Silhouette Score : ",round(silhouette_score(X_scaled_is,labels_ap),2))

sns.scatterplot(tsne_result[:,0],tsne_result[:,1],hue=labels_ap,palette = 'Paired')

plt.show()
pd.crosstab(labels_is,labels_ap)
score = []

for i in range(3,10):

    model = SpectralClustering(n_clusters=i).fit(X_scaled_is)

    score.append(silhouette_score(X_scaled_is,model.labels_))

plt.figure(figsize=(18,4))

sns.lineplot(x=range(len(score)),y=score)

plt.xticks(range(len(score)),range(5,10))

plt.show()
model = SpectralClustering(n_clusters = 8).fit(X_scaled_is)

labels_sc = model.labels_

print("Silhouette Score : ",round(silhouette_score(X_scaled_is,labels_sc),2))

sns.scatterplot(tsne_result[:,0],tsne_result[:,1],hue=labels_sc,palette = 'Paired')

plt.show()
pd.crosstab(labels_is,labels_sc)