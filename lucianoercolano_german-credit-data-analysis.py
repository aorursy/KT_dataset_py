!pip install pandas-profiling
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans, AffinityPropagation

import warnings

warnings.filterwarnings("ignore")

import pandas_profiling



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
datos = pd.read_csv ("/kaggle/input/german-credit/german_credit_data.csv", index_col=0)
datos.profile_report(style={'full_width':True})
def scatters(data, h=None, pal=None):

    fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(8,8))

    sns.scatterplot(x="Credit_amount",y="Duration", hue=h, palette=pal, data=data, ax=ax1)

    sns.scatterplot(x="Age",y="Credit_amount", hue=h, palette=pal, data=data, ax=ax2)

    sns.scatterplot(x="Age",y="Duration", hue=h, palette=pal, data=data, ax=ax3)

    plt.tight_layout()
scatters(datos, h="Sex")
import scipy.stats as stats

r1 = sns.jointplot(x="Credit_amount",y="Duration", data=datos, kind="reg", height=8)

r1.annotate(stats.pearsonr)

plt.show()
sns.lmplot(x="Credit_amount",y="Duration", hue="Sex", data=datos, palette="Set1", aspect=2)

plt.show()
sns.lmplot(x="Credit_amount",y="Duration", hue="Housing", data=datos, palette="Set1", aspect=2)

plt.show()
sns.jointplot("Credit_amount","Duration", data=datos, kind="kde", space=0, color="g",  height=8)

plt.show()
n_credits = datos.groupby("Purpose")["Age"].count().rename("Count").reset_index()

n_credits.sort_values(by=["Count"], ascending=False, inplace=True)



plt.figure(figsize=(10,6))

bar = sns.barplot(x="Purpose",y="Count",data=n_credits)

bar.set_xticklabels(bar.get_xticklabels(), rotation=60)

plt.ylabel("Number of granted credits")

plt.tight_layout()
def boxes(x,y,h,r=45):

    fig, ax = plt.subplots(figsize=(10,6))

    box = sns.boxplot(x=x,y=y, hue=h, data=datos)

    box.set_xticklabels(box.get_xticklabels(), rotation=r)

    fig.subplots_adjust(bottom=0.2)

    plt.tight_layout()
boxes("Purpose","Credit_amount","Sex")
boxes("Purpose","Duration","Sex")
boxes("Housing","Credit_amount","Sex",r=0)
boxes("Job","Credit_amount","Sex",r=0)

boxes("Job","Duration","Sex",r=0)
#Selección de columnas para clusters con k-means

selected_cols = ["Age","Credit_amount", "Duration"]

cluster_data = datos.loc[:,selected_cols]
#Función para crear un histograma

def distributions(df):

    fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(8,8))

    sns.distplot(df["Age"], ax=ax1)

    sns.distplot(df["Credit_amount"], ax=ax2)

    sns.distplot(df["Duration"], ax=ax3)

    plt.tight_layout()
#Impresión de histogramas

distributions(cluster_data)
cluster_log = np.log(cluster_data)

distributions(cluster_log)
scaler = StandardScaler()

cluster_scaled = scaler.fit_transform(cluster_log)
clusters_range = [2,3,4,5,6,7,8,9,10,11,12,13,14]

inertias =[]



for c in clusters_range:

    kmeans = KMeans(n_clusters=c, random_state=0).fit(cluster_scaled)

    inertias.append(kmeans.inertia_)



plt.figure()

plt.plot(clusters_range,inertias, marker='o')
from sklearn.metrics import silhouette_samples, silhouette_score



clusters_range = range(2,15)

random_range = range(0,20)

results =[]

for c in clusters_range:

    for r in random_range:

        clusterer = KMeans(n_clusters=c, random_state=r)

        cluster_labels = clusterer.fit_predict(cluster_scaled)

        silhouette_avg = silhouette_score(cluster_scaled, cluster_labels)

        #print("For n_clusters =", c," and seed =", r,  "\nThe average silhouette_score is :", silhouette_avg)

        results.append([c,r,silhouette_avg])



result = pd.DataFrame(results, columns=["n_clusters","seed","silhouette_score"])

pivot_km = pd.pivot_table(result, index="n_clusters", columns="seed",values="silhouette_score")



plt.figure(figsize=(15,6))

sns.heatmap(pivot_km, annot=True, linewidths=.5, fmt='.3f', cmap=sns.cm.rocket_r)

plt.tight_layout()
kmeans_sel = KMeans(n_clusters=3, random_state=1).fit(cluster_scaled)

labels = pd.DataFrame(kmeans_sel.labels_)

clustered_data = cluster_data.assign(Cluster=labels)
import matplotlib.cm as cm



clusterer = KMeans(n_clusters=3, random_state=1)

cluster_labels = clusterer.fit_predict(cluster_scaled)

silhouette_avg = silhouette_score(cluster_scaled, cluster_labels)

print("For n_clusters =", 3," and seed =", r,  "\nThe average silhouette_score is :", silhouette_avg)



# Compute the silhouette scores for each sample

sample_silhouette_values = silhouette_samples(cluster_scaled, cluster_labels)



fig, ax1 = plt.subplots(figsize=(10,6))



y_lower = 10

for i in range(3):

    # Aggregate the silhouette scores for samples belonging to

    # cluster i, and sort them

    ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

    ith_cluster_silhouette_values.sort()



    size_cluster_i = ith_cluster_silhouette_values.shape[0]

    y_upper = y_lower + size_cluster_i

    

    color = cm.nipy_spectral(float(i) / 3)

    ax1.fill_betweenx(np.arange(y_lower, y_upper),0, ith_cluster_silhouette_values, facecolor=color, edgecolor="black", alpha=0.7)

    

    # Label the silhouette plots with their cluster numbers at the middle

    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

    

    # Compute the new y_lower for next plot

    y_lower = y_upper + 10  # 10 for the 0 samples



ax1.get_yaxis().set_ticks([])

ax1.set_title("The silhouette plot for various clusters")

ax1.set_xlabel("The silhouette coefficient values")

ax1.set_ylabel("Cluster label")

# The vertical line for average silhouette score of all the values

ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
scatters(clustered_data, 'Cluster')
grouped_km = clustered_data.groupby(['Cluster']).mean().round(1)

grouped_km
preferences = np.arange(-30,-190,-10)

clusters = []



for p in preferences:

    af = AffinityPropagation(preference=p, damping=0.6, max_iter=400, verbose=False).fit(cluster_scaled)

    labels_af = pd.DataFrame(af.labels_)

    clusters.append(len(af.cluster_centers_indices_))



plt.figure(figsize=(10,7))

plt.xlabel("Preference")

plt.ylabel("Number of clusters")

plt.plot(preferences,clusters, marker='o')
af = AffinityPropagation(preference=-140, damping=0.6, verbose=False).fit(cluster_scaled)

labels_af = pd.DataFrame(af.labels_)

n_clusters_ = len(af.cluster_centers_indices_)



clustered_data_af = cluster_data.assign(Cluster=labels_af)

scatters(clustered_data_af,'Cluster')



grouped_af = clustered_data_af.groupby(['Cluster']).mean().round(1)
grouped_af = clustered_data_af.groupby(['Cluster']).mean().round(1)

grouped_af