%matplotlib inline

from pprint import PrettyPrinter

from typing import Dict, List

import pandas as pd 

import numpy as np 

from sklearn.model_selection import train_test_split 

from sklearn.metrics import silhouette_score

from sklearn.cluster import KMeans

# Graphs 

import matplotlib.pyplot as plt 

from matplotlib.colors import ListedColormap

import seaborn as sns 

plt.rcParams["figure.figsize"] = (20,10)

plt.rcParams["font.size"] = 16

plt.style.use("ggplot")
# Constants 

color_map: List[str] = ['#F6FA06', '#06ACFA']

colors: List[str] = np.array( ['red', 'yellow', 'green', 'yellow', 'purple','orange', 'blue', 'crimson', 'salmon', 'darksalmon','firebrick', 'plum', 'gold', 'violet'])

scores: List[float] = [] 
df = pd.read_csv("../input/clustering-data.csv")

df.index = np.arange(1,len(df) +1 )
# Seed 

np.random.seed(10000)

df.shape 
sns.heatmap(df.corr() )
df.corr() 
df.cov()
# correlation

print(f"[\033[92m+\033[0m] Correlation : {df.corr().to_dict() }")

# covariance 

print(f"[\033[92m+\033[0m] Covariance :  {df.cov().to_dict() } ")
# Column to group clusters 

df['color_map'] = np.random.randint(0,1,size=len(df))
# Question A 

df.plot(kind="scatter", x='x', y='y', c='color_map', colormap=ListedColormap(color_map))

plt.title("Correlation between X and Y variables")

plt.grid(True)
# Question B: Graph different Cluster from n_cluster=2, n_cluster=12

for k in range(2,13):

    print(f"[\033[92m+\033[0m] \033[42m Number of clusters: {k}\033[0m")

    kmeans = KMeans(n_clusters=k, random_state=np.random.randint(100,150))

    kmeans.fit(df)

    # store the labels

    df[f'cluster_{k}'] = kmeans.labels_



    # Get the mean of the nth cluster as a bench mark point 

    centers = df.groupby(f"cluster_{k}").mean() # center means

    

    # Compute the silhouette score 

    scores.append( silhouette_score(df, df[f'cluster_{k}']) )

    # Plot x vs y 

    plt.scatter(df.x, df.y, c=colors[df[f"cluster_{k}"]] ,s=40)

    # Plot cluster center 

    plt.scatter(centers.x, centers.y, linewidths=5, marker='x', s=200, c='black')

    plt.title(f"K-Means with n_cluster={k}")

    plt.grid(True)

    plt.savefig(f"cluster_{k}.jpg")

    plt.show() 

# Graph Cluster Assignment

# Answer: In order to evaluate Clustering peformance and unkwon True cluster assignments, I'm going to use sillhouette score. The score ranges from -1 (worst) to 1 (best)

pd.DataFrame( {'silhouette_score' : scores }, index=range(2,13)).plot() 

plt.title("Silhouette Score Distributions")

plt.xticks(range(2,14,1))

plt.xlabel("Number of clusters")

plt.ylabel("Silhouette Score")

plt.grid(True)
pd.DataFrame( kmeans.cluster_centers_)
elbow_method: List[float] = []

scores: List[float] = [] # Over write constant 

for k in range(2,100):

    print(f"[\033[92m+\033[0m] \033[42m Number of clusters: {k}\033[0m")

    kmeans = KMeans(n_clusters=k, random_state=np.random.randint(100,170))

    kmeans.fit(df)

    scores.append(silhouette_score(df, kmeans.labels_))

    elbow_method.append(kmeans.inertia_)
pd.DataFrame({'Silhouette Score': scores}, index=range(2,100)).plot() 

plt.grid(True)

plt.xticks(range(0,100,5))

plt.xlabel("Number of Clusters")

plt.ylabel("Silhouette Coefficients")
pd.DataFrame(np.array(elbow_method), columns=['inertia'], index=range(2,100)).plot(figsize=(30,10)) 

plt.xticks(range(0,101,2))

plt.xlabel("Number of Clusters")

plt.ylabel("Inertia")

plt.title("Elbow Method when n_cluster=9")

plt.grid(True)
em = pd.DataFrame(np.array(elbow_method), columns=['inertia'], index=range(2,100))
em[ em['inertia'] == float(em[4:12].min()) ]
em.describe() 
for index in range(6,12):

    inertia: float = em[ em.index == index].to_dict().get('inertia').get(index)

    print( f"[+] n_cluster={index}\tInertia={inertia:.3f}")
sc = pd.DataFrame(data=np.array(scores), columns=['silhoutte_score'], index=range(2,100))
sc[ sc['silhoutte_score'] == sc['silhoutte_score'].max() ]