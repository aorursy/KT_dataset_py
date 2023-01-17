import numpy as np

import pandas as pd



import seaborn as sea



from scipy.stats import multivariate_normal

from sklearn.preprocessing import StandardScaler

from sklearn import metrics

from sklearn.cluster import KMeans

from sklearn.cluster import AgglomerativeClustering

from sklearn.cluster import DBSCAN



import matplotlib.pyplot as plt
sea.set_style("darkgrid")
mean = [[ 4.0,  2.0],

        [10.0,  7.5],

        [ 2.0,  8.5],

        [ 1.0, -3.5]]



covariance = [[[9,  0.1], [ 0.1, 1]],

              [[2, -0.8], [-0.8, 3]],

              [[1,  0.8], [ 0.8, 2]],

              [[2,  0.0], [ 0.0, 2]]]

size = 1000

data = []

label = []

for c in range(4):

    mlt_nrm = multivariate_normal(mean[c], covariance[c])

    data.extend(mlt_nrm.rvs(size = size, random_state = 0))

    label.extend(c*np.ones(size))

data = np.array(data)

label = np.array(label)
scaler = StandardScaler()

data_S = scaler.fit_transform(data)
fig, axes = plt.subplots(1, 2, figsize=(10, 5))



sea.scatterplot(x = data[:,0], y = data[:,1], hue = label,

                palette = "rocket", ax=axes[0]);

sea.scatterplot(x = data_S[:,0], y = data_S[:,1], hue = label,

                palette = "rocket", ax=axes[1]);



axes[0].set_title("Dataset")

axes[0].set_ylabel("Feature 1")

axes[0].set_xlabel("Feature 0");

axes[1].set_title("Standardized Dataset")

axes[1].set_ylabel("Feature 1")

axes[1].set_xlabel("Feature 0");
SC = []

WCSS = []



K_set = range(2,11)

for K in K_set:

    kmeans = KMeans(n_clusters=K, random_state=1).fit(data_S)

    SC.append(metrics.silhouette_score

                 (data_S, kmeans.labels_, metric="euclidean"))

    WCSS.append(kmeans.inertia_)

    

fig, axes = plt.subplots(constrained_layout = True,

                         nrows=1, ncols=2, figsize=(10, 5))



sea.lineplot(x = K_set, y = SC, color = "#111111", ax=axes[0]);

sea.lineplot(x = K_set, y = WCSS, color = "#111111", ax=axes[1]);



axes[0].axvline(x=4, linestyle = "--", color ="#111111", alpha = 0.5)

axes[1].axvline(x=4, linestyle = "--", color ="#111111", alpha = 0.5)



axes[0].set_ylabel("SC")

axes[0].set_xlabel("K");

axes[1].set_ylabel("WCSS")

axes[1].set_xlabel("K");
kmeans = KMeans(n_clusters=4, random_state=1).fit(data_S)



plt.figure(figsize=(6,6))

sea.scatterplot(x = data_S[:,0], y = data_S[:,1], hue = kmeans.labels_,

                palette = "rocket");

plt.legend().remove()

plt.title("Clustered Data with K=4")

plt.xlabel("Feature 0")

plt.ylabel("Feature 1");
SC = []

link = []

n_clusters = []



linkage = ["ward", "complete", "average", "single"]

for l in linkage:

    for n in range(2, 10, 1):

        agg = AgglomerativeClustering(linkage=l,

                                      n_clusters = n).fit(data_S)

        SC.append(metrics.silhouette_score(data_S, agg.labels_,

                                      metric="euclidean"))

        link.append(l)

        n_clusters.append(n)
print(" linkage    n_clusters      SC")

print("---------  ------------   ------")

for i in np.array(SC).argsort()[::-1][:6]:

    print(" {:10}     {}         {:.4f}".format(link[i],

                                                n_clusters[i],

                                                SC[i]))
fig, axes = plt.subplots(constrained_layout = True,

                         nrows=3, ncols=2, figsize=(10, 12))



for f, i in enumerate(np.array(SC).argsort()[::-1][:6]):

    agg = AgglomerativeClustering(linkage=link[i],

                                  n_clusters = n_clusters[i]).fit(data_S)

    sea.scatterplot(x = data_S[:,0], y = data_S[:,1],

                    palette = "rocket", legend="full",

                    hue = agg.labels_, ax=axes[f//2][f%2]);

    axes[f//2][f%2].set_title("linkage: {}, n_clusters: {},\nSC: {:.4f}". \

                              format(link[i],n_clusters[i],SC[i]))
dbscan = DBSCAN().fit(data_S)

dbscan.fit(data_S)

plt.figure(figsize=(6,6))

sea.scatterplot(x = data_S[:,0], y = data_S[:,1],

                hue = dbscan.labels_, palette = "rocket");

plt.title("DBCSAN with default parameters");
SC = []

eps = []

min_samples = []



for d in np.arange(0.1, 0.2, 0.01):

    for m in range(10, 30, 2):

        dbscan = DBSCAN(eps=d, min_samples=m).fit(data_S)

        SC.append(metrics.silhouette_score(data_S, dbscan.labels_,

                                           metric="euclidean"))

        eps.append(d)

        min_samples.append(m)
print(" eps      min_sample      SC")

print("------   ------------   ------")

for i in np.array(SC).argsort()[::-1][:6]:

    print(" {:.2f}         {}        {:.4f}".format(eps[i],

                                                    min_samples[i],

                                                    SC[i]))
fig, axes = plt.subplots(constrained_layout = True,

                         nrows=3, ncols=2, figsize=(10, 12))



for f, i in enumerate(np.array(SC).argsort()[::-1][:6]):

    dbscan = DBSCAN(eps=eps[i], min_samples=min_samples[i]).fit(data_S)

    sea.scatterplot(x = data_S[:,0], y = data_S[:,1],

                    palette = "rocket", legend="full",

                    hue = dbscan.labels_, ax=axes[f//2][f%2]);

    noc = len(np.unique(dbscan.labels_)) - 1

    axes[f//2][f%2].set_title("eps: {:.2f}, min_samples: {},\nSC: {:.4f},"

                              " number of detected clusters: {}". \

                              format(eps[i],min_samples[i],SC[i],noc))