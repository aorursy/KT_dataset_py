from warnings import filterwarnings

filterwarnings('ignore')

import numpy as np

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

import scipy as sp

from sklearn.cluster import KMeans
df = pd.read_csv('../input/customer-segmentation/customer_segmentation.csv').copy()
df.head()
df.shape
df.info()
df.isna().sum()
df.describe().T
df.hist(figsize = (10,10));
kmeans = KMeans()

kmeans
k_means_model = kmeans.fit(df)
k_means_model.n_clusters
k_means_model.cluster_centers_
k_means_model.labels_
from yellowbrick.cluster import KElbowVisualizer

kmeans = KMeans()

visualizer = KElbowVisualizer(kmeans, k=(2,11))

visualizer.fit(df) 

visualizer.poof()
sonuclar = []
for i in range(1,11):

    k_means = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)

    k_means.fit(df)

    sonuclar.append(k_means.inertia_)
sonuclar
plt.plot(range(1,11), sonuclar)

plt.show()
kmeans = KMeans(n_clusters = 4)

kmeans
k_means_model = kmeans.fit(df)
k_means_model.n_clusters
k_means_model.cluster_centers_
k_means_model.labels_
kumeler = k_means_model.labels_
plt.scatter(df["maas"], df["aylik_harcama"], c = kumeler, s = 50, cmap = "rainbow")



merkezler = k_means_model.cluster_centers_



plt.scatter(merkezler[:,0], merkezler[:,1], c = "black", s = 200, alpha = 0.5);
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()

fig.set_size_inches(24, 20, 24)

ax = Axes3D(fig)

ax.scatter(df["maas"], df["aylik_harcama"], df["yas"], c=kumeler,  cmap = "rainbow")

ax.scatter(merkezler[:, 0], merkezler[:, 1], merkezler[:, 3], 

           marker='*', 

           c='#050505',

           cmap = "rainbow",

           s=1000);

plt.show()
df_kumelenmis = df.copy()
df_kumelenmis["kume_no"] = kumeler
df_kumelenmis.head(10)
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=4, affinity="euclidean", linkage="ward")

ac_model = ac.fit(df)
kumeler_ac = ac_model.labels_
plt.scatter(df["maas"], df["aylik_harcama"], c = kumeler_ac, s = 50, cmap = "rainbow")
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(df, method="ward"))

plt.show()
dendogram = sch.dendrogram(sch.linkage(df, method="ward"), truncate_mode = "lastp", p = 4)

plt.show()
kumeler_ac
df_kumelenmis["kume_no_ac"] = kumeler_ac
df_kumelenmis.head(10)
farklilik = 0

for i in df_kumelenmis.index:

    if df_kumelenmis.loc[i,"kume_no"] != df_kumelenmis.loc[i,"kume_no_ac"]:

        farklilik = farklilik + 1

    else:

        continue

print(farklilik, "gözlem farklı kümelenmiştir.")
plt.scatter(df["maas"], df["aylik_harcama"], c = kumeler_ac, s = 50, cmap = "rainbow")

plt.show()

plt.scatter(df["maas"], df["aylik_harcama"], c = kumeler, s = 50, cmap = "rainbow")

plt.show()
from sklearn.preprocessing import StandardScaler



df = StandardScaler().fit_transform(df)

df[0:6,0:5]
from sklearn.decomposition import PCA

pca = PCA(n_components = 3)

pca_fit = pca.fit_transform(df)
pca_fit[:5,:5]
bilesen_df = pd.DataFrame(data = pca_fit, 

                          columns = ["component_1","component_2","component_3"])
bilesen_df.head()
pca.explained_variance_ratio_
pca = PCA().fit(df)
plt.plot(np.cumsum(pca.explained_variance_ratio_))