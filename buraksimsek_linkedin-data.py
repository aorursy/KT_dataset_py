# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tqdm import tqdm
df = pd.read_csv("../input/dump.csv")
df.head()
df.isnull().sum()
df.info()
df1 = df.fillna(df.mean())
df1.isnull().sum()
df1.info()
X = pd.concat([df1.ageEstimate, df1.companyFollowerCount, df1.companyStaffCount, df1.connectionsCount, df1.followable, df1.followersCount,
               df1.isPremium, df1.positionId, df1.avgMemberPosDuration, df1.avgCompanyPosDuration], axis = 1)
X
X.isnull().sum()
X.describe()
plt.subplots(figsize = (15, 10))
sns.heatmap(df.corr(), annot = True, cmap = "PuBu")
plt.title("Linkedin Correlation", fontsize = 16)
plt.show()
sns.set_style("darkgrid")
sns.lmplot(data= df, y = "companyFollowerCount", x = "companyStaffCount", height = 10)
plt.title("Company Follower Count V Company Staff Count", fontsize = 16)
plt.show()
sns.set(style = "white", font_scale = 1.2, rc = {'figure.figsize' :(20,20)})
ax = X.hist(bins=20,color = "lightblue")
plt.subplots(figsize = (12, 8))
sns.countplot(x = "genderEstimate", data = df, hue = "isPremium", palette = "BuPu")
plt.title("Use of Premium Memberships by Gender")
fig = plt.figure(figsize = (15,16))
ax1 = fig.add_subplot(2,1,1)
ax1.set_title("Age Distribution of Male Users", fontsize = 16)
ax1.tick_params(labelbottom =  "off", axis = "x")
sns.countplot(df[df.genderEstimate == "male"].ageEstimate);
plt.xticks(rotation= 90, fontsize = 12)

ax2 = fig.add_subplot(2,1,2)
ax2.set_title("Age Distribution of Female Users", fontsize = 16)
sns.countplot(df[df.genderEstimate == "female"].ageEstimate);
plt.xticks(rotation= 90, fontsize = 12)
print("Different Company Participating in Linkedin (Top-10)")
df["companyName"].value_counts().head(10)
plt.style.use("dark_background")
df["companyName"].value_counts().head(50).plot.bar(color = "yellow", figsize = (20, 8), fontsize = 16)
plt.title("Different Company Participating in Linkedin", fontsize = 20, fontweight = 15)
plt.xlabel("Name of The Company")
plt.ylabel("Count")
plt.show()
print("Different Nations Participating in Linkedin (Top-10)")
df["posLocation"].value_counts().head(10)
plt.style.use("dark_background")
df["posLocation"].value_counts().head(50).plot.bar(color = "yellow", figsize = (20, 8), fontsize = 16)
plt.title("Different Nations Participating in Linkedin", fontsize = 20, fontweight = 15)
plt.xlabel("Name of The Nations")
plt.ylabel("Count")
plt.show()
print("Different Professions Participating in Linkedin (Top-10)")
df["posTitle"].value_counts().head(10)
plt.style.use("dark_background")
df["posTitle"].value_counts().head(50).plot.bar(color = "yellow", figsize = (20, 8), fontsize = 16)
plt.title("Different Professions Participating in Linkedin", fontsize = 20, fontweight = 20)
plt.xlabel("Name of The Country")
plt.ylabel("Count")
plt.show()
from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler()
X_cluster = X.copy()
X_cluster[X_cluster.columns] = std_scaler.fit_transform(X_cluster)
X_cluster.describe()
from sklearn.decomposition import PCA
pca_2 = PCA(2)
pca_2_result = pca_2.fit_transform(X_cluster)

print ("Cumulative Variance Explained by Two Principal Components:{:.2%}".format(np.sum(pca_2.explained_variance_ratio_)))
sns.set(style = "white", rc = {"figure.figsize" : (15,8)}, font_scale = 1.1)

plt.scatter(x = pca_2_result[:, 0], y = pca_2_result[:, 1], color = "lightblue", lw = 0.1)
plt.title("Data Represented by the Two Strongest Principal Components", fontweight = "bold")
plt.show()
import sklearn.cluster as cluster

inertia = []
for i in tqdm(range(2,10)):
    kmeans = cluster.KMeans(n_clusters = i,
               init = "k-means++",
               n_init = 15,
               max_iter = 500,
               random_state = 17)
    kmeans.fit(X_cluster)
    inertia.append(kmeans.inertia_)
sns.set(style = "white", font_scale = 1.1, rc = {"figure.figsize":(9,9)})

plt.plot(range(2,len(inertia)+2), inertia, marker = "o", lw = 2, ms = 8,color = "darkblue")
plt.title("K-means Inertia", fontweight = "bold")
plt.grid(True)

plt.show()
kmeans = cluster.KMeans(n_clusters = 5, random_state = 17,init = "k-means++")
kmeans_labels = kmeans.fit_predict(X_cluster)

centroids = kmeans.cluster_centers_
centroids_pca = pca_2.transform(centroids)

pd.Series(kmeans_labels).value_counts()
X2 = X.copy()
X2["Cluster"] = kmeans_labels

aux = X2.columns.tolist()
aux[0:len(aux)-1]

for cluster in aux[0:len(aux)-1]:
    grid = sns.FacetGrid(X2, col = "Cluster")
    grid.map(plt.hist, cluster, color = "darkblue")
sns.set(style = "white", rc = {"figure.figsize":(20,10)}, font_scale=1.1)

plt.scatter(x = pca_2_result[:, 0], y = pca_2_result[:, 1], c = kmeans_labels, cmap = "tab20b")
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
            marker = "x", s = 150, linewidths = 3,
            color = "red", zorder = 10, lw = 5)
plt.title("Clustered Data (PCA Visualization)", fontweight = "bold")
plt.show()