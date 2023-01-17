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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



data = pd.read_csv("/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv")

data.head()
len(data)
data["Gender"].value_counts()
pd.pivot_table(data = data, columns = "Gender", values = ["Annual Income (k$)", "Spending Score (1-100)"],

               aggfunc = np.mean)
pd.pivot_table(data = data, columns = "Gender", index = "Age", values = ["Annual Income (k$)", "Spending Score (1-100)"],

               aggfunc = np.mean)
import seaborn as sns

sns.scatterplot(x = data["Age"], y = data["Spending Score (1-100)"], hue = data["Gender"])
sns.scatterplot(x = data["Age"], y = data["Annual Income (k$)"], hue = data["Gender"])
sns.scatterplot(x = data["Spending Score (1-100)"], y = data["Annual Income (k$)"], hue = data["Gender"])
yas_1 = data.loc[data["Age"] < 36]

yas_2 = data.loc[data["Age"] > 35]



yas_1["Age_"] = 0

yas_2["Age_"] = 1



data_2 = pd.concat([yas_1, yas_2], axis = 0)

data_2["Age_"].value_counts()
sns.scatterplot(x = data_2["Spending Score (1-100)"], y = data_2["Annual Income (k$)"], hue = data_2["Age_"])
corr = data_2.corr()

corr
df = data_2.iloc[:, 1:6]

del df["Age"]



from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

df["Gender"] = le.fit_transform(df["Gender"])



df.head()
from sklearn.preprocessing import StandardScaler



sc = StandardScaler()



df = sc.fit_transform(df)

    

df = pd.DataFrame(df, columns = ["Gender", "Income", "Spending Score", "Age"])



from sklearn.decomposition import PCA



pca = PCA(n_components=2)



df_pca = pca.fit_transform(df)



df_pca
print(sum(pca.explained_variance_ratio_))
df_2 = data.copy() 

del df_2["CustomerID"]



df_2["Gender"] = le.fit_transform(df_2["Gender"])



df_2 = sc.fit_transform(df_2)

    

df_2 = pd.DataFrame(df_2, columns = ["Gender", "Age", "Income", "Spending Score"])



pca_2 = PCA(n_components=2)



df_2_pca = pca_2.fit_transform(df_2)



print(sum(pca_2.explained_variance_ratio_))
df_2_pca = pd.DataFrame(df_2_pca, columns = ["Veri1", "Veri2"])

plt.scatter(x = df_2_pca["Veri1"], y = df_2_pca["Veri2"])
from sklearn.cluster import KMeans



k_means = KMeans(n_clusters = 4, init = "k-means++")

x = data.iloc[:, 2:]

k_means.fit(x)



print(k_means.cluster_centers_)
liste = []



for i in range(1, 11):

    k_means = KMeans(n_clusters = i, init = "k-means++", random_state = 123)

    k_means.fit(df.iloc[:, 2:])

    liste.append(k_means.inertia_)



plt.plot(range(1, 11), liste)
x = df.iloc[:,1:3]

k_means = KMeans(n_clusters = 4, init = "k-means++")

k_fit = k_means.fit(x)

kumeler = k_fit.labels_



fig, graf = plt.subplots(figsize = (15, 6))



plt.scatter(x.iloc[:,0], x.iloc[:,1], c = kumeler, s = 50, cmap = "viridis")



merkezler = k_fit.cluster_centers_



graf = plt.scatter(merkezler[:,0], merkezler[:,1], c = "black", s = 200, alpha = 0.5);

plt.show()
x = df.iloc[:,1:3]

k_means = KMeans(n_clusters = 5, init = "k-means++")

k_fit = k_means.fit(x)

kumeler = k_fit.labels_



fig, graf = plt.subplots(figsize = (15, 6))



plt.scatter(x.iloc[:,0], x.iloc[:,1], c = kumeler, s = 50, cmap = "viridis")



merkezler = k_fit.cluster_centers_



graf = plt.scatter(merkezler[:,0], merkezler[:,1], c = "black", s = 200, alpha = 0.5);

plt.show()
x = df.iloc[:,1:3]

k_means = KMeans(n_clusters = 6, init = "k-means++")

k_fit = k_means.fit(x)

kumeler = k_fit.labels_



fig, graf = plt.subplots(figsize = (15, 6))



plt.scatter(x.iloc[:,0], x.iloc[:,1], c = kumeler, s = 50, cmap = "viridis")



merkezler = k_fit.cluster_centers_



graf = plt.scatter(merkezler[:,0], merkezler[:,1], c = "black", s = 200, alpha = 0.5);

plt.show()
x = df.iloc[:,1:3]

k_means = KMeans(n_clusters = 5, init = "k-means++")

k_fit = k_means.fit(x)

kumeler = k_fit.labels_



fig, graf = plt.subplots(figsize = (15, 6))



plt.scatter(x.iloc[:,0], x.iloc[:,1], c = kumeler, s = 50, cmap = "viridis")



merkezler = k_fit.cluster_centers_



graf = plt.scatter(merkezler[:,0], merkezler[:,1], c = "black", s = 200, alpha = 0.5);

plt.show()