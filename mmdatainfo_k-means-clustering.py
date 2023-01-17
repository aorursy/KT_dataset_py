# Scikit-learn

from sklearn import datasets

from sklearn.cluster import KMeans

from sklearn.metrics import adjusted_rand_score

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.decomposition import PCA

# other libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
# Use vector drawing inside jupyter notebook

%config InlineBackend.figure_format = "svg"

# Set matplotlib default axis font size (inside this notebook)

plt.rcParams.update({'font.size': 8})
iris = datasets.load_iris()

# This is extra step that can be omitted but Pandas DataFrame contains some powerfull features

df = pd.DataFrame(iris.data,columns=iris.feature_names)

df = df.assign(target=iris.target)
# Compute selected stats

dfinfo = pd.DataFrame(df.dtypes,columns=["dtypes"])

for (m,n) in zip([df.count(),df.isna().sum()],["count","isna"]):

    dfinfo = dfinfo.merge(pd.DataFrame(m,columns=[n]),right_index=True,left_index=True,how="inner");

# Add to `describe` output

dfinfo.T.append(df.describe())
df.drop(["target"],axis=1).corr().abs().round(2).style.background_gradient(cmap="viridis")
scx = StandardScaler();

X = scx.fit_transform(df.drop(["target"],axis=1).values)
kchoose = np.arange(1,6)

mean_dist = np.array([]);

# Plot true

plt.figure(figsize=(9,2.5));

plt.subplot(1,kchoose.size,1)

plt.scatter(df["petal length (cm)"],df["sepal length (cm)"],c=df["target"]);

plt.title("IRIS labels")

# Fit model for each k

for (i,k) in enumerate(kchoose):

    km = KMeans(n_clusters=k).fit(X)

    mean_dist = np.append(mean_dist,km.inertia_)

    if k != 1:

        plt.subplot(1,kchoose.size,i+1)

        label = km.labels_

        plt.scatter(df["petal length (cm)"],df["sepal length (cm)"],c=km.labels_)

        plt.title("k choice = {}".format(k))

        # add centroid

        temp = scx.inverse_transform(km.cluster_centers_);

        plt.plot(temp[:,2],temp[:,0],"r.")
plt.figure(figsize=(5,2))

plt.plot(kchoose,mean_dist,"ko-");

plt.xlabel("k",fontsize=10)

plt.ylabel("$\sum$ distances$^2$ centroid",fontsize=9);

plt.title("Visual method to choose k");
for i in range(0,3):

    print(km.predict(km.cluster_centers_[i,:].reshape(1,-1))[0])