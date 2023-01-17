import time

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from matplotlib import pyplot as plt

from mpl_toolkits.mplot3d import Axes3D



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

from sklearn.cluster import KMeans



import seaborn as sns



import os

print(os.listdir("../input/"))



np.random.seed(42)

variance_perc = 0.90

t_sne_sample = 10000
# mnist data

df_train = pd.read_csv('../input/mnist_train.csv')

df_test = pd.read_csv('../input/mnist_test.csv')



#Normalization

X_train = df_train.iloc[:, 1:785]

X_train = X_train.values.astype('float32')/255.

y_train = df_train.iloc[:, 0]



X_test = df_test.iloc[:, 1:785]

X_test = X_test.values.astype('float32')/255.

y_test = df_test.iloc[:, 0]



print(X_train.shape, X_test.shape)
# PCA

time_start = time.time()

pca = PCA()

pca_result = pca.fit_transform(X_train)

df_train['y'] = y_train

df_train['pca-one'] = pca_result[:,0]

df_train['pca-two'] = pca_result[:,1]

df_train['pca-three'] = pca_result[:,2]

print('PCA done! Time elapsed: {} seconds'.format(time.time()-time_start))
cum_ev = np.cumsum((pca.explained_variance_ratio_))

plt.plot(cum_ev)

print(cum_ev[10])

reduced_pc = cum_ev[cum_ev <= variance_perc]

n_comp = reduced_pc.shape[0]

print(n_comp)
#Top 2 PCA

plt.figure(figsize=(16,10))

sns.scatterplot(

    x="pca-one", y="pca-two",

    hue="y",

    palette=sns.color_palette("hls", 10),

    data=df_train,

    legend="full",

    alpha=0.8

)
#Top 3 PCA

ax = Axes3D(plt.figure(figsize=(16,10)))

ax.scatter(

    xs=df_train["pca-one"], 

    ys=df_train["pca-two"], 

    zs=df_train["pca-three"], 

    c=df_train["y"], 

    cmap='tab10'

)

ax.set_xlabel('pca-one')

ax.set_ylabel('pca-two')

ax.set_zlabel('pca-three')

plt.show()
#t-SNE

time_start = time.time()



rndperm = np.random.permutation(df_train.shape[0])

df_train_tsne = df_train.iloc[rndperm[:t_sne_sample],:].copy()

X_train_tsne = df_train_tsne.iloc[:, 1:785]

X_train_tsne = X_train_tsne.values.astype('float32')/255.

y_train_tsne = df_train_tsne.iloc[:, 0]



# PCA

pca2 = PCA(n_components=n_comp)

pca_result2 = pca2.fit_transform(X_train_tsne)



tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)

tsne_results = tsne.fit_transform(pca_result2)

print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
#Top 2 t-SNE

df_train_tsne['tsne-one'] = tsne_results[:,0]

df_train_tsne['tsne-two'] = tsne_results[:,1]



plt.figure(figsize=(16,10))

sns.scatterplot(

    x="tsne-one", y="tsne-two",

    hue="y",

    palette=sns.color_palette("hls", 10),

    data=df_train_tsne,

    legend="full",

    alpha=0.8

)
pca = PCA(n_components=n_comp)

pca_result = pca.fit_transform(X_train)

kmeans = KMeans(n_clusters=10)

kmeans = kmeans.fit(pca_result)

X_test_trns = pca.transform(X_test)

kmeanslabels = kmeans.predict(X_test_trns)

from sklearn import metrics

score = metrics.accuracy_score(y_test,kmeanslabels)

print(score)
plt.figure(figsize=(16,10))

sns.scatterplot(

    x="pca-one", y="pca-two",

    hue="kmeans_label",

    palette=sns.color_palette("hls", 10),

    data=df_train,

    legend="full",

    alpha=0.3

)

sns.scatterplot(

    x="x", y="x",

    hue=1,

    data=c_df,

    size=10,

    alpha=0.8

)