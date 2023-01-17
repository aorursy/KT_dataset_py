import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

from scipy.spatial.distance import cdist

from mpl_toolkits.axes_grid1 import host_subplot

from sklearn import metrics

import matplotlib.gridspec as gridspec

import math

import seaborn as sns





#import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))

path = '/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv'

data = pd.read_csv(path)
for column in data.columns:

    print(column, ' - ', data[column].dtype)
for col in data.columns:

    nan = round(data[col].isnull().sum() / len(data[col]) * 100, 2)

    print(col, ' : ', nan, '%')
data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].hist(figsize = (20,20))

plt.show()
corr = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].corr()

# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1,vmin = -1, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
# Standarize data

scaler = StandardScaler()

data_sc = scaler.fit_transform(data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])

data_sc = pd.DataFrame(data_sc, columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)'])
def elbow_plot_for_k_means(X, k_range = range(2,15)):

    distortions = []

    for k in k_range:

        kmeanModel = KMeans(n_clusters=k).fit(X)

        kmeanModel.fit(X)

        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0]) 

    reduction = [distortions[i-1] - distortions[i] for i in range(1,len(distortions))]

    reduction.insert(0,np.nan)

    # Plot    

    plt.figure(figsize=[12,8])

    host = host_subplot(111)

    par = host.twinx()

    host.set_title("Optimal number of clusters")

    host.set_xlabel("k")

    host.set_ylabel("Distortion")

    par.set_ylabel("Distortion reduction")

    plt.xticks(k_range)

    p1, = host.plot(k_range, distortions)

    p2, = par.plot(k_range, reduction)

    host.yaxis.get_label().set_color(p1.get_color())

    par.yaxis.get_label().set_color(p2.get_color())

    plt.show()
elbow_plot_for_k_means(data_sc)
def ch_plot_for_k_means(X, k_range = range(2,15), resample = 3):

    plt.figure(figsize=[12,8])

    for i in range(3):

        scores = []

        for k in k_range:

            kmeansModel = KMeans(n_clusters = k).fit(X)

            labels = kmeansModel.labels_

            scores.append(metrics.calinski_harabasz_score(X, labels)) 

        plt.plot(k_range, scores)

    plt.xticks(k_range)

    plt.title("Optimal number of clusters Calinski-Harabasz criterion")

    plt.xlabel("k")

    plt.ylabel("Calinski - Harabasz statistic")

    plt.show()
ch_plot_for_k_means(data_sc, k_range = range(2,10))
class KMeansCust(KMeans):

    """

    Custumized KMeans Class. Add new prediction method, returned cluster label and distances to each cluster. 

    It's usefull when using distances as explanatory variables in other model.

    """    

    def predict_dist(self, X, columns = None, scaler = True, join = True):

        """

        There arent missing values in X. 

        IN: X - data (pandas, numpy)

            columns - if X is pandas obiect then its possible to indicate columns and X can have redundant columns.

            scaler - if True then StandardScaler

            inplace - if True results are joined to X and then 

        """

        if columns != None: data = X[columns]

        else: data = X

        if scaler:

            scaler = StandardScaler()

            data = scaler.fit_transform(data)

        df = pd.DataFrame(np.array(cdist(data, self.cluster_centers_, 'euclidean')))

        df.columns = ['dist_clu_' + str(i) for i in range(len(df.columns))]

        labels = self.predict(data)

        df['clu_label'] = list(labels)

        if join:

            X_ = X.copy()

            X_.reset_index(inplace = True, drop = True)

            df.reset_index(inplace = True, drop = True)

            return X_.join(df, how = 'left')

        else:

            return df
kmeans_model = KMeansCust(n_clusters = 6).fit(data_sc)

data_pred = kmeans_model.predict_dist(X = data, columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)'])
def kmeans_present_cluster_stats(data, label, cols = None):

    """

    """

    means = data.groupby(label).mean()

    rows = math.ceil(len(means.columns)/5)

    fig = plt.figure(constrained_layout=False, figsize=(10, rows * 5))

    gs = gridspec.GridSpec(rows, 5, figure=fig)

    i=0

    for col in means.columns:

        if col in cols:

            ax = fig.add_subplot(gs[i])

            if i % 5 == 0:

                ax.set_yticks([i + 0.5 for i in means.index])

                ax.set_yticklabels(['Cluster ' + str(i) for i in means.index])

                ax.set_ylabel('Mean by cluster')

            else:

                ax.set_yticks([])

            ax.set_xticks([0.5])

            ax.set_xticklabels([col], rotation = 45)

            col_t = [[i] for i in list(means[col])] 

            im = ax.pcolormesh(col_t, cmap="RdYlBu_r")

            plt.colorbar(im)

            i+=1

    plt.tight_layout()

    plt.show()
kmeans_present_cluster_stats(data = data_pred, label = 'clu_label', cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)'])