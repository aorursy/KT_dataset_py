import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))

df = pd.read_csv("../input/forestfires.csv")
df.head()
df_coordinates = df.loc[:, ["X", "Y"]]
coordinates = df_coordinates.values
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 20):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, 
                    n_init = 10, random_state = 0)
    #max_iter - max number of iteration to define the final clusers
    #n_init - number of k_means algorithm running
    kmeans.fit(coordinates)
    wcss.append(kmeans.inertia_)
    #inertia_ Sum of squared distances of samples to their closest cluster center.
plt.plot(range(1, 20), wcss)
plt.title('Define the number of clusters')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
from scipy.spatial import distance
from sklearn.metrics.pairwise import euclidean_distances
plt.figure()
kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, 
                    n_init = 10, random_state = 0)
clusters = kmeans.fit_predict(coordinates)
df["Cluster"]= clusters
plt.subplot()
plt.scatter(df['X'].values, df['Y'].values, marker='o', c=clusters, alpha=0.8)
plt.title("Clusters")
plt.show()
centroids = kmeans.cluster_centers_
print(centroids)
print("\nCalculating distance between clusters\n")
print(euclidean_distances(centroids,centroids))
df_cluster0 = df[(df["Cluster"] == 0)] 
df_cluster0.head()
import seaborn as sns
import numpy as np
def build_cluster_corr(df_cluster):
    df_cluster_indicators = df_cluster.loc[:, ["area","FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain"]]
    plt.clf()
    plt.figure(figsize=(10,10))
    cmap = sns.diverging_palette(20, h_pos=220, s=75, l=50, sep=10, center='light', as_cmap=True)     
    corr_matrix = df_cluster_indicators.corr()
    corr_matrix[np.abs(corr_matrix) < 0.65] = 0
    sns.heatmap(corr_matrix, cmap=cmap, annot=True)     
    plt.show()
build_cluster_corr(df_cluster0)
df_cluster0_indicators = df_cluster0.loc[:, ["FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind"]]
X = df_cluster0_indicators.values
Y = df_cluster0['area'].values
SL = 0.05
X_opt_ = X[:, [0, 1, 2, 3, 4, 5, 6]]
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import numpy as np
def forward_selection(x, y, sl):
    result = np.empty((len(X),1))
    numVars = len(x[0])
    all_regressors_OLS = smf.OLS(y, x).fit()
    maxVar = max(all_regressors_OLS.pvalues).astype(float)
    for i in range(0, numVars):
        regressor_OLS = smf.OLS(y, x[:,i]).fit()
        for j in range(0, numVars - i):
            p = regressor_OLS.pvalues[0].astype(float)
            if p > sl:
                if (p == maxVar):
                    result = np.insert(result, 0, j, axis=1)
                    
    plt.figure(figsize=(10,10))
    plt.scatter(result, y, color = 'red')
    plt.plot(result, regressor_OLS.predict(result), color = 'blue')
    plt.title('Forward Selection results')
    plt.xlabel('Predictor')
    plt.ylabel('area')
    plt.show()
    print(regressor_OLS.summary())
    print(result)
    return result

X_Modeled_ = forward_selection(X_opt_, Y, SL)
def backward_elimination(x, y, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = smf.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
                    
    plt.figure(figsize=(10,10))
    plt.scatter(x, y, color = 'red')
    plt.plot(x, regressor_OLS.predict(x), color = 'blue')
    plt.title('Backward Elimination results')
    plt.xlabel('Predictor')
    plt.ylabel('area')
    plt.show()
    print(regressor_OLS.summary())
    print(x)
    return x

X_Modeled_ = backward_elimination(X_opt_, Y, SL)
df_cluster1 = df[(df["Cluster"] == 1)] 
df_cluster1.head()
build_cluster_corr(df_cluster1)
df_cluster1_indicators = df_cluster1.loc[:, ["FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", 'rain']]
X = df_cluster1_indicators.values
Y = df_cluster1['area'].values
SL = 0.05
X_opt_ = X[:, [0, 1, 2, 3, 4, 5, 6, 7]]
X_Modeled_ = backward_elimination(X_opt_, Y, SL)
df_cluster2 = df[(df["Cluster"] == 2)] 
df_cluster2.head()
build_cluster_corr(df_cluster2)
df_cluster2_indicators = df_cluster2.loc[:, ["FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", 'rain']]
X = df_cluster2_indicators.values
Y = df_cluster2['area'].values
SL = 0.05
X_opt_ = X[:, [0, 1, 2, 3, 4, 5, 6, 7]]
X_Modeled_ = backward_elimination(X_opt_, Y, SL)
df_cluster3 = df[(df["Cluster"] == 3)] 
df_cluster3.head()
build_cluster_corr(df_cluster3)
df_cluster3_indicators = df_cluster2.loc[:, ["FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind"]]
X = df_cluster3_indicators.values
Y = df_cluster3['area'].values
SL = 0.05
X_opt_ = X[:, [0, 1, 2, 3, 4, 5, 6]]
X_Modeled_ = backward_elimination(X_opt_, Y, SL)
df_cluster4 = df[(df["Cluster"] == 4)] 
df_cluster4.head()
build_cluster_corr(df_cluster4)
df_cluster4_indicators = df_cluster4.loc[:, ["FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", 'rain']]
X = df_cluster4_indicators.values
Y = df_cluster4['area'].values
SL = 0.05
X_opt_ = X[:, [0, 1, 2, 3, 4, 5, 6, 7]]
X_Modeled_ = backward_elimination(X_opt_, Y, SL)