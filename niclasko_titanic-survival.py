import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot

from scipy.cluster.vq import whiten

from scipy.cluster.vq import kmeans, vq

from sklearn.decomposition import PCA



# Read training data

data_train = pd.read_csv("../input/train.csv")

# One hot code categorical feature variables

# and leave continuous/ordinal variables as they are

data_train_onehot = pd.get_dummies(

    data_train[["Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked"]]

)

# Set NaN's to zero

data_train_onehot.loc[data_train_onehot["Age"]*0 != 0, "Age"] = 0
# Perform Singular Value Decomposition of rescaled (to unit variance) feature matrix

U, s, V = np.linalg.svd(

    whiten(data_train_onehot),

    full_matrices=True

)

# Project feature matrix onto the 2 first right singular vectors, i.e. from V

# Assign to new dataframe called data_train_onehot_dr (dimensionality reduced)

data_train_onehot_svd_dr = pd.DataFrame(

    np.dot(data_train_onehot, V[:,1:3]),

    columns=["X","Y"]

)



# PCA

data_train_onehot_pca = PCA().fit(whiten(data_train_onehot))

data_train_onehot_pca_dr = pd.DataFrame(

    np.dot(data_train_onehot, np.transpose(data_train_onehot_pca.components_[1:3])),

    columns=["X","Y"]

)



# Cluster PCA dimensionality reduced data using K-means

data_train_onehot_centroids, _ = kmeans(

    data_train_onehot_pca_dr,

    5

)

data_train["Cluster"], _ = vq(

    data_train_onehot_pca_dr,

    data_train_onehot_centroids

)
# Visualise clusters

pyplot.scatter(

    data_train_onehot_pca_dr["X"],

    data_train_onehot_pca_dr["Y"],

    c=data_train["Cluster"],

    s=3

)



pyplot.show()



pyplot.scatter(

    data_train_onehot_svd_dr["X"],

    data_train_onehot_svd_dr["Y"],

    c=data_train["Cluster"],

    s=3

)



pyplot.show()
round(pd.crosstab(

    data_train["Cluster"],

    data_train["Survived"]

).apply(lambda r: r/r.sum(), axis=1)*100)