import numpy as np

import pandas as pd

df = pd.read_csv('../input/matplotlib-datasets/iris_dataset.csv')
df.head()
df2 = df.drop('species',axis = 1)
df2.head()
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

df2_scaled = ss.fit_transform(df2)

df2
cov_matrix = np.cov(df2_scaled.T)

cov_matrix
import seaborn as sns

sns.heatmap(cov_matrix)
eigen_values,eigen_vectors = np.linalg.eig(cov_matrix)

print('Eigen Values\n',eigen_values)

print()

print('Eigen Vectors\n',eigen_vectors)
sorted_eig_vals = pd.Series(eigen_values).sort_values(ascending =False)

pcs = eigen_vectors[:,list(sorted_eig_vals.index)].T

pcs
tot = np.sum(sorted_eig_vals)

var_exp = [(i/tot) * 100 for i in sorted_eig_vals]

cum_var_exp = np.cumsum(var_exp)

print('Explained variance by each principal component : \n [PC1,PC2,PC3,PC4]\n',var_exp)

print()

print('Cumulative explained variance : ',cum_var_exp)
selected_pcs = pcs[:2]

selected_pcs
tranps_select_pcs = selected_pcs.T

tranps_select_pcs
projected_data = np.dot(df2_scaled,tranps_select_pcs)

projected_data.shape
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters= 3,n_init=15,random_state=2)

kmeans.fit(projected_data)
kmeans.inertia_
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters= 3,n_init=15,random_state=2)

kmeans.fit(df2_scaled)
kmeans.inertia_