import pandas as pd # for creating dataframe from numerical array 

import numpy as np # for handling all the mathematics 

import seaborn as sns # for the better visualization which will help you learn how dimensionality reduction happens

import matplotlib.pyplot as plt

from sklearn import datasets # for the import of mniset dataset by another way

from sklearn import manifold# to perform the t-SNE



%matplotlib inline
df = pd.read_csv("../input/mnist-in-csv/mnist_train.csv")

df
# removing the label from the dataset as we are not gonna ever doing dimension reduction on our label

df1 = df.iloc[:,1:]

df1
from sklearn.preprocessing import StandardScaler

df_std = StandardScaler().fit_transform(df1)

df_std
df_cov_matrix = np.cov(df_std.T)

df_cov_matrix
eig_vals, eig_vecs = np.linalg.eig(df_cov_matrix)

print(eig_vecs)

print(eig_vals)
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

print("Eigenvalues in descending order:")

for i in eig_pairs:

 print(i[0])
total = sum(eig_vals)

var_exp = [(i / total)*100 for i in sorted(eig_vals, reverse=True)]

cum_var_exp = np.cumsum(var_exp)

print("Variance captured by each component is",var_exp)

print("Cumulative variance captured as we travel with each component",cum_var_exp)
from sklearn.decomposition import PCA

pca = PCA().fit(df_std)

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel("No of components")

plt.ylabel("Cumulative explained variance")

plt.show()
from sklearn.decomposition import PCA

pca = PCA(n_components = 10)

pcs = pca.fit_transform(df_std)

df_new = pd.DataFrame(data=pcs, columns={"PC1","PC2","PC3","PC4","PC5","PC6","PC7","PC8","PC9","PC10"})

df_new["target"] = df["label"]
sns.lmplot(x='PC1',

           y='PC2',

           data=df_new,

           fit_reg=False,

           legend=True,

           size=9,

           hue='target',

           scatter_kws={"s":80})
data = datasets.fetch_openml('mnist_784',version=1,return_X_y = True)

pixel_value, target = data

targets = target.astype(int)
tnse = manifold.TSNE(n_components=2,random_state = 42)

new_data = tnse.fit_transform(pixel_value[:3000,:])
tnse_df = pd.DataFrame(np.column_stack((new_data,targets[:3000])),

                      columns = ["x","y","targets"])

tnse_df.loc[:,"targets"] = tnse_df.targets.astype(int)
grid = sns.FacetGrid(tnse_df,hue="targets",size=8)

grid.map(plt.scatter,"x","y").add_legend()