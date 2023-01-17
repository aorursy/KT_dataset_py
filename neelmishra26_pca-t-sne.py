import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore") #just imported so that no warning is showed 
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv(r"/kaggle/input/digit-recognizer/train.csv") #reading the csv file as a dataframe

data.head() #displaying the first five records
data.shape
l=data['label']
d=data.drop("label",axis=1)
l
data
d
d.shape
l.shape
l_1000=l[0:1000]
l_1000.shape
plt.figure(figsize=(8,8))

idx=850

grid_data=d.iloc[idx].as_matrix().reshape(28,28)

plt.imshow(grid_data, interpolation="none", cmap="gray")

plt.show()

print(l[idx])
from sklearn.preprocessing import StandardScaler

standard_data=StandardScaler().fit_transform(d)
d_1000=standard_data[0:1000,:]
d_1000.shape
from sklearn import decomposition

pca=decomposition.PCA(n_components=2)

pca_data=pca.fit_transform(standard_data)

pca_data.shape
pca_data_final=np.vstack((pca_data.T,l)).T

pca_df=pd.DataFrame(data=pca_data_final,columns=("PCA1","PCA2","label"))

sns.FacetGrid(pca_df,hue="label",size=6).map(plt.scatter,'PCA1','PCA2').add_legend()

plt.show()
pca_1=decomposition.PCA(n_components=784)

pca_data_1=pca_1.fit_transform(standard_data)

pca_data_1.shape
percentage_var_explained=pca_1.explained_variance_/np.sum(pca_1.explained_variance_)

n_components_1=784

cum_var_explained=np.cumsum(percentage_var_explained)

plt.figure(1,figsize=(6,4))

plt.clf()

plt.plot(cum_var_explained,linewidth=2)

plt.axis('tight')

plt.grid()

plt.xlabel('n_components_1')

plt.ylabel('cumalative_explained_variance')

plt.show()
from sklearn.manifold import TSNE

model=TSNE(n_components=2,perplexity=50.0,n_iter=10000)

tsne_data=model.fit_transform(d_1000)

tsne_data.shape
tsne_data_final=np.vstack((tsne_data.T,l_1000)).T

tsne_df=pd.DataFrame(data=tsne_data_final,columns=("DIM1","DIM2","label"))

sns.FacetGrid(tsne_df,hue="label",size=6).map(plt.scatter,'DIM1','DIM2').add_legend()

plt.show()
from sklearn.manifold import TSNE

model=TSNE(n_components=2,perplexity=50.0,n_iter=10000)

tsne_data_1=model.fit_transform(d)

tsne_data_1.shape
tsne_data_final_1=np.vstack((tsne_data_1.T,l)).T

tsne_df_1=pd.DataFrame(data=tsne_data_final_1,columns=("DIM1","DIM2","label"))

sns.FacetGrid(tsne_df_1,hue="label",size=6).map(plt.scatter,'DIM1','DIM2').add_legend()

plt.show()