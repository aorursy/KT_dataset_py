import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from scipy.linalg import eigh

import seaborn as sns

from sklearn.manifold import TSNE
data = pd.read_csv('../input/train.csv')

data.head()
label = data['label']

label.head()
data = data.drop('label',axis= 1)

data.head()
data.shape
label.shape
std_data = StandardScaler().fit_transform(data)

std_data.shape
std_data.T.shape
std_data.shape
model = TSNE(n_components=2,random_state=0)
tsne_data = model.fit_transform(std_data)

tsne_data = np.vstack((tsne_data.T,label)).T

tsne_df = pd.DataFrame(data=tsne_data,columns=("f1","f2","label"))
sns.FacetGrid(tsne_df,hue="label",size=8).map(plt.scatter,'f1','f2').add_legend()

plt.show()
model = TSNE(n_components=2,random_state=0,perplexity=50,n_iter=1000)
tsne_data = model.fit_transform(std_data)

tsne_data = np.vstack((tsne_data.T,label)).T

tsne_df = pd.DataFrame(data=tsne_data,columns=("f1","f2","label"))
sns.FacetGrid(tsne_df,hue="label",size=8).map(plt.scatter,'f1','f2').add_legend()

plt.show()
model = TSNE(n_components=2,random_state=0,perplexity=100,n_iter=2000)
tsne_data = model.fit_transform(std_data)

tsne_data = np.vstack((tsne_data.T,label)).T

tsne_df = pd.DataFrame(data=tsne_data,columns=("f1","f2","label"))
sns.FacetGrid(tsne_df,hue="label",size=8).map(plt.scatter,'f1','f2').add_legend()

plt.show()