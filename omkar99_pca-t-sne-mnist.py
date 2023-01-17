import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
data = pd.read_csv('../input/train.csv')

label = data['label']

pixels = data.drop('label',axis=1)
from sklearn import decomposition

pca = decomposition.PCA()

pca.n_components = 2
pca_data = pca.fit_transform(pixels)

label = np.reshape(label.values,(label.shape[0],1))
data_transformed = np.hstack((pca_data,label))

dframe = pd.DataFrame(data = data_transformed, columns = ('pc1','pc2','label'))
sns.FacetGrid(dframe,hue='label',size=5).map(plt.scatter,'pc1','pc2').add_legend()
from sklearn.manifold import TSNE



model = TSNE(n_components=2, random_state=0)

tsne_transform = model.fit_transform(pixels[:10000])
tsne_trans_data = np.hstack((tsne_transform,label[:10000]))

tsne_dframe = pd.DataFrame(data=tsne_trans_data, columns = ('c1','c2','label'))
sns.FacetGrid(tsne_dframe,hue='label',height=5).map(plt.scatter,'c1','c2').add_legend()