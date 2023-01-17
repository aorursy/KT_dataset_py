# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.manifold import TSNE 

from sklearn.decomposition import PCA

import umap

%matplotlib inline

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/heart.csv')

df.head(5)
#Checking missing values

df.isnull().sum()
feat=df.drop(['target'],axis=1)
target=df['target']
X=df.drop(['target'],axis=1)

X.corrwith(df['target']).plot.bar(

        figsize = (20, 10), title = "Correlation with Target", fontsize = 20,

        rot = 90, grid = True)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

pca_result = pca.fit_transform(feat.values)
plt.plot(range(2), pca.explained_variance_ratio_)

plt.plot(range(2), np.cumsum(pca.explained_variance_ratio_))

plt.title("Component-wise and Cumulative Explained Variance")
def pca_results(good_data, pca):

	'''

	Create a DataFrame of the PCA results

	Includes dimension feature weights and explained variance

	Visualizes the PCA results

	'''



	# Dimension indexing

	dimensions = dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]



	# PCA components

	components = pd.DataFrame(np.round(pca.components_, 4), columns = list(good_data.keys()))

	components.index = dimensions



	# PCA explained variance

	ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)

	variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])

	variance_ratios.index = dimensions



	# Create a bar plot visualization

	fig, ax = plt.subplots(figsize = (14,8))



	# Plot the feature weights as a function of the components

	components.plot(ax = ax, kind = 'bar');

	ax.set_ylabel("Feature Weights")

	ax.set_xticklabels(dimensions, rotation=0)





	# Display the explained variance ratios

	for i, ev in enumerate(pca.explained_variance_ratio_):

		ax.text(i-0.40, ax.get_ylim()[1] + 0.05, "Explained Variance\n          %.4f"%(ev))



	# Return a concatenated DataFrame

	return pd.concat([variance_ratios, components], axis = 1)



pca_results = pca_results(feat, pca)