import math

import numpy as np

import pandas as pd

import psutil

from time import time



from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import log_loss

import matplotlib.pyplot as plt

from matplotlib import pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from mpl_toolkits.mplot3d import proj3d

from matplotlib import offsetbox

import seaborn as sns

from sklearn.linear_model import RandomizedLasso

from sklearn.decomposition import PCA, IncrementalPCA

from sklearn import (manifold, datasets, decomposition, ensemble,

                     discriminant_analysis, random_projection)



import sys
# indices of validation examples

X = pd.read_csv( '../input/numerai_training_data.csv' )

X_temp = pd.read_csv( '../input/numerai_training_data.csv' )

y = X['target']

X.drop( 'target', axis = 1, inplace=True)

n_samples, n_features = X.shape



t0 = time()



print("PCA")



corr = X.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 7))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3,

            square=True, xticklabels=5, yticklabels=5,

            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
pca = IncrementalPCA(batch_size=3, copy=True, n_components=10, whiten=False)

X_pca = pca.fit_transform(X,y)
X_pca

data = pd.DataFrame(data=X_pca[0:,0:])    # values

data["target"] = y



sns.heatmap(data.corr(), mask=None, 

            square=True

            )
sns.violinplot(x=X_temp.columns.values, y=None, hue="target",

                     data=X_temp, palette="Set2")
names = X.columns

 

rlasso = RandomizedLasso(alpha=0.005)

rlasso.fit(X, y)
rlasso.scores_

rlasso.all_scores_
pca = IncrementalPCA(batch_size=10, copy=True, n_components=10, whiten=False)

X_pca = pca.fit_transform(X,y)
data = pd.DataFrame(data=X_pca[0:,0:])

data2 = data.corr()

 # Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(data2)
data2