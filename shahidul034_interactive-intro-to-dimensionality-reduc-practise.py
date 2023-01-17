# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import seaborn as sns

import matplotlib.image as mpimg

import matplotlib.pyplot as plt

import matplotlib

%matplotlib inline



# Import the 3 dimensionality reduction methods

from sklearn.manifold import TSNE

from sklearn.decomposition import PCA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
train = pd.read_csv('../input/train.csv')

train.head()
print(train.shape)
from sklearn.preprocessing import StandardScaler

X = train.values

X_std = StandardScaler().fit_transform(X)

mean_vec = np.mean(X_std, axis=0)

cov_mat = np.cov(X_std.T)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)

# Create a list of (eigenvalue, eigenvector) tuples

eig_pairs = [ (np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]



# Sort the eigenvalue, eigenvector pair from high to low

eig_pairs.sort(key = lambda x: x[0], reverse= True)



# Calculation of Explained Variance from the eigenvalues

tot = sum(eig_vals)

var_exp = [(i/tot)*100 for i in sorted(eig_vals, reverse=True)] # Individual explained variance

cum_var_exp = np.cumsum(var_exp) # Cumulative explained variance
eig_pairs