import pandas as pd

import numpy as np

import scipy

import sklearn

from scipy.io import loadmat

from sklearn.decomposition import PCA

import matplotlib.pylab as plt
def mat_to_pandas(path):

    mat = loadmat(path)

    names = mat['dataStruct'].dtype.names

    ndata = {n: mat['dataStruct'][n][0, 0] for n in names}

    return pd.DataFrame(ndata['data'], columns=ndata['channelIndices'][0])
df = mat_to_pandas('../input/test_1/1_999.mat')

df[1].hist()
pca = PCA(n_components=5)

pca.fit(df)

_df = pca.transform(df)
plt.plot(_df[:,-1])
df[1].plot()