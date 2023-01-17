import pandas as pd

import numpy as np

from scipy.spatial.distance import cosine

pd.__version__
# custom functions for DataFrame.corr() are only available in pandas>=0.24.0, define a method based on the pandas implementation in 0.24.0

from pandas.core.dtypes.common import _ensure_float64 as ensure_float64



from pandas.core import nanops

from pandas import DataFrame

import numpy as np



def corr(df1, method, min_periods=1):

    numeric_df = df1._get_numeric_data()

    cols = numeric_df.columns

    idx = cols.copy()

    mat = numeric_df.values



    if callable(method):

        if min_periods is None:

            min_periods = 1

        mat = ensure_float64(mat).T

        K = len(cols)

        correl = np.empty((K, K), dtype=float)

        mask = np.isfinite(mat)

        for i, ac in enumerate(mat):

            for j, bc in enumerate(mat):

                if i > j:

                    continue



                valid = mask[i] & mask[j]

                if valid.sum() < min_periods:

                    c = np.nan

                elif i == j:

                    c = 1.

                elif not valid.all():

                    c = corrf(ac[valid], bc[valid])

                else:

                    c = method(ac, bc)

                correl[i, j] = c

                correl[j, i] = c

    return DataFrame(correl, index=idx, columns=cols)
df = pd.read_hdf('../input/painting-to-vector-encoding/df_with_vectors.h5', key='vectors')
df.head()
sample = df.sample(30)

sample.index = sample['file']

values = sample['vector'].apply(lambda x: pd.Series(x)).T
from scipy.spatial.distance import cosine

import seaborn as sns

import matplotlib.image as mpimg

import matplotlib.pyplot as plt



similarity = lambda x, y: cosine(x, y)+1

similarity_matrix = corr(values, similarity)

plt.figure(figsize=(14,14))

sns.heatmap(similarity_matrix, cmap='BuGn_r')
import os



def plot_top_n_similar_imgs(series, n=3):

    similar_imgs = series.sort_values(ascending=True)[:n+1]

    fig, ax = plt.subplots(1, n+1)

    fig.set_size_inches(h=12, w=14)

    for idx, (file, score) in enumerate(similar_imgs.items()):

        ax[idx].imshow(mpimg.imread(file.replace('../input/', '../input/best-artworks-of-all-time/')))

        ax[idx].set_title("{}:\n{:.2f}".format(file.split('/')[-1], score))





_ = similarity_matrix.apply(plot_top_n_similar_imgs, axis=1)
