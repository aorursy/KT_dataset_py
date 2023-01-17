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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
X_unlabeled = pd.read_csv('../input/dataset_X.csv', index_col='id')
X_unlabeled.head()
X_unlabeled.describe()
sns.pairplot(X_unlabeled.sample(10000), diag_kind='hist')
plt.show()
def plot_2d(df): 
    data = (
        pd.DataFrame(
            {'x': df[:, 0], 
             'y': df[:, 1]})
    )

    sns.lmplot(
        data=data, 
        x='x', 
        y='y', 
        fit_reg=True)

    plt.show()
X_embedded = TSNE(n_components=2).fit_transform(X_unlabeled.sample(5000).values)
plot_2d(X_embedded)
sns.jointplot(
    data=X_unlabeled.sample(100000), 
    x='d4', 
    y='d5', 
    kind='kde')
plt.show()
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X=X_unlabeled)
plot_2d(X_pca[0:10000, :])