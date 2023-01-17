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
import pandas as pd



df = pd.read_csv(

    filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data',

    header=None,

    sep=',')



df.columns=['age', 'operation_year', 'positive_nodes', 'status']

df.dropna(how="all", inplace=True) # drops the empty line at file-end



df.tail()

# split data table into data X and class labels y



X = df.ix[:,0:3].values

y = df.ix[:,3].values

from matplotlib import pyplot as plt

import numpy as np

import math



label_dict = {1: 'Survive 5 or more',

              2: 'Survive less than five'}



feature_dict = {0: 'age',

                1: 'operation_year',

                2: 'positive_nodes'}



with plt.style.context('seaborn-whitegrid'):

    plt.figure(figsize=(8, 6))

    for cnt in range(3):

        plt.subplot(3, 1, cnt+1)

        for lab in (1, 2):

            plt.hist(X[y==lab, cnt],

                     label=lab,

                     bins=10,

                     alpha=0.3,)

        plt.xlabel(feature_dict[cnt])

    plt.legend(loc='upper right', fancybox=True, fontsize=8)



    plt.tight_layout()

    plt.show()

from sklearn.preprocessing import StandardScaler

X_std = StandardScaler().fit_transform(X)

from sklearn.decomposition import PCA as sklearnPCA

sklearn_pca = sklearnPCA(n_components=2)

Y_sklearn = sklearn_pca.fit_transform(X_std)

with plt.style.context('seaborn-whitegrid'):

    plt.figure(figsize=(6, 4))

    for lab, col in zip((1,2),

                        ('blue', 'black')):

        plt.scatter(Y_sklearn[y==lab, 0],

                    Y_sklearn[y==lab, 1],

                    label=lab,

                    c=col)

    plt.xlabel('Principal Component 1')

    plt.ylabel('Principal Component 2')

    plt.legend(loc='upper left')

    plt.tight_layout()

    plt.show()
