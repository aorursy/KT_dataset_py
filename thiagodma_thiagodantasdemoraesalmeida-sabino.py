# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv', index_col=0)
df.head()
df.SalePrice.hist()
df.SalePrice.apply(np.log).hist()
df = pd.read_json('../input/iris-dataset-json-version/iris.json')
df.head()
y = df.species

X = df.drop(columns=['species'])
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

import seaborn as sns



sns.set()



pca = PCA(n_components=2)

Xt = pca.fit_transform(X)

Xt = pd.DataFrame(Xt,index=X.index,columns=['f','s'])

Xt['c'] = y



for cls,color in zip(y.unique(),['r','g','b']):

    plt.scatter(Xt.loc[Xt.c==cls]['f'],Xt.loc[Xt.c==cls]['s'],color=color, label = cls)



plt.legend()