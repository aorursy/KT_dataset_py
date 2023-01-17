# Supress unnecessary warnings so that presentation looks clean

import warnings

warnings.filterwarnings('ignore')



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



train.head()
X = train.drop('loss', axis=1)

y = train['loss']
cats = [name for name in X.columns if name.startswith('cat')]

conts = [name for name in X.columns if name.startswith('cont')]
sns.pairplot(X[conts[:3]],size=2.5)
from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_samples



#conts = np.sort(np.random.choice(conts, 10, replace=False))

km = KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=100, tol=1e-04)



predicts = km.fit_predict(X[conts])

cluster_labels = np.unique(predicts)

n_clusters = len(cluster_labels)
distortions = []

for i in range(1, 10):

    km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, tol=1e-04)

    km.fit(X[conts])

    distortions.append(km.inertia_)

    

plt.plot(distortions)
corr_cats = train[cats].corr()