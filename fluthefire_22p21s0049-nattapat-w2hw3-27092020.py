# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import scipy.cluster.hierarchy as sch

from sklearn.cluster import AgglomerativeClustering

import matplotlib.pyplot as plt

from sklearn import preprocessing
dataset = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')[:30000]

dataset.head()
dataset.head()

cols = ['price', 'minimum_nights', 'number_of_reviews', 'calculated_host_listings_count', 'availability_365']
pt = preprocessing.PowerTransformer(method='yeo-johnson', standardize=True)

mat = pt.fit_transform(dataset[cols])

mat[:5].round(4)
X = pd.DataFrame(mat, columns=cols)

X.head()
X.info()
dataset[cols].hist(layout=(1, len(cols)), figsize=(3*len(cols), 3.5));
X[cols].hist(layout=(1, len(cols)), figsize=(3*len(cols), 3.5), color='orange', alpha=0.5);
fig, ax = plt.subplots(figsize=(20,7))

dg = sch.dendrogram(sch.linkage(X, method='ward'))
import seaborn as sns
sns.clustermap(X, col_cluster=False, cmap='Blues')