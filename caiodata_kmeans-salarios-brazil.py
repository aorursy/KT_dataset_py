# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/monthly_salary_brazil.csv",error_bad_lines=False)

df.head()
df.info()
df.total_salary.hist(bins=3)
df[df.total_salary < 0].head()
df[df.total_salary < 0].min()
df['total_salary'] = df['total_salary'].abs()
df.total_salary.hist(bins=50)
df[df.total_salary > 0].max()
from sklearn.cluster import KMeans 
df.total_salary.isnull().sum()
n_centroids = [x+2 for x in range(20)]

wcss = []

for c in n_centroids:

    kmeans = KMeans(n_clusters=c)

    kmeans.fit(df.total_salary.values.reshape(-1, 1))

    wcss.append(kmeans.inertia_)

    print(c)
plt.scatter(range(1,21), wcss)

plt.plot(range(1,21), wcss)

plt.xticks(range(1,21))

plt.grid()

np.log(pd.DataFrame(wcss)/pd.DataFrame(wcss).shift(1))
kmeans = KMeans(n_clusters=6)

previsao = kmeans.fit_predict(df.total_salary.values.reshape(-1, 1))

df['cluster'] = previsao
pd.crosstab(df.job, df.cluster)
pd.crosstab(df.sector, df.cluster)
agg_cluster = df.groupby('cluster')
agg_cluster.total_salary.mean()
agg_cluster.total_salary.count()