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
import matplotlib.pyplot as plt



import seaborn as sns; sns.set() #for plot styling

%matplotlib inline



plt.rcParams['figure.figsize'] = (16,9)

plt.style.use('ggplot')



dataset = pd.read_csv('../input/clustering-customer-data/CLV.csv')

dataset.head()

##len(dataset)

dataset.describe().transpose()

plot_income = sns.distplot(dataset["INCOME"])

plot_spend = sns.distplot(dataset["SPEND"])

plt.xlabel('Income/Spend')
scatter_income = sns.scatterplot(x = "INCOME", y ="SPEND", data=dataset)
from sklearn.cluster import KMeans

wcss = []

for i in range (1,11): # this refers to 1 - 11 possible clusters

    km = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

    km.fit(dataset)

    wcss.append(km.inertia_)

plt.plot(range(1,11), wcss)

plt.xlabel('Number of Clusters')

plt.ylabel('wcss')

plt.show()
km6 = KMeans(n_clusters = 6, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

X = dataset

y_means = km6.fit_predict(X)

X.head()



plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_means, s=50, cmap='viridis')

plt.scatter(km6.cluster_centers_[:,0], km6.cluster_centers_[:,1],s=200,marker='s', c='red', alpha=0.7, label='Customer Group')



plt.title('Customer segments')

plt.xlabel('Annual income of customer')

plt.ylabel('Annual spend from customer on site')

plt.legend()

plt.show()


