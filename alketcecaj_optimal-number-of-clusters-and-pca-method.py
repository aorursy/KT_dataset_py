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
df_wine_offers = pd.read_excel("../input/WineKMC.xlsx", sheetname=0)

df_wine_offers.columns = ["offer_id", "campaign", "varietal", "min_qty", "discount", "origin", "past_peak"]

df_wine_offers.head()
df_wine_transactions = pd.read_excel("../input/WineKMC.xlsx", sheetname=1)

df_wine_transactions.columns = ["customer_name", "offer_id"]

df_wine_transactions['n'] = 1

df_wine_transactions.head()
# merge the two dataframes

my_df = pd.merge(df_wine_offers, df_wine_transactions)

my_df.head()
# create a matrix 

matrix = my_df.pivot_table(index=['customer_name'], columns=['offer_id'], values='n',fill_value=0)

matrix.head(5)
# run a first KMeans clustering algorithm as required in the exercise

from sklearn.cluster import KMeans

cluster = KMeans(n_clusters=5)

matrix['cluster'] = cluster.fit_predict(matrix[matrix.columns[2:]])

matrix.cluster.value_counts()
# the sum of squared error 

ssd = []

K = range(2,11)

for cluster_i in K:

    kmeans = KMeans(n_clusters=cluster_i)

    kmodel = kmeans.fit(matrix[matrix.columns[2:]])

    ssd.append(kmodel.inertia_)
import sklearn

import matplotlib.pyplot as plt

import seaborn as sns

plt.plot(K, ssd, 'bx-')

plt.xlabel('nr_clusters')

plt.ylabel('ssd')

plt.title('Elbow method for chosing the best number of clusters')

plt.show()
# prepare the x_cols first.

x_cols = matrix.columns[:-1]

x_cols
#run PCA



from sklearn.decomposition import PCA



pca = PCA(n_components=2)

matrix['x'] = pca.fit_transform(matrix[x_cols])[:,0]

matrix['y'] = pca.fit_transform(matrix[x_cols])[:,1]

matrix = matrix.reset_index()
matrix.head(5)
customer_clusters = matrix[['customer_name', 'cluster', 'x', 'y']]

customer_clusters.head()

# Get current size of figure 

fig_size = plt.rcParams["figure.figsize"]

print(fig_size)



# And reset figure width to 12 and height to 9

fig_size[0] = 12

fig_size[1] = 9

plt.rcParams["figure.figsize"] = fig_size



plt.scatter(customer_clusters['x'], customer_clusters['y'], c = customer_clusters['cluster'])