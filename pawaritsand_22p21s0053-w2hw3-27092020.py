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
df = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

df
df.info()
df.isnull().sum()
df1 = df[['neighbourhood','price','number_of_reviews','minimum_nights']] 

df1 = df1.groupby('neighbourhood').mean()

df_grouped = df1[['price','number_of_reviews','minimum_nights']]
df_grouped
import numpy as np



from matplotlib import pyplot as plt

from scipy.cluster.hierarchy import dendrogram

from sklearn.cluster import AgglomerativeClustering

from sklearn.preprocessing import MinMaxScaler





def plot_dendrogram(model, **kwargs):

    # Create linkage matrix and then plot the dendrogram



    # create the counts of samples under each node

    counts = np.zeros(model.children_.shape[0])

    n_samples = len(model.labels_)

    for i, merge in enumerate(model.children_):

        current_count = 0

        for child_idx in merge:

            if child_idx < n_samples:

                current_count += 1  # leaf node

            else:

                current_count += counts[child_idx - n_samples]

        counts[i] = current_count



    linkage_matrix = np.column_stack([model.children_, model.distances_,

                                      counts]).astype(float)



    # Plot the corresponding dendrogram

    dendrogram(linkage_matrix, **kwargs)



scaler = MinMaxScaler()

iris = scaler.fit_transform(df_grouped)

X = iris.data



# setting distance_threshold=0 ensures we compute the full tree.

model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)



model = model.fit(X)

fig = plt.figure(figsize=(18,12))

plt.title('Hierarchical Clustering Dendrogram')

# plot the top three levels of the dendrogram

plot_dendrogram(model, truncate_mode='level', p=5,orientation='right')

plt.show()