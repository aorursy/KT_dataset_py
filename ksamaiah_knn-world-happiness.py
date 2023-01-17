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
import pandas as pd

import numpy as np

import matplotlib as mpl



import seaborn as sns



from sklearn.preprocessing import scale

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

import pandas as pd

df_hap=pd.read_csv("../input/2017.csv") 

# Dataset that includes happiness ranking

x_h = df_hap[['Happiness.Rank','Economy..GDP.per.Capita.', 'Family', 'Health..Life.Expectancy.', 'Freedom','Generosity','Trust..Government.Corruption.']].values

# Dataset that does not include happiness ranking

x = df_hap[['Economy..GDP.per.Capita.', 'Family', 'Health..Life.Expectancy.', 'Freedom','Generosity','Trust..Government.Corruption.']].values
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

wcss = []



# exploring the data by dividing upto 20 clusters



for i in range(1, 20):

    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

    kmeans.fit(x)

    wcss.append(kmeans.inertia_)

    

#Plotting the results onto a line graph, allowing us to observe 'The elbow'

plt.plot(range(1, 20), wcss)

plt.title('The elbow method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS') #within cluster sum of squares

plt.show()



# we see the WCSS takes a deep fall before 5 
kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

y_kmeans = kmeans.fit_predict(x)

#Visualising the clusters

plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'red', label = 'CLuster 5')

plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'CLuster 1')

plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'green', label = 'CLuster 2')

plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s = 100, c = 'black', label = 'CLuster 3')

plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s = 100, c = 'aqua', label = 'CLuster 4')



#Plotting the centroids of the clusters

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'yellow', label = 'Centroids')



plt.legend()
# Making a dataframe with cluster numbers, country and happiness rank

h=set(y_kmeans)

df_f=pd.DataFrame(columns=('cluster','country','rank'))

for i in h:

    for j in x_h[y_kmeans == i, 0]: 

        df_f=df_f.append({'cluster':i.item(),'country':df_hap['Country'][df_hap['Happiness.Rank'] == int(j)].item(),'rank':int(j.item())},ignore_index=True)

df_f.head()   
import geopandas

world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

world.iloc[4,2]='United States' # some of the countries have different names in these data sets. showing example for USA.

df_f=df_f.sort_values(by='country')

df_cl=world.set_index('name').join(df_f.set_index('country'))

df_cl.plot(column ='cluster',figsize=(12, 10))