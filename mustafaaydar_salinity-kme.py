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
#Import the dataset



dataset = pd.read_csv('/kaggle/input/calcofi/bottle.csv')

data = dataset[['Salnty','T_degC','Depthm']]

data.head(5)

data.fillna(data.mean(), inplace=True)

data.info()

#dataset['Sta_ID'].astype('float')
# %% KMEANS

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

wcss = []



for k in range(1,15):

    kmeans = KMeans(n_clusters=k)

    kmeans.fit(data)

    wcss.append(kmeans.inertia_)

    

plt.plot(range(1,15),wcss)

plt.xlabel("number of k (cluster) value")

plt.ylabel("wcss")

plt.show()

data.head()

xn=list(data.columns)[1:]

datan=data[xn]

#y=data.iloc[:,:1]



#%% k = 4 icin modelim



kmeans2 = KMeans(n_clusters=4)

clusters = kmeans2.fit_predict(datan)

datan["label"] = clusters

#data.tail()



from sklearn.decomposition import PCA

import seaborn as sns

reduced_data = PCA(n_components=2).fit_transform(datan)

results = pd.DataFrame(reduced_data,columns=['pca1','pca2'])



sns.scatterplot(x="pca1", y="pca2", hue=datan['label'], data=results)

plt.title('K-means Clustering with 2 dimensions')

plt.show()