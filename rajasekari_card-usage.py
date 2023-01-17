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





import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
ccu_data=pd.read_csv("../input/CreditCardUsage.csv")
ccu_data.info()
ccu_data.head()
X = ccu_data.iloc[:,[4,14]].values
ccu_data.loc[:,['PURCHASES','CREDIT_LIMIT']]
ccu_data.corr()
ccu_data.loc[:,['PURCHASES','CREDIT_LIMIT']].isna().sum()
ccu_data.fillna(0,inplace=True)
sns.scatterplot(x=ccu_data['PURCHASES'],y=ccu_data['CREDIT_LIMIT'])
den_data=ccu_data.loc[:,['PURCHASES','CREDIT_LIMIT']].values
import scipy.cluster.hierarchy as sch

plt.figure(figsize=(15,6))

plt.title('Dendrogram')

plt.xlabel('PURCHASES')

plt.ylabel('CREDIT LIMIT')

plt.hlines(y=98000,xmin=0,xmax=1000000,lw=3,linestyles='--')

dendrogram = sch.dendrogram(sch.linkage(den_data, method = 'ward'))

plt.show()
from sklearn.cluster import KMeans 

#from sklearn.datasets.samples_generator import make_blobs 

#import pylab as pl

clusterNum = 6

k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)

k_means.fit(den_data)

labels = k_means.labels_

print(labels)
ccu_data['k_mean_col']=labels
ccu_data.head(5)
ccu_data.groupby('k_mean_col').mean()
from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters = 6, affinity = 'euclidean', linkage = 'ward')

y_hc = hc.fit_predict(den_data)
y_hc
plt.figure(figsize=(12,7))

plt.scatter(den_data[y_hc == 0, 0], den_data[y_hc == 0, 1], s = 100, c = 'red', label = 'Low Credit Low Purchases')

plt.scatter(den_data[y_hc == 1, 0], den_data[y_hc == 1, 1], s = 100, c = 'blue', label = 'Moderate Credit Moderate Purchases')

plt.scatter(den_data[y_hc == 2, 0], den_data[y_hc == 2, 1], s = 100, c = 'green', label = 'Low to Moderate Credit Low Purchases')

plt.scatter(den_data[y_hc == 3, 0], den_data[y_hc == 3, 1], s = 100, c = 'orange', label = 'High Credit High Purchases')

plt.scatter(den_data[y_hc == 4, 0], den_data[y_hc == 4, 1], s = 100, c = 'magenta', label = 'High Credit Low Purchases')

plt.scatter(den_data[y_hc == 5, 0], den_data[y_hc == 5, 1], s = 100, c = 'brown', label = 'Moderate to high Credit Low Purchases')

plt.title('Clustering of Creditcard Holders',fontsize=20)

plt.xlabel('Purchases',fontsize=16)

plt.ylabel('Credit Limit',fontsize=16)

plt.legend(fontsize=16)

plt.grid(True)

plt.axhspan(ymin=60,ymax=100,xmin=0.4,xmax=0.96,alpha=0.3,color='yellow')

plt.show()