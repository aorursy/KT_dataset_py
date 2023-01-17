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
# import dataset:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import dataset:
df=pd.read_csv("../input/Mall_Customers.csv")
df.head()
# make the dataset in form of dependent and independent:
X=df.iloc[:,[3,4]].values
X
# using the dendogram to find the optimal no on cluster:
import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(X,method='ward'))
plt.title('Dendrogram')
plt.xlabel('Cluster')
plt.ylabel('Ecluidistance')
plt.show()
# fitting the Hirarchical cluster on mall datset:
from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y_ch=hc.fit_predict(X)
y_ch
# visulization of cluster:
plt.scatter(X[y_ch==0,0],X[y_ch==0,1],s=100,color='red',label='cluster 1')
plt.scatter(X[y_ch==1,0],X[y_ch==1,1],s=100,color='blue',label='cluster 2')
plt.scatter(X[y_ch==2,0],X[y_ch==2,1],s=100,color='green',label='cluster 3')
plt.scatter(X[y_ch==3,0],X[y_ch==3,1],s=100,color='cyan',label='cluster 4')
plt.scatter(X[y_ch==4,0],X[y_ch==4,1],s=100,color='magenta',label='cluster 5')
plt.title('Cluster of clint')
plt.xlabel('Annual Income')
plt.ylabel('Salary')
plt.show()

