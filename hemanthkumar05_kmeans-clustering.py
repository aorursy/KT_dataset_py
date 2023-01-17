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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
dataset=pd.read_csv('../input/bank-customers/Bank-Customers.csv')
dataset.head(n=10)
dataset.info()
dataset.describe()
dataset.isnull().sum()
features=dataset.iloc[:,[2,3]].values
#Feature scaling
from sklearn.preprocessing import StandardScaler
features=StandardScaler().fit_transform(features)
#Determing the optimal number of clusters using elbow method
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,10):
    kmeans=KMeans(n_clusters=i,random_state=0)
    kmeans.fit_transform(features)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,10),wcss)
plt.scatter(5, wcss[5], c = 'red',s = 100)
plt.text(5 + 0.5, wcss[5], s = '5 - Clusters', fontsize = 14)
plt.title('Elbow Method')
plt.xlabel('number of clusters')
plt.ylabel('wcss')
sns.set_style('darkgrid')
plt.show()
kmeans=KMeans(n_clusters=5,random_state=0)
y_kmeans=kmeans.fit_predict(features)

plt.scatter(features[y_kmeans==0,0],features[y_kmeans==0,1])
plt.scatter(features[y_kmeans==1,0],features[y_kmeans==1,1])
plt.scatter(features[y_kmeans==2,0],features[y_kmeans==2,1])
plt.scatter(features[y_kmeans==3,0],features[y_kmeans==3,1])
plt.scatter(features[y_kmeans==4,0],features[y_kmeans==4,1])
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],color='black')
plt.title('KMeans clustering')
plt.xlabel('Earning')
plt.ylabel('Credit Score')
sns.set_style('darkgrid')
plt.show()
