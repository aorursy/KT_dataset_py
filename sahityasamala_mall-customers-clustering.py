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
import pandas as pd

import numpy as np

import matplotlib.pylab as plt 

from sklearn.cluster import KMeans

from scipy.spatial.distance import cdist
dataset = pd.read_csv("/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv")
dataset.head()
dataset.info()
dataset.describe()
def norm_func(i):

    x = (i-i.min())	/	(i.max()	-	i.min())

    return (x)
X = dataset.drop(["Gender"], axis =1)
df_norm = norm_func(X)

df_norm.head()
k = list(range(2,23))

k

TWSS = [] # variable for storing total within sum of squares for each kmeans 

for i in k:

    kmeans = KMeans(n_clusters = i)

    kmeans.fit(df_norm)

    WSS = [] # variable for storing within sum of squares for each cluster 

    for j in range(i):

        WSS.append(sum(cdist(df_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,df_norm.shape[1]),"euclidean")))

    TWSS.append(sum(WSS))
#scree plot

plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)
model=KMeans(n_clusters=5) 

model.fit(df_norm)
model.labels_ # getting the labels of clusters assigned to each row 

md=pd.Series(model.labels_)  # converting numpy array into pandas series object 

dataset['clust']=md # creating a  new column and assigning it to new column 

dataset.head()



dataset.iloc[:,2:].groupby(dataset.clust).mean()
