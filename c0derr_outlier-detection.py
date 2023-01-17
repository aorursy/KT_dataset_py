# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

from sklearn.cluster import KMeans

from numpy import sqrt, random, array, argsort

from sklearn.preprocessing import scale



data=pd.read_csv("/kaggle/input/abalone-dataset/abalone.csv")

data

data.columns
data.info()
df=data.drop(columns=['Sex'],axis=1)

df
from collections import Counter

def detection(df,features):

    outlier_indices=[]

    

    for c in features:

        #1st quartile

        Q1 = np.percentile(df[c],25)

        

        #3rd quartile

        Q3 = np.percentile(df[c],75)

        

        #IQR calculation

        IQR = Q3 - Q1

        outlier_step = IQR * 1.5

        lower_range = Q1 - (outlier_step)

        upper_range = Q3 + (outlier_step)

        

        #Outlier detection                                    #Outlier indexes

        outlier_list_col=df[  (df[c] < lower_range) | (df[c] > upper_range)  ].index

       

        #Store indexes

        outlier_indices.extend(outlier_list_col)

        

    outlier_indices=Counter(outlier_indices)

    # number of outliers

    # If we have more then 2 outliers in a sample, this sample ll be drop

    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2 )

    #we are taking indexes

    

    return multiple_outliers
df.columns=['length', 'diameter', 'height', 'weight.w', 'weight.s',

       'weight.v', 'weight.sh', 'rings']

df.info()
outliers=detection(df,["length","weight.w","height","diameter"])

                       

df.loc[outliers]
df=df.drop(outliers,axis=0).reset_index(drop = True)

df
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

from sklearn.cluster import KMeans

from numpy import sqrt, random, array, argsort

from sklearn.preprocessing import scale

df.info()

sns.pairplot(df)

from sklearn import preprocessing



scaled_preprocessing=preprocessing.scale(df)

scaled_preprocessing
from scipy.stats import zscore

scaled = df.apply(zscore)

scaled.head()
from sklearn.cluster import KMeans

wcss=[] #liste oluştur

cluster_range=range(1,10)

for k  in cluster_range:

    kmeans=KMeans(n_clusters=k)

    kmeans.fit(scaled)

    wcss.append(kmeans.inertia_)

    

plt.plot(cluster_range,wcss,marker='x')

plt.xlabel("number of k(cluster)")

plt.ylabel("WCSS")

plt.title("Elbow Method for optimal number of clusters")

plt.show()
clusters_df = pd.DataFrame({'clusters':cluster_range,

                            'inertia': wcss})

clusters_df

kmeans=KMeans(n_clusters = 3)

kmeans.fit(scaled)
centroids = kmeans.cluster_centers_

centroids

centroid_df = pd.DataFrame(centroids,columns = list(scaled.columns))

centroid_df
clusters=scaled.copy()

clusters['cluster_pred']=kmeans.fit_predict(scaled)

scaled["labels"]=clusters['cluster_pred']

scaled
sns.pairplot(scaled,hue = 'labels')

#



from mpl_toolkits.mplot3d import Axes3D



fig = plt.figure(figsize=(10,8))

ax = Axes3D(fig,rect = [0,0,1,1],elev = 10,azim = 120)

labels = kmeans.labels_

ax.scatter(scaled.iloc[:,0],scaled.iloc[:,2],scaled.iloc[:,3],c = labels.astype(np.float),edgecolor = 'k')

ax.w_xaxis.set_ticklabels([])

ax.w_yaxis.set_ticklabels([])

ax.w_zaxis.set_ticklabels([])

ax.set_xlabel('Length')

ax.set_ylabel('Weight sh')

ax.set_zlabel('Height')

ax.set_title('3D plot for KMeans Clustering')



from sklearn.cluster import DBSCAN

from sklearn.preprocessing import StandardScaler

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
data=pd.read_csv("/kaggle/input/abalone-dataset/abalone.csv")

df=data.drop(columns=['Sex'],axis=1)

df.columns=['length', 'diameter', 'height', 'weight.w', 'weight.s',

       'weight.v', 'weight.sh', 'rings']

df.info()
lenWeightSh=df[["length","weight.sh"]]

lenWeightSh
plt.scatter(df['length'],df['weight.sh'],marker="o")

plt.xlabel("length",fontsize=12)

plt.ylabel("Shell Weight",fontsize=12)

plt.title('Length & Shell Weight', fontsize=16)

plt.show()
#  lenWeightSh

from sklearn.cluster import DBSCAN

outlier_detection = DBSCAN(

  eps = 0.05,

  metric="euclidean",

  min_samples = 15,

  n_jobs = -1)

clusters = outlier_detection.fit_predict(lenWeightSh)



clusters
from matplotlib import cm

cmap = cm.get_cmap('Accent')

lenWeightSh.plot.scatter(

  x = "length",

  y = "weight.sh",

  c = clusters,

  cmap = cmap,

  colorbar = False,

  marker="o"

)
