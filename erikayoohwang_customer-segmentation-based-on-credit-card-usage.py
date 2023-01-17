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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from sklearn.metrics.pairwise import cosine_similarity



import warnings

warnings.filterwarnings(action="ignore")

data=pd.read_csv('/kaggle/input/ccdata/CC GENERAL.csv')

data.head(50)
data.isnull().sum()
# fill null with the mean of each column

data.loc[(data['MINIMUM_PAYMENTS'].isnull()==True),'MINIMUM_PAYMENTS']=data['MINIMUM_PAYMENTS'].mean()

data.loc[(data['CREDIT_LIMIT'].isnull()==True),'CREDIT_LIMIT']=data['CREDIT_LIMIT'].mean()
data.isnull().sum()
data.describe()
# Dealing with outliers

# based on the mean of each column



columns=['BALANCE','PURCHASES','ONEOFF_PURCHASES','INSTALLMENTS_PURCHASES','CASH_ADVANCE','CREDIT_LIMIT','PAYMENTS','MINIMUM_PAYMENTS']



for c in columns:

    RANGE=c+'_RANGE'

    data[RANGE]=0

    data.loc[((data[c]>0) & (data[c]<500)), RANGE]=1

    data.loc[((data[c]>=500) & (data[c]<1000)), RANGE]=2

    data.loc[((data[c]>=1000) & (data[c]<3000)), RANGE]=3

    data.loc[((data[c]>=3000) & (data[c]<5000)), RANGE]=4

    data.loc[((data[c]>=5000) & (data[c]<10000)), RANGE]=5

    data.loc[(data[c]>=10000), RANGE]=6
columns=['BALANCE_FREQUENCY', 'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY','CASH_ADVANCE_FREQUENCY','PRC_FULL_PAYMENT']



for c in columns:

    RANGE= c + '_RANGE'

    data[RANGE]=0

    data.loc[((data[c]>0)&(data[c]<0.1)), RANGE]=1

    data.loc[((data[c]>=0.1)&(data[c]<0.2)), RANGE]=2

    data.loc[((data[c]>=0.2)&(data[c]<0.3)), RANGE]=3

    data.loc[((data[c]>=0.3)&(data[c]<0.4)), RANGE]=4

    data.loc[((data[c]>=0.4)&(data[c]<0.5)), RANGE]=5

    data.loc[((data[c]>=0.5)&(data[c]<0.6)), RANGE]=6

    data.loc[((data[c]>=0.6)&(data[c]<0.7)), RANGE]=7

    data.loc[((data[c]>=0.7)&(data[c]<0.8)), RANGE]=8

    data.loc[((data[c]>=0.8)&(data[c]<0.9)), RANGE]=9

    data.loc[((data[c]>=0.9)&(data[c]<1.0)), RANGE]=10

    data.loc[(data[c]>=1.0), RANGE]=11

    
columns=['CASH_ADVANCE_TRX','PURCHASES_TRX']



for c in columns:

    RANGE= c+'_RANGE'

    data[RANGE]=0

    data.loc[((data[c]>0)&(data[c]<=5)),RANGE]=1

    data.loc[((data[c]>5)&(data[c]<=10)),RANGE]=2

    data.loc[((data[c]>10)&(data[c]<=15)),RANGE]=3

    data.loc[((data[c]>15)&(data[c]<=20)),RANGE]=4

    data.loc[((data[c]>20)&(data[c]<=30)),RANGE]=5

    data.loc[((data[c]>30)&(data[c]<=50)),RANGE]=6

    data.loc[((data[c]>50)&(data[c]<=100)),RANGE]=7

    data.loc[((data[c]>100)),RANGE]=8
data.drop(['BALANCE','BALANCE_FREQUENCY','PURCHASES','ONEOFF_PURCHASES','INSTALLMENTS_PURCHASES','CASH_ADVANCE','PURCHASES_FREQUENCY','ONEOFF_PURCHASES_FREQUENCY','PURCHASES_INSTALLMENTS_FREQUENCY',

          'CASH_ADVANCE_FREQUENCY','CASH_ADVANCE_TRX','PURCHASES_TRX','CREDIT_LIMIT','PAYMENTS','MINIMUM_PAYMENTS','PRC_FULL_PAYMENT'], axis=1, inplace=True)
data.head()



x=np.array(data.drop(['CUST_ID'], axis=1))
# Normalizing input data

scale=StandardScaler()

x=scale.fit_transform(x)

x.shape
# Modeling using KMeans clustering

# Elbow Method



distortion=[]

iterations=3000

num_centroid_seeds=10

rand_state=0



for i in range(1, 30):

    kmeans=KMeans(n_clusters=i, max_iter=iterations, n_init=num_centroid_seeds, random_state=rand_state)

    kmeans.fit(x)

    distortion.append(kmeans.inertia_)



    

plt.plot(range(1,30), distortion, marker='o')

plt.title('Elbow Method')

plt.xlabel('Number of Clusters')

plt.ylabel('distortion')

plt.show()
# number of cluster=7

kmeans=KMeans(n_clusters=7, max_iter=iterations, n_init=num_centroid_seeds, random_state=rand_state)

kmeans_preds=kmeans.fit_predict(x)

kmeans_preds # predict included class 

labels=kmeans.labels_

labels
clusters=pd.concat([data, pd.DataFrame({'cluster':labels})], axis=1)

clusters.head()
# scores of clustering

from sklearn.metrics import silhouette_samples, silhouette_score

print('silhouette_score:',silhouette_score(x, labels=kmeans.labels_))

print('K-Means centers:', kmeans.cluster_centers_)

print('k-Menas labels:', kmeans.labels_)
# interpretation of Clusters

for c in clusters:

    grid=sns.FacetGrid(clusters, col='cluster')

    grid.map(plt.hist, c)
# Visualization of Clusters

# using PCA to transform data to 2 dimensions for visualization



dist=1-cosine_similarity(x) # apply cosine distance for x



pca=PCA(2) # transform 17 dimensions to 2 dimensions

pca.fit(dist)

x_pca=pca.transform(dist)

x_pca.shape
x_pca
x, y=x_pca[:, 0], x_pca[:,1]



colors={0:'red',

       1: 'blue',

       2:'green',

       3:'yellow',

       4:'orange',

       5:'purple',

       6:'magenta'}



names={0:'cluster0',

      1:'cluster1',

      2:'cluster2',

      3:'cluster3',

      4:'cluster4',

      5:'cluster5',

      6:'cluster6'}



df=pd.DataFrame({'x':x, 'y':y, 'label':labels})

groups=df.groupby('label')







fig, ax=plt.subplots(figsize=(20,13))





for name, group in groups:

    ax.plot(group.x, group.y, marker='o', linestyle='', ms=5,

           color=colors[name], label=names[name], mec='none')

    ax.set_aspect('auto')

    ax.tick_params(axis='x',which='both', bottom='off', top='off', labelbottom='off')

    ax.tick_params(axis='y',which='both', bottom='off', top='off', labelbottom='off')

    

ax.legend()

ax.set_title("Customer Segmentation based on their credit card using behavior")

plt.show()


