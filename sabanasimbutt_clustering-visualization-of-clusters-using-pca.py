# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



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



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data= pd.read_csv("../input/CC GENERAL.csv")

print(data.shape)

data.head()
data.describe()
data.isnull().sum().sort_values(ascending=False).head()
data.loc[(data['MINIMUM_PAYMENTS'].isnull()==True),'MINIMUM_PAYMENTS']=data['MINIMUM_PAYMENTS'].mean()

data.loc[(data['CREDIT_LIMIT'].isnull()==True),'CREDIT_LIMIT']=data['CREDIT_LIMIT'].mean()
data.isnull().sum().sort_values(ascending=False).head()
columns=['BALANCE', 'PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT',

        'PAYMENTS', 'MINIMUM_PAYMENTS']



for c in columns:

    

    Range=c+'_RANGE'

    data[Range]=0        

    data.loc[((data[c]>0)&(data[c]<=500)),Range]=1

    data.loc[((data[c]>500)&(data[c]<=1000)),Range]=2

    data.loc[((data[c]>1000)&(data[c]<=3000)),Range]=3

    data.loc[((data[c]>3000)&(data[c]<=5000)),Range]=4

    data.loc[((data[c]>5000)&(data[c]<=10000)),Range]=5

    data.loc[((data[c]>10000)),Range]=6

 
columns=['BALANCE_FREQUENCY', 'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY', 

         'CASH_ADVANCE_FREQUENCY', 'PRC_FULL_PAYMENT']



for c in columns:

    

    Range=c+'_RANGE'

    data[Range]=0

    data.loc[((data[c]>0)&(data[c]<=0.1)),Range]=1

    data.loc[((data[c]>0.1)&(data[c]<=0.2)),Range]=2

    data.loc[((data[c]>0.2)&(data[c]<=0.3)),Range]=3

    data.loc[((data[c]>0.3)&(data[c]<=0.4)),Range]=4

    data.loc[((data[c]>0.4)&(data[c]<=0.5)),Range]=5

    data.loc[((data[c]>0.5)&(data[c]<=0.6)),Range]=6

    data.loc[((data[c]>0.6)&(data[c]<=0.7)),Range]=7

    data.loc[((data[c]>0.7)&(data[c]<=0.8)),Range]=8

    data.loc[((data[c]>0.8)&(data[c]<=0.9)),Range]=9

    data.loc[((data[c]>0.9)&(data[c]<=1.0)),Range]=10

    
columns=['PURCHASES_TRX', 'CASH_ADVANCE_TRX']  



for c in columns:

    

    Range=c+'_RANGE'

    data[Range]=0

    data.loc[((data[c]>0)&(data[c]<=5)),Range]=1

    data.loc[((data[c]>5)&(data[c]<=10)),Range]=2

    data.loc[((data[c]>10)&(data[c]<=15)),Range]=3

    data.loc[((data[c]>15)&(data[c]<=20)),Range]=4

    data.loc[((data[c]>20)&(data[c]<=30)),Range]=5

    data.loc[((data[c]>30)&(data[c]<=50)),Range]=6

    data.loc[((data[c]>50)&(data[c]<=100)),Range]=7

    data.loc[((data[c]>100)),Range]=8
data.drop(['CUST_ID', 'BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES',

       'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE',

       'PURCHASES_FREQUENCY',  'ONEOFF_PURCHASES_FREQUENCY',

       'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY',

       'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'CREDIT_LIMIT', 'PAYMENTS',

       'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT' ], axis=1, inplace=True)



X= np.asarray(data)
scale = StandardScaler()

X = scale.fit_transform(X)

X.shape
n_clusters=30

cost=[]

for i in range(1,n_clusters):

    kmean= KMeans(i)

    kmean.fit(X)

    cost.append(kmean.inertia_)  
plt.plot(cost, 'bx-')
kmean= KMeans(6)

kmean.fit(X)

labels=kmean.labels_
clusters=pd.concat([data, pd.DataFrame({'cluster':labels})], axis=1)

clusters.head()
for c in clusters:

    grid= sns.FacetGrid(clusters, col='cluster')

    grid.map(plt.hist, c)
dist = 1 - cosine_similarity(X)



pca = PCA(2)

pca.fit(dist)

X_PCA = pca.transform(dist)

X_PCA.shape
x, y = X_PCA[:, 0], X_PCA[:, 1]



colors = {0: 'red',

          1: 'blue',

          2: 'green', 

          3: 'yellow', 

          4: 'orange',  

          5:'purple'}



names = {0: 'who make all type of purchases', 

         1: 'more people with due payments', 

         2: 'who purchases mostly in installments', 

         3: 'who take more cash in advance', 

         4: 'who make expensive purchases',

         5:'who don\'t spend much money'}

  

df = pd.DataFrame({'x': x, 'y':y, 'label':labels}) 

groups = df.groupby('label')



fig, ax = plt.subplots(figsize=(20, 13)) 



for name, group in groups:

    ax.plot(group.x, group.y, marker='o', linestyle='', ms=5,

            color=colors[name],label=names[name], mec='none')

    ax.set_aspect('auto')

    ax.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')

    ax.tick_params(axis= 'y',which='both',left='off',top='off',labelleft='off')

    

ax.legend()

ax.set_title("Customers Segmentation based on their Credit Card usage bhaviour.")

plt.show()