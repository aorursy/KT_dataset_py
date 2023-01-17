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
import matplotlib.pyplot as plt  # for visualization 

%matplotlib inline

import seaborn as sns           # for visualization 
# read the data

df = pd.read_csv('/kaggle/input/ccdata/CC GENERAL.csv')

df.head()
df.head()
df.info()
df.describe().T
df.nunique()
def missing_percentage(df):



    total = df.isnull().sum().sort_values(

        ascending=False)[df.isnull().sum().sort_values(ascending=False) != 0]

    percent = (df.isnull().sum().sort_values(ascending=False) / len(df) *

               100)[(df.isnull().sum().sort_values(ascending=False) / len(df) *

                     100) != 0]

    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])





missing_data = missing_percentage(df)



fig, ax = plt.subplots( figsize=(16, 6))



sns.barplot(x=missing_data.index,

            y='Percent',

            data=missing_data)





ax.set_title('Missing Values')

plt.show()
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)

sns.distplot(df.MINIMUM_PAYMENTS, color='#fdc029')

plt.subplot(1,2,2)

sns.distplot(df.CREDIT_LIMIT, color='#fdc029')

plt.show()
df.MINIMUM_PAYMENTS.fillna(df.MINIMUM_PAYMENTS.median(),inplace=True)
print('MINIMUM_PAYMENTS FEATURE HAS',df.MINIMUM_PAYMENTS.isna().sum(),'MISSING VALUE')
df.CREDIT_LIMIT.fillna(df.CREDIT_LIMIT.median(),inplace=True)
print('CREDIT_LIMIT FEATURE HAS',df.CREDIT_LIMIT.isna().sum(),'MISSING VALUE')
g = sns.PairGrid(df)

g.map(plt.scatter)

plt.title('relations between features')

plt.show()
def scatter_purchases(x):

    sns.scatterplot(y='PURCHASES',x=x,data = df,color='#171820',alpha=0.7)
scatter_purchases('BALANCE')
plt.figure(figsize=(16,5))



plt.subplot(1,2,1)

sns.lineplot(x='TENURE',y='PURCHASES',data=df)

plt.title('The Purchases based on Tenure of credit card service for use')

plt.subplot(1,2,2)

scatter_purchases('TENURE')

plt.hist(df.CREDIT_LIMIT)

plt.title('credit limit distribution')

plt.show()
col = list(df.drop('CUST_ID',axis=1).columns)
plt.figure(figsize=(30,30))

for idx,val in enumerate(col):

    plt.subplot(6,3,idx+1)

    sns.boxplot(x=val,data=df)
from sklearn.preprocessing import StandardScaler

scale = StandardScaler()

X = scale.fit_transform(df.drop('CUST_ID',axis=1))
from sklearn.cluster import KMeans

n_clusters=30

cost=[]

for i in range(1,n_clusters):

    kmean= KMeans(i)

    kmean.fit(X)

    cost.append(kmean.inertia_)  
plt.plot(cost, 'bx-')
n_clusters=10

cost=[]

for i in range(1,n_clusters):

    kmean= KMeans(i)

    kmean.fit(X)

    cost.append(kmean.inertia_)  
plt.plot(cost, 'gx-')

plt.title('Elbow Criterion')

plt.show()
kmean= KMeans(6)

kmean.fit(X)

labels=kmean.labels_
clusters=pd.concat([df, pd.DataFrame({'cluster':labels})], axis=1)

clusters.head()
clusters.info()
for c in clusters.iloc[:,1:]:

    grid= sns.FacetGrid(clusters.iloc[:,1:], col='cluster')

    grid.map(plt.hist, c)
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.decomposition import PCA

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
