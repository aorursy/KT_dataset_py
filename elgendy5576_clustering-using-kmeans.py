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
import matplotlib.pyplot as plt 

import seaborn as sns 

import scipy

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from sklearn.metrics.pairwise import cosine_similarity
df = pd.read_csv("/kaggle/input/ccdata/CC GENERAL.csv")
df.head()
df.shape
for i,v in enumerate(df.columns):

    print(i,v)
df.shape
df.isnull().sum()
df["MINIMUM_PAYMENTS"].plot(kind='hist')
sns.boxplot(df["MINIMUM_PAYMENTS"])
print(df["MINIMUM_PAYMENTS"].median())

print(df["MINIMUM_PAYMENTS"].mean())
df["MINIMUM_PAYMENTS"].fillna(df["MINIMUM_PAYMENTS"].median(),inplace=True)
df["CREDIT_LIMIT"].fillna(df["CREDIT_LIMIT"].median(),inplace=True)
X=df.iloc[:,1:]

ID =df.CUST_ID

display(X.head())

display(ID.head())
X.duplicated().any()
ID.duplicated().any()
df.boxplot(column=[ 'BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES',

       'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE',

       'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY',

       'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY',

       'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'CREDIT_LIMIT', 'PAYMENTS',

       'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT', 'TENURE'],figsize=(15,15))

scale = StandardScaler()

        

X = scale.fit_transform(X)

X.shape
X
n=30

cost =[]

for i in range(1,30):

    km =KMeans(n_clusters=i)

    km.fit(X)

    cost.append(km.inertia_)
plt.plot(cost,'bx-')
km=KMeans(n_clusters=6)

km.fit(X)

labels=km.labels_
IDLabel =pd.DataFrame({"ID":ID,"label":labels})

IDLabel.head()
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



  

df = pd.DataFrame({'x': x, 'y':y, 'label':labels}) 



groups = df.groupby('label')

fig, ax = plt.subplots(figsize=(20, 13)) 



for name, group in groups:

    ax.plot(group.x, group.y, marker='o', linestyle='', ms=5,

            color=colors[name], mec='none')

    ax.set_aspect('auto')

    ax.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')

    ax.tick_params(axis= 'y',which='both',left='off',top='off',labelleft='off')

    

ax.legend()

ax.set_title("Customers Segmentation based on their Credit Card usage bhaviour.")

plt.show()

len(labels[labels==3])