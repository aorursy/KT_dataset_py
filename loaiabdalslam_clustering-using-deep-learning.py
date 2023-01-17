# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="darkgrid")

from sklearn.cluster import KMeans







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/Mall_Customers.csv')
df.shape
df.describe()
df.dtypes
df.head()
df.drop(['CustomerID'],axis=1,inplace=True)
df.head()
df['Gender'].unique()
df.columns
sns.relplot(hue="Gender", x='Annual Income (k$)' ,y="Spending Score (1-100)", data=df);

sns.catplot(x="Gender", y="Spending Score (1-100)", kind="box", data=df);

explode = (0.1, 0)

fig1, ax1 = plt.subplots(figsize=(12,7))

ax1.pie(df['Gender'].value_counts(),explode=explode, labels=df['Gender'].unique(), autopct='%1.1f%%',

        shadow=True, startangle=90)

# Equal aspect ratio ensures that pie is drawn as a circle

ax1.axis('equal')  

plt.tight_layout()

plt.legend()

plt.show()
def pieChart(label):

    fig1, ax1 = plt.subplots(figsize=(12,7))

    ax1.pie(df[label].value_counts(), labels=df[label].unique(), autopct='%1.1f%%',

            shadow=True, startangle=90)

    # Equal aspect ratio ensures that pie is drawn as a circle

    ax1.axis('equal')  

    plt.tight_layout()

    plt.legend()

    plt.show()
pieChart('Gender')
X1 = df[['Age' , 'Spending Score (1-100)']].iloc[: , :].values

inertia = []

for n in range(1 , 11):

    algorithm = (KMeans(n_clusters = n ,init='k-means++', n_init = 10 ,max_iter=300, 

                        tol=0.0001,  random_state= 111  , algorithm='elkan') )

    algorithm.fit(X1)

    inertia.append(algorithm.inertia_)
plt.figure(1 , figsize = (15 ,6))

plt.plot(np.arange(1 , 11) , inertia , 'o')

plt.plot(np.arange(1 , 11) , inertia , '-' , alpha = 0.5)

plt.xlabel('Number of Clusters') , plt.ylabel('Inertia')

plt.show()
algorithm = (KMeans(n_clusters = 4 ,init='k-means++', n_init = 10 ,max_iter=300, 

                        tol=0.0001,  random_state= 111  , algorithm='elkan') )

algorithm.fit(X1)

labels1 = algorithm.labels_

centroids1 = algorithm.cluster_centers_
h = 0.02

x_min, x_max = X1[:, 0].min() - 1, X1[:, 0].max() + 1

y_min, y_max = X1[:, 1].min() - 1, X1[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()]) 
plt.figure(1 , figsize = (15 , 7) )

plt.clf()

Z = Z.reshape(xx.shape)

plt.imshow(Z , interpolation='nearest', 

           extent=(xx.min(), xx.max(), yy.min(), yy.max()),

           cmap = plt.cm.Pastel2, aspect = 'auto', origin='lower')



plt.scatter( x = 'Age' ,y = 'Spending Score (1-100)' , data = df , c = labels1 , 

            s = 200 )

plt.scatter(x = centroids1[: , 0] , y =  centroids1[: , 1] , s = 300 , c = 'red' , alpha = 0.5)

plt.ylabel('Spending Score (1-100)') , plt.xlabel('Age')

plt.show()