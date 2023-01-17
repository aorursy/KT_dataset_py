# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly as py

import plotly.graph_objs as go

from sklearn.cluster import KMeans

import warnings

import os

from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

py.offline.init_notebook_mode(connected=True)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv")
df.head()
df.shape
df.describe()
label_encoder = LabelEncoder()
int_enc = label_encoder.fit_transform(df.iloc[:,1].values)

df['Gender'] = int_enc

df.head()
df.head()
df.dtypes
df.isnull().any()
sns.pairplot(df.iloc[:,1:5])
#HEATMAP

hm = sns.heatmap(df.corr(),annot=True,linewidths=5,cmap='Blues')
plt.figure(1,figsize=(15,8))

n = 0

for i in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']:

    n += 1

    plt.subplot(1,3,n)

    plt.subplots_adjust(hspace=0.5,wspace=0.5)

    sns.distplot(df[i],bins=20)

    plt.title('Displot of {}'.format(i))

plt.show()
labels = ["Female", "Male"]

size = df['Gender'].value_counts()

colors = ['pink','orange']

explode=[0, 0.1]



plt.figure(1,figsize=(9,9))

plt.pie(size,colors=colors,explode=explode, labels = labels,shadow=True, autopct= '%.2f%%')



plt.title('Gender', fontsize=20)

plt.axis('off')

plt.legend()

plt.show()
x1 = df[['Age', 'Spending Score (1-100)']].iloc[: , :].values

inertia=[]



for n in range(1,11):

    algorithm =(KMeans(n_clusters=n, init='k-means++',n_init=10, max_iter=300,tol=0.0001,random_state=111,algorithm='elkan'))

    algorithm.fit(x1)

    inertia.append(algorithm.inertia_)
plt.figure(1,figsize=(15,6))

plt.plot(np.arange(1,11), inertia, 'o')

plt.plot(np.arange(1,11), inertia, '-', alpha=0.5)

plt.xlabel('Number of clusters'), plt.ylabel('Inertia')

plt.show()
algorithm = (KMeans(n_clusters = 5 ,init='k-means++', n_init = 10 ,max_iter=300, 

                        tol=0.0001,  random_state= 111  , algorithm='elkan') )

algorithm.fit(x1)

labels1 = algorithm.labels_

centroids1 = algorithm.cluster_centers_
h = 0.02

x_min,x_max = x1[:,0].min() - 1, x1[:,0].max() + 1

y_min,y_max = x1[:,1].min() - 1, x1[:,1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))

z1 = algorithm.predict(np.c_[xx.ravel(),yy.ravel()])
plt.figure(1 , figsize = (15 , 7) )

plt.clf()

z1 = z1.reshape(xx.shape)

plt.imshow(z1 , interpolation='nearest', 

           extent=(xx.min(), xx.max(), yy.min(), yy.max()),

           cmap = plt.cm.Pastel1, aspect = 'auto', origin='lower')



plt.scatter( x = 'Age' ,y = 'Spending Score (1-100)' , data = df , c = labels1, 

            s = 200 )

plt.scatter(x = centroids1[: , 0] , y =  centroids1[: , 1] , s = 300 , c = 'red' , alpha = 0.5)

plt.ylabel('Spending Score (1-100)') , plt.xlabel('Age')

plt.show()
x2 = df[['Annual Income (k$)', 'Spending Score (1-100)']].iloc[: , :].values

inertia=[]



for n in range(1,11):

    algorithm =(KMeans(n_clusters=n, init='k-means++',n_init=10, max_iter=300,tol=0.0001,random_state=111,algorithm='elkan'))

    algorithm.fit(x1)

    inertia.append(algorithm.inertia_)
plt.figure(1,figsize=(15,6))

plt.plot(np.arange(1,11), inertia, 'o')

plt.plot(np.arange(1,11), inertia, '-', alpha=0.5)

plt.xlabel('Number of clusters'), plt.ylabel('Inertia')

plt.show()
algorithm = (KMeans(n_clusters = 5 ,init='k-means++', n_init = 10 ,max_iter=300, 

                        tol=0.0001,  random_state= 111  , algorithm='elkan') )

algorithm.fit(x2)

labels2 = algorithm.labels_

centroids2 = algorithm.cluster_centers_
h = 0.02

x_min,x_max = x2[:,0].min() - 1, x2[:,0].max() + 1

y_min,y_max = x2[:,1].min() - 1, x2[:,1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))

z2 = algorithm.predict(np.c_[xx.ravel(),yy.ravel()])
plt.figure(1 , figsize = (15 , 7) )

plt.clf()

z2 = z2.reshape(xx.shape)

plt.imshow(z2 , interpolation='nearest', 

           extent=(xx.min(), xx.max(), yy.min(), yy.max()),

           cmap = plt.cm.Pastel2, aspect = 'auto', origin='lower')



plt.scatter( x = 'Annual Income (k$)' ,y = 'Spending Score (1-100)' , data = df , c = labels2, 

            s = 200 )

plt.scatter(x = centroids2[: , 0] , y =  centroids2[: , 1] , s = 300 , c = 'red' , alpha = 0.5)

plt.ylabel('Spending Score (1-100)') , plt.xlabel('Annual Income (k$)')

plt.show()
x = df['Annual Income (k$)']

y = df['Age']

z = df['Spending Score (1-100)']



sns.lineplot(x,y,color='blue')

sns.lineplot(x,z,color='pink')

plt.title('Annual Income vs Age and Spending Score', fontsize=25)

plt.show()