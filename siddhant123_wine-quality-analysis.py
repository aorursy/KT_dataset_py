# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



import scipy 

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('/kaggle/input/wine-quality-clustering-unsupervised/winequality-red.csv')

df.head()
df.describe()
df.info()
df.isnull().any()
df.columns
df.corr()
plt.figure(figsize = (15,12))

sns.heatmap(df.corr(),cmap = 'inferno',annot = True)

plt.show()
slope,intercept,r_value,p_value,std_err = scipy.stats.linregress(x = df['fixed_acidity'],y = df['pH'])

reg1 = sns.regplot(x = 'fixed_acidity', y = 'pH', data = df, line_kws = 

                   {'label':"y = {0:f}x + {1:f}".format(slope,intercept)})

reg1.legend()

print("The correlation coefficient is " + str(r_value))
slope,intercept,r_value,p_value,std_err = scipy.stats.linregress(x = df['citric_acid'],y = df['pH'])

reg1 = sns.regplot(x = 'citric_acid', y = 'pH', data = df, line_kws = 

                   {'label':"y = {0:f}x + {1:f}".format(slope,intercept)})

reg1.legend()

print("The correlation coefficient is " + str(r_value))
slope,intercept,r_value,p_value,std_err = scipy.stats.linregress(x = df['volatile_acidity'],y = df['citric_acid'])

reg1 = sns.regplot(x = 'volatile_acidity', y = 'citric_acid', data = df, line_kws = 

                   {'label':"y = {0:f}x + {1:f}".format(slope,intercept)})

reg1.legend()

print("The correlation coefficient is " + str(r_value))
slope,intercept,r_value,p_value,std_err = scipy.stats.linregress(x = df['density'],y = df['fixed_acidity'])

reg1 = sns.regplot(x = 'density', y = 'fixed_acidity', data = df, line_kws = 

                   {'label':"y = {0:f}x + {1:f}".format(slope,intercept)})

reg1.legend()

print("The correlation coefficient is " + str(r_value))
slope,intercept,r_value,p_value,std_err = scipy.stats.linregress(x = df['citric_acid'],y = df['fixed_acidity'])

reg1 = sns.regplot(x = 'citric_acid', y = 'fixed_acidity', data = df, line_kws = 

                   {'label':"y = {0:f}x + {1:f}".format(slope,intercept)})

reg1.legend()

print("The correlation coefficient is " + str(r_value))
slope,intercept,r_value,p_value,std_err = scipy.stats.linregress(x = df['free_sulfur_dioxide'],y = df['total_sulfur_dioxide'])

reg1 = sns.regplot(x = 'free_sulfur_dioxide', y = 'total_sulfur_dioxide', data = df, line_kws = 

                   {'label':"y = {0:f}x + {1:f}".format(slope,intercept)})

reg1.legend()

print("The correlation coefficient is " + str(r_value))
sns.distplot(df['total_sulfur_dioxide'],bins = 20)
sns.distplot(df['free_sulfur_dioxide'],bins = 20)
sns.distplot(df['fixed_acidity'],bins = 20)
sns.distplot(df['volatile_acidity'],bins = 20)
sns.distplot(df['citric_acid'],bins = 20)
sns.distplot(df['residual_sugar'],bins = 20)
sns.distplot(df['chlorides'],bins = 20)
#density','pH', 'sulphates', 'alcohol', 'quality'

sns.distplot(df['density'],bins = 20)
sns.distplot(df['pH'],bins = 20)
sns.distplot(df['sulphates'],bins = 20)
sns.distplot(df['alcohol'],bins = 20)
X = df.iloc[:,[0,1]].values

from sklearn.cluster import KMeans

inertia = []

for i in range(1,5):

    kmeans = KMeans(n_clusters=i,init = 'k-means++',max_iter = 300,n_init = 10,random_state = 0)

    kmeans.fit(X)

    inertia.append(kmeans.inertia_)

plt.plot(range(1,5),inertia)

plt.title('The Elbow Method')

plt.xlabel('Number of Clusters')

plt.ylabel('Inertia')

plt.show()
#Applying KMeans to the Dataset



kmeans = KMeans(n_clusters=3,init = 'k-means++',max_iter = 300,n_init = 10,random_state = 0)

y_kmeans = kmeans.fit_predict(X)
plt.figure(figsize = (12,8))

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'cyan', label = 'Low Acidity')

plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'yellow', label = 'Medium Acidity')

plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'blue', label = 'High Acidity')

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:, 1], s = 50, c = 'red' , label = 'centroids')

plt.title('Fixed Acidity vs Volatile Acidity')

plt.xlabel('Fixed Acidity')

plt.ylabel('Volatile Acidity')

plt.legend()

plt.show()
X = df.iloc[:,[5,6]].values

from sklearn.cluster import KMeans

inertia = []

for i in range(1,5):

    kmeans = KMeans(n_clusters=i,init = 'k-means++',max_iter = 300,n_init = 10,random_state = 0)

    kmeans.fit(X)

    inertia.append(kmeans.inertia_)

plt.plot(range(1,5),inertia)

plt.title('The Elbow Method')

plt.xlabel('Number of Clusters')

plt.ylabel('Inertia')

plt.show()
#Applying KMeans to the Dataset



kmeans = KMeans(n_clusters=3,init = 'k-means++',max_iter = 300,n_init = 10,random_state = 0)

y_kmeans = kmeans.fit_predict(X)
plt.figure(figsize = (12,8))

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'magenta', label = 'Low Sulfur Dioxide')

plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'pink', label = 'Medium Sulfur Dioxide')

plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'High Sulfur Dioxide')

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:, 1], s = 50, c = 'red' , label = 'centroids')

plt.title('Fixed Acidity vs Volatile Acidity')

plt.xlabel('Fixed Acidity')

plt.ylabel('Volatile Acidity')

plt.legend()

plt.show()
X = df.iloc[:,[3,10]].values

from sklearn.cluster import KMeans

inertia = []

for i in range(1,11):

    kmeans = KMeans(n_clusters=i,init = 'k-means++',max_iter = 300,n_init = 10,random_state = 0)

    kmeans.fit(X)

    inertia.append(kmeans.inertia_)

plt.plot(range(1,11),inertia)

plt.title('The Elbow Method')

plt.xlabel('Number of Clusters')

plt.ylabel('Inertia')

plt.show()
#Applying KMeans to the Dataset



kmeans = KMeans(n_clusters=3,init = 'k-means++',max_iter = 300,n_init = 10,random_state = 0)

y_kmeans = kmeans.fit_predict(X)
plt.figure(figsize = (12,8))

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'purple', label = 'Low residual sugar and alcohol')

plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Medium residual sugar and alcohol')

plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'cyan', label = 'High residual sigar and alcohol')

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:, 1], s = 50, c = 'red' , label = 'centroids')

plt.title('Residual Sugar vs Alcohol')

plt.xlabel('Residual Sugar')

plt.ylabel('Alcohol')

plt.legend()

plt.show()
from sklearn.cluster import AgglomerativeClustering

from scipy.cluster.hierarchy import dendrogram,linkage
X = df.iloc[:,:11].values

y =  df.iloc[:,11].values

z = linkage(X,'ward')
dendrogram(z,truncate_mode='lastp',p = 12,leaf_rotation=45,leaf_font_size=15,show_contracted=True)

plt.title('Truncated Hierarchial Clustering')

plt.xlabel('Cluster Size')

plt.ylabel('Distance')



plt.axhline(y = 500)

plt.axhline(y = 150)

plt.show()