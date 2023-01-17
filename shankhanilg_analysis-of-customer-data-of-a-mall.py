# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

warnings.filterwarnings("ignore")

import os

print(os.listdir("../input"))







# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/Mall_Customers.csv")



pd.set_option('display.max_columns', 10)



print("Data Sample\n{}\n".format(data.head()))

print("Data description\n{}\n".format(data.describe()))

print("Data types\n{}\n". format(data.dtypes))
# getting the list of columns



columns = data.columns



print(columns)
import matplotlib.pyplot as plt

import seaborn as sns



n=0



plt.figure(1, figsize = (25,6))



for i in ['Age', 'Annual Income (k$)','Spending Score (1-100)']:

    n = n+1

    plt.subplot(1,3,n)

    plt.subplots_adjust(hspace = 0.5, wspace = 0.5)

    sns.distplot(data[i], bins=20)

    plt.title("Distplot of {}".format(i))

plt.show()
# plot of Male count Vs Female count

plt.figure(1, figsize = (10,5))

sns.countplot(y = 'Gender', data = data)

plt.show()
# plotting [Age, income, spending score] with one another



plt.figure(1, figsize = (25,25))

n = 0



for x in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']:

    for y in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']:

        if( x != y ):

            n = n + 1

            plt.subplot(3, 3, n)

            plt.subplots_adjust(hspace = 0.5, wspace = 0.5)

            sns.regplot(x = x, y = y, data = data)

            plt.title("Plot of {} vs {}".format(x,y))



plt.show()
# plotting Age vs Annual income wrt Gender



plt.figure(1, figsize = (25,25) )

n = 0



for x in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']:

    for y in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']:

        if( x != y ):

            n = n + 1

            plt.subplot(3, 3, n)

            plt.subplots_adjust(hspace = 0.5, wspace = 0.5)

            for gender in ['Male', 'Female']:

                sns.regplot(x = x, y = y, data = data[ data['Gender'] == gender ], label = gender)

                plt.title("{} vs {} wrt Gender".format(x,y))

plt.legend()

plt.show()
# applying Elbow method to calculate the optimum K

from sklearn.cluster import KMeans

from scipy.spatial.distance import cdist



attr = ['Annual Income (k$)', 'Spending Score (1-100)']



K = range(1,10)

distorts = []

for k in K:

    kmeans = KMeans(k)

    kmeans.fit(data[attr])

    distorts.append(sum(np.min(cdist(data[attr], kmeans.cluster_centers_, 'euclidean'), axis=1)) / data.shape[0])



plt.plot(K, distorts)

plt.xlabel('K')

plt.ylabel('Distortions')

plt.title("Plot for K vs Distortions")
# Applying K means clustering on Annual Income and Spending score



attr = ['Annual Income (k$)', 'Spending Score (1-100)']



# From the result of the elbow algorithm, the elbow occurs at K = 5



N = 5



kmeans = KMeans(n_clusters = 5 ,init='k-means++', n_init = 10 ,max_iter=300,

                tol=0.0001,  random_state= 111  , algorithm='elkan')



kmeans.fit(data[attr])



centroids = kmeans.cluster_centers_



print("The centroids are:{}".format(centroids))



cluster = kmeans.fit_predict(data[attr])



data['cluster'] = cluster
# visualising the clusters and the centroids.





for c in range(0,5):

    plt.scatter(x = data[data['cluster'] == c]['Annual Income (k$)'], 

               y = data[data['cluster'] == c]['Spending Score (1-100)'], label = 'Cluster{}'.format(c))



for c in range(0,5):

    plt.scatter(x = centroids[c][0], 

               y = centroids[c][1],s = 300, label = 'Centroid{}'.format(c))

    

plt.title("Cluster plot of Annual income vs Spending Score")

plt.xlabel('Annual Income (k$)')

plt.ylabel('Spending Score (1-100)')

plt.legend()

plt.show()
# count plot of all the clusters



plt.figure(1, figsize = (10,6))

sns.countplot(y = 'cluster', data = data)

plt.title('Count plot of cusomters wrt clusters')

plt.show()
# Analysis of the customers belonging to "High spending score and low income" cluster, aka cluster1



data_c0 = data[data['cluster'] == 0]

data_c0.head(10)


data_c1 = data[data['cluster'] == 1]

data_c1.head(10)
# analysing the age distribution of cluster 0 data wrt gender



plt.figure(1, figsize = (10,6))

for gender in ['Male', 'Female']:

    sns.distplot(data_c0[data_c0['Gender'] == gender]['Age'], hist=False, rug = True, label=gender)

plt.title('Age distribution for Cluster 0 cusomters wrt Gender')

plt.legend()

plt.show()
# Visualizing Spending score distribution of cluster 0 customers wrt gender



plt.figure(1, figsize = (10,6))

for gender in ['Male', 'Female']:

    sns.distplot(data_c0[data_c0['Gender'] == gender]['Spending Score (1-100)'], hist=False, rug = True, label=gender)

plt.title('Spending Score distribution for Cluster 0 cusomters wrt Gender')

plt.legend()

plt.show()
# analysing the age distribution of cluster 1 data wrt gender



plt.figure(1, figsize = (10,6))

for gender in ['Male', 'Female']:

    sns.distplot(data_c1[data_c1['Gender'] == gender]['Age'], hist=False, rug = True, label=gender)

plt.title('Age distribution for Cluster 0 cusomters wrt Gender')

plt.legend()

plt.show()