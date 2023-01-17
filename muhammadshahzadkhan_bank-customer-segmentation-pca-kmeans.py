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
# import libraries

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, normalize

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA
creditcard_df = pd.read_csv('/kaggle/input/credit-card-segmentation/CC GENERAL.csv')
creditcard_df
# By using following info function, we can see data types and get to know about null value existance 

#(i.e, credit limit and Min payments)

creditcard_df.info()
# By using following descibe function, we can get to know about important features of a coulmn, i.e, min, max and mean values

creditcard_df.describe()

#This helps to give insights about data, i.e, Balance is frequently updated on average ~0.9, scale-->(0,1)

# On average 15 percent people make full payment using CC
# Suppose we want to know about a person who made maximum "ONEOFF_PURCHASES" which is "40761.250000" given in above "describe" fun.

creditcard_df[creditcard_df['ONEOFF_PURCHASES']==40761.250000]
# Now lets get the features of customer who made the maximum cash advance transactions.

creditcard_df[creditcard_df['CASH_ADVANCE']>= 47137]
# Lets check misiing values, it seems that we have very less amount of missing values

sns.heatmap(creditcard_df.isnull(), yticklabels= False, cbar =False, cmap = 'winter_r')
#We can see we have 1 null value in "CREDIT_LIMIT" and 313 in "Minimum_Payments"

creditcard_df.isnull().sum()
# Lets fill these missing values with meab

creditcard_df.loc[(creditcard_df['MINIMUM_PAYMENTS'].isnull() == True), 'MINIMUM_PAYMENTS'] = creditcard_df['MINIMUM_PAYMENTS'].mean()

#We will use an alternate method to fill NAN value with mean in "CREDIT_LIMIT" coulmn

creditcard_df['CREDIT_LIMIT'].fillna(value=creditcard_df['CREDIT_LIMIT'].mean(), inplace= True)
creditcard_df['MINIMUM_PAYMENTS'].isnull().sum()
creditcard_df['CREDIT_LIMIT'].isnull().sum()
# So now we can see that we dont have any missing values left

sns.heatmap(creditcard_df.isnull(), yticklabels= False, cbar =False, cmap = 'winter_r')
# Now lets see if we have any duplicated entries and the result shows that all entries are unique

creditcard_df.duplicated().sum()
# Lets drop the ID column which dosent provide any info but a sequentail order

creditcard_df.drop(columns= 'CUST_ID', axis = 1, inplace= True)
print( 'Number of columns = {}'.format(len(creditcard_df.columns)))
creditcard_df.columns
# Now er are going to use dist_plot which is a combination of "hist" function in matplotlib and "KDE" in seaborn

# KDE is used to plot the probability distribution function of a variable

plt.figure(figsize=(10,50))

for i in range (len(creditcard_df.columns)):

    plt.subplot(17,1,i+1)

    sns.distplot(creditcard_df[creditcard_df.columns[i]], kde_kws= {'color' : 'b', 'lw': 3, 'label': 'KDE', 'bw': 1.5}, hist_kws= {'color' : 'g'})

    plt.title(creditcard_df.columns[i])

plt.tight_layout()
creditcard_df.head()
#Using Pearson Correlation

plt.figure(figsize=(12,10))

corr = creditcard_df.corr()

sns.heatmap(corr, annot=True, cmap=plt.cm.Reds)

plt.show()
#Lets re-scale data

scaler = StandardScaler()

creditcard_df_scaled = scaler.fit_transform(creditcard_df)
creditcard_df_scaled.shape
creditcard_df_scaled
# Now we are going to implement Elbow method to final optimal number of clusters

first_score = []

for i in range(1,20):

    kmeans = KMeans(n_clusters=i)

    kmeans.fit(creditcard_df_scaled)

    first_score.append(kmeans.inertia_) #inertia gives the within cluster distance of each point from its centroid as we discussed above.

plt.plot (first_score, 'bx')
# We can see from above plot that the optimal number of clusters in this case are 7 or 8.

# So lets apply kmeans method.

kmeans = KMeans(7)

kmeans.fit(creditcard_df_scaled)

labels = kmeans.labels_ #labels --> clusters
labels
kmeans.cluster_centers_.shape
# Lets create a dataframe consists of cluster centers

cluster_centers = pd.DataFrame(data = kmeans.cluster_centers_, columns= [creditcard_df.columns])

cluster_centers
# As the data is scaled so lets perform inverse transform to know better what this data actually means

cluster_centers = scaler.inverse_transform(cluster_centers)

cluster_centers = pd.DataFrame(data = cluster_centers, columns= [creditcard_df.columns])

cluster_centers

#We can seprate the four clusters given at the start of problem (i.e, Transactors, VIP) by monitoring the given attributes.
labels.shape # values associated to each poin
labels.max()
labels.min()
# Now we can have the label associated with each point

ykmeans = kmeans.fit_predict(creditcard_df_scaled)

ykmeans
#Lets concatenate the cluster labels with original data, which will help to plot the histograms of each cluster

creditcard_df_cluster = pd.concat([creditcard_df, pd.DataFrame({'cluster': labels})], axis = 1)

creditcard_df_cluster.head()
# Now lets plot histogram of each cluster

for i in creditcard_df.columns:

    plt.figure(figsize=(35,5))

    for j in range(7):

        plt.subplot(1,7,j+1)

        cluster = creditcard_df_cluster[creditcard_df_cluster['cluster']==j]

        cluster[i].hist(bins=20)

        plt.title('{} \nCluster {} '.format(i,j))

    plt.show()
#Lets convert our data to only 2D using PCA

pca = PCA(n_components=2)

pca_components = pca.fit_transform(creditcard_df_scaled)

pca_components
# create a dataframe of these two componenets

pca_df = pd.DataFrame(data = pca_components, columns = ['pca1', 'pca2'])

pca_df.head()
#concatenate with labels

pca_df = pd.concat([pca_df, pd.DataFrame({'cluster': labels})], axis = 1)

pca_df.head()
plt.figure(figsize=(10,10))

ax = sns.scatterplot(x='pca1', y='pca2', hue = 'cluster', data=pca_df, palette=['red', 'green', 'blue', 'pink', 'yellow', 'gray', 'black'])

plt.show()