%reset -f

# 1.1 Data manipulation



import pandas as pd

import numpy as np

# 1.2 Import GaussianMixture class

from sklearn.mixture import GaussianMixture

# 1.3 Plotting

import matplotlib.pyplot as plt

from matplotlib.pyplot import imshow

from matplotlib.colors import LogNorm

import seaborn as sns

# 1.4 For data processing

from sklearn.preprocessing import StandardScaler

# 1.4

import time

from sklearn.cluster import KMeans

# 1.1 For creating elliptical-shaped clusters

from sklearn.datasets import make_blobs

# 1.4 TSNE

from sklearn.manifold import TSNE

# reading dataset

df = pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')

#df.info()



#print(df.columns)

df.shape

df.head()



#df.describe()
# renaming the columns 

print(df.columns)  

df.rename(columns = {'CustomerID':'CID','Gender':'GEN','Age':'AGE', 'Annual Income (k$)':'AI (Rs)','Spending Score (1-100)':'SCORE'

                              },inplace=True) 

df.shape

df.head()

#df.describe()    
# dropping customerid column and transforming Gender column to [0,1]



df.drop(['CID'], axis=1,inplace=True)

#df.info()



df['GEN'].replace(['Male','Female'],[0,1])



#df.shape

#df.head()

df.describe()
sns.countplot(df["GEN"])

sns.stripplot(x="GEN", y="AGE", data=df)

sns.pairplot(hue="GEN",data=df)

sns.pointplot(x="GEN", y="SCORE", data=df)

sns.catplot(x="GEN", y="AGE", data=df)


sns.barplot(x="GEN", y="SCORE", data=df)
# Create Scatter Plot:



#df.shape     # (1000,2)

df.plot.scatter(x='GEN', y='SCORE', c='blue', s=5)

plt.show()
# 4.0 Import GaussianMixture class

from sklearn.mixture import GaussianMixture



# 4.1 Perform clsutering

gm = GaussianMixture(

                     n_components = 3,

                     n_init = 10,

                     max_iter = 100)



# 4.2 Train the algorithm

gm.fit(X)



# 4.3 Where are the clsuter centers

gm.means_



# 4.6 Clusters labels

gm.predict(X)



# 5.0 Plot cluster and cluster centers

#     both from kmeans and from gmm



fig = plt.figure()



# 5.1 from kmeans

plt.scatter(X[:, 0], X[:, 1],

            c=gm.predict(X),

            s=2)

plt.show()

# 5.2 from gmm

plt.scatter(gm.means_[:, 0], gm.means_[:, 1],

            marker='v',

            s=5,               # marker size

            linewidths=5,      # linewidth of marker edges

            color='red'

            )

plt.show()

# 3.0 Apply kmeans

kmeans = KMeans(n_clusters=3,

                    n_init =10,

                    max_iter = 800)

kmeans.fit(X)



# 3.1 Get cluster centers

centroids=kmeans.cluster_centers_



# 3.2 Plot clusters and cluster centers

fig = plt.figure()

plt.scatter(X[:, 0], X[:, 1],

            c=kmeans.labels_,

            s=2)

plt.scatter(centroids[:, 0], centroids[:, 1],

            marker='x',

            s=100,               # marker size

            linewidths=150,      # linewidth of marker edges

            color='red'

            )

plt.show()



# 4.1 Perform clsutering

gm = GaussianMixture(

                     n_components = 3,

                     n_init = 10,

                     max_iter = 100)



# 4.2 Train the algorithm

gm.fit(X)



# 4.3 Where are the clsuter centers

gm.means_



# 4.4 Did algorithm converge?

gm.converged_



# 4.5 How many iterations did it perform?

gm.n_iter_



# 4.6 Clusters labels

gm.predict(X)
# 8.0 How many clusters?

#     Use either AIC or BIC as criterion



bic = []

aic = []

for i in range(8):

    gm = GaussianMixture(

                     n_components = i+1,

                     n_init = 10,

                     max_iter = 100)

    gm.fit(X)

    bic.append(gm.bic(X))

    aic.append(gm.aic(X))

    

    gm = GaussianMixture(

                     n_components = 3,

                     n_init = 10,

                     max_iter = 100)

    gm.fit(X)

       

    

fig = plt.figure()

plt.plot([1,2,3,4,5,6,7,8], aic)

plt.plot([1,2,3,4,5,6,7,8], bic)

plt.show()



# Anomaly detection

densities = gm.score_samples(X)

densities



density_threshold = np.percentile(densities,4)

density_threshold



anomalies = X[densities < density_threshold]

anomalies

anomalies.shape



# Show anomalous points

fig = plt.figure()

plt.scatter(X[:, 0], X[:, 1], c = gm.predict(X))

plt.scatter(anomalies[:, 0], anomalies[:, 1],

            marker='x',

            s=50,               # marker size

            linewidths=5,      # linewidth of marker edges

            color='red'

            )

plt.show()



#  Get first unanomalous data

unanomalies = X[densities >= density_threshold]

unanomalies.shape    # (1200, 2)



#  Transform both anomalous and unanomalous data

#     to pandas DataFrame

df_anomalies = pd.DataFrame(anomalies, columns = ['x', 'y'])

df_anomalies['z'] = 'anomalous'   # Create a IIIrd constant column

df_normal = pd.DataFrame(unanomalies, columns = ['x','y'])

df_normal['z'] = 'unanomalous'    # Create a IIIrd constant column





#  Let us see density plots

sns.distplot(df_anomalies['x'])

sns.distplot(df_normal['x'])



#  Draw side-by-side boxplots

#  Ist stack two dataframes

df = pd.concat([df_anomalies,df_normal])

#  Draw featurewise boxplots

sns.boxplot(x = df['z'], y = df['y'])

sns.boxplot(x = df['z'], y = df['x'])
