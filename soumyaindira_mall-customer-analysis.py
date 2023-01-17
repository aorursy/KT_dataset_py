# 1.1 Call libraries

%reset -f

import warnings

warnings.filterwarnings('ignore')

# 1.2 For data manipulations

import numpy as np

import pandas as pd

import seaborn as sns

# 1.3 For plotting

import matplotlib.pyplot as plt

import matplotlib

import matplotlib.pyplot

import plotly.express as px

# 1.4 For data processing

from sklearn.preprocessing import StandardScaler

# 1.4 OS related

import os

# 1.5 Modeling librray

# 1.5.1 Scale data

from sklearn.preprocessing import StandardScaler

# 1.5.2 Split dataset

from sklearn.model_selection import train_test_split

# 1.5.3 Class to develop kmeans model

from sklearn.cluster import KMeans

# 1.6 Plotting library

import seaborn as sns

# 1.7 Import GaussianMixture class

from sklearn.mixture import GaussianMixture

# 1.8 How good is clustering?

from sklearn.metrics import silhouette_score

from yellowbrick.cluster import SilhouetteVisualizer

# 1.9 TSNE

from sklearn.manifold import TSNE

warnings.filterwarnings('ignore')
%matplotlib inline
#Load the data frame

Mall_df1=pd.read_csv("../input//customer-segmentation-tutorial-in-python/Mall_Customers.csv")
Mall_df1.head()
Mall_df1.drop('CustomerID',axis=1,inplace=True)
mapping = { "Gender" : {"Male":0, "Female":1}} 

Mall_df1.replace(mapping, inplace=True)
Mall_df1.head()
Mall_df1.describe()
#Renaming column names

Mall_df1.columns=Mall_df1.columns.str.replace('(','')

Mall_df1.columns=Mall_df1.columns.str.replace(')','')

Mall_df1.columns=Mall_df1.columns.str.replace('k','')

Mall_df1.columns=Mall_df1.columns.str.replace('$','')

Mall_df1.columns=Mall_df1.columns.str.replace('1-100','')

Mall_df1.columns=Mall_df1.columns.str.replace(' ','_')

Mall_df1.columns=Mall_df1.columns.str.replace('Annual_Income_','Annual_Income')

Mall_df1.columns=Mall_df1.columns.str.replace('Spending_Score_','Spending_Score')
Mall_df1.head()
#Comparison of annual income with respect to age

plt.subplot(1, 2, 1)

sns.distplot(Mall_df1['Annual_Income'],color = 'yellow')

plt.title('Distribution of Annual Income', fontsize = 15)

plt.xlabel('Range of Annual Income')

plt.ylabel('Count')



plt.subplot(1, 2, 2)

sns.distplot(Mall_df1['Age'], color = 'Green')

plt.title('Distribution of Age', fontsize = 15)

plt.xlabel('Range of Age')

plt.ylabel('Count')

#Analysing annual income genderwise

sns.boxplot( x= 'Gender', y = 'Annual_Income', data = Mall_df1 )
#Annual income 

sns.lineplot('Annual_Income', 'Age', color = 'blue',data=Mall_df1)

sns.lineplot('Annual_Income','Spending_Score', color = 'green',data=Mall_df1)

plt.title('Annual Income vs Age and Spending Score', fontsize = 20)

plt.show()
#Gender Vs Spending score

sns.violinplot(Mall_df1['Gender'], Mall_df1['Spending_Score'], palette = 'CMRmap')

plt.title('Gender vs Spending_Score', fontsize = 14)

plt.show()
x = Mall_df1['Gender'].values

Mall_df2=Mall_df1[['Age','Annual_Income','Spending_Score']]
ss = StandardScaler()           # Create an instance of class

ss.fit(Mall_df2)                # Train object on the data

X = ss.transform(Mall_df2)      # Transform data

X[:5, :]                        # See first 5 rows
# Perform clsutering

gm = GaussianMixture(

                     n_components = 3,

                     n_init = 10,

                     max_iter = 100)



# Train the algorithm

gm.fit(X)



#Where are the clsuter centers

gm.means_



# Did algorithm converge?

gm.converged_


#How many iterations did it perform?

gm.n_iter_

#Clusters labels

gm.predict(X)
#Weights of respective gaussians.

gm.weights_
# 4.7.1 What is the frequency of data-points



np.unique(gm.predict(X), return_counts = True)[1]/len(X)

# GMM is a generative model.

# Generate a sample from each cluster

# ToDo: Generate digits using MNIST

gm.sample()
#Plot cluster and cluster centers from gmm

fig = plt.figure()



plt.scatter(X[:, 0], X[:, 1],

            c=gm.predict(X),

            s=10)

plt.scatter(gm.means_[:, 0], gm.means_[:, 1],

            marker='v',

            s=8,               # marker size

            linewidths=8,      # linewidth of marker edges

            color='green'

            )

plt.show()
#  Anomaly detection

#     Anomalous points are those that

#     are in low-density region

#     Or where density is in low-percentile

#     of 4%

#     score_samples() method gives score or

#     density of a point at any location.

#     Higher the value, higher its density



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
# Get first unanomalous data

unanomalies = X[densities >= density_threshold]

unanomalies.shape    # (1200, 2)
df_anomalies = pd.DataFrame(anomalies, columns = ['x', 'y','z'])

df_anomalies['a'] = 'anomalous'   # Create a 4th constant column

df_normal = pd.DataFrame(unanomalies, columns = ['x','y','z'])

df_normal['a'] = 'unanomalous'   
# Let us see density plots

sns.distplot(df_anomalies['x'])

sns.distplot(df_normal['x'])
# Draw side-by-side boxplots

# Ist stack two dataframes

df = pd.concat([df_anomalies,df_normal])

# Draw featurewise boxplots

sns.boxplot(x = df['a'], y = df['x'])



# 8.0 How many clusters?

#     Use either AIC or BIC as criterion

#     Ref: https://en.wikipedia.org/wiki/Akaike_information_criterion

#          https://en.wikipedia.org/wiki/Bayesian_information_criterion

#          https://www.quora.com/What-is-an-intuitive-explanation-of-the-Akaike-information-criterion

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
fig = plt.figure()

plt.plot([1,2,3,4,5,6,7,8], aic)

plt.plot([1,2,3,4,5,6,7,8], bic)

plt.show()
tsne = TSNE(n_components = 2)

tsne_out = tsne.fit_transform(X)

plt.scatter(tsne_out[:, 0], tsne_out[:, 1],

            marker='x',

            s=50,              # marker size

            linewidths=5,      # linewidth of marker edges

            c=gm.predict(X)   # Colour as per gmm

            )