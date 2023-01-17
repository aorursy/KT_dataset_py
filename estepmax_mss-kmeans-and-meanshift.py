import numpy as np

import pandas as pd

import seaborn as sb

import matplotlib.pyplot as plt



from sklearn.cluster import KMeans,MeanShift,estimate_bandwidth

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder



df = pd.read_csv('../input/Mall_Customers.csv')

list(df)



pd.options.mode.chained_assignment = None 
df.describe()
plt.rcParams['figure.figsize'] = (10,8)

sb.jointplot(x='Annual Income (k$)',y='Age',data=df,kind="kde")
sb.scatterplot(x='Age',y='Spending Score (1-100)',data=df)
F = df[df.Gender == 'Female']

np.mean(F)
M = df[df.Gender == 'Male']

np.mean(M)
sb.scatterplot(x='Annual Income (k$)',y='Spending Score (1-100)',data=df)
features = ['Annual Income (k$)','Spending Score (1-100)']

X = df[features]



kmeans = KMeans(n_clusters=5,random_state=13).fit(X)

centers = kmeans.cluster_centers_

labels = kmeans.labels_



X['labels'] = labels
sb.scatterplot(x='Annual Income (k$)',y='Spending Score (1-100)',data = X,hue='labels',palette="Set1",legend="full")
features = ['Age','Spending Score (1-100)']

X = df[features]



bandwidth = estimate_bandwidth(X,quantile=0.20)

meanshift = MeanShift(bandwidth=bandwidth,bin_seeding=True)

meanshift.fit(X)



labels = meanshift.labels_



X['labels'] = labels
sb.scatterplot(x='Age',y='Spending Score (1-100)',data = X,hue='labels',palette="Set1")