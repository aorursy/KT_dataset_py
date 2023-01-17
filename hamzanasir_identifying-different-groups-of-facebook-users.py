import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



%matplotlib inline
import os



os.chdir('/kaggle/')

os.listdir('/kaggle/input')
data = pd.read_csv('/kaggle/input/fb_users.csv')

df = data.copy()  # To keep the data as backup



df.head()
df.shape
df.isnull().sum()
df.describe()
from sklearn.preprocessing import MinMaxScaler



# Let's instantiate MinMaxScaler object

scaler = MinMaxScaler()



# Scale

scaled_array = scaler.fit_transform(X=df)



type(scaled_array)
df_scaled = pd.DataFrame(data=scaled_array, columns=df.columns)

df_scaled.head()
# First, we need to import the sklearn's KMeans library

from sklearn.cluster import KMeans



cost = []



for k in range(1, 16):

    kmean = KMeans(n_clusters=k, random_state=0)

    kmean.fit(df_scaled)

    cost.append([k, kmean.inertia_])



# Plotting the K's against cost using matplotlib library which we imported at the start

plt.figure(figsize=(15, 8))

plt.plot(pd.DataFrame(cost)[0], pd.DataFrame(cost)[1])

plt.xlabel('K')

plt.ylabel('Cost')

plt.title('Elbow Analysis')

plt.grid(True)



plt.show()
# Let's first import the relevant library

from sklearn.metrics import silhouette_score



score = []



for k in range(2, 16):

    kmean = KMeans(n_clusters=k, random_state=0)

    kmean.fit(df_scaled)

    score.append([k, silhouette_score(df_scaled, kmean.labels_)])



# Let's plot the Number of K's against Silhouette Score

plt.figure(figsize=(15, 8))

plt.plot(pd.DataFrame(score)[0], pd.DataFrame(score)[1])

plt.xlabel('K')

plt.ylabel('Silhouette Score')

plt.grid(True)

plt.title("Silhouette Score against each number of clusters")



plt.show()
# Let's make a new dataframe named 'predict' and copy the contents of our original df into it

pred = df.copy()



# Let's train the model based on 3 clusters

kmean_3k = KMeans(n_clusters=3, random_state=0)

labels = kmean_3k.fit_predict(df_scaled)



pred['clusters'] = labels

pred.head()
pivoted = pred.groupby(['clusters']).median().reset_index()

pivoted
pred.clusters.value_counts()
import seaborn as sns

sns.set()



plt.figure(figsize=(10, 6))

sns.countplot(x='clusters', data=pred)