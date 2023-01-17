import numpy as np 

import pandas as pd

import seaborn as sns

import pandas_profiling as pp

import matplotlib.pyplot as plt

from sklearn import decomposition

from sklearn.cluster import KMeans

from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('../input/rural-residents-daily-mobile-phone-data/datashare.csv')

df.head()
df.info()
labelencoder = LabelEncoder()



df['rural']                     = labelencoder.fit_transform(df['rural'])   

df['Diversity']                 = labelencoder.fit_transform(df['Diversity'])   

df['Number']                    = labelencoder.fit_transform(df['Number'])  

df['Standard_distance_nonwork'] = labelencoder.fit_transform(df['Standard_distance_nonwork'])  

df['Standard_distance_nwork']   = labelencoder.fit_transform(df['Standard_distance_nwork'])  

df['Distance_to_central_city']  = labelencoder.fit_transform(df['Distance_to_central_city']) 

df['Slope']                     = labelencoder.fit_transform(df['Slope']) 

df['Work_in_urban_areas']       = labelencoder.fit_transform(df['Work_in_urban_areas']) 
df.head()
pp.ProfileReport(df)
#thank you https://www.kaggle.com/kashnitsky/topic-7-unsupervised-learning-pca-and-clustering



X = df.drop('rural',axis=1)

y = df['rural']



pca = decomposition.PCA(n_components=2)

X_reduced = pca.fit_transform(X)



print('Projecting %d-dimensional data to 2D' % X.shape[1])



plt.figure(figsize=(12,10))

plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, 

            edgecolor='none', alpha=0.5, s=20,

            cmap=plt.cm.get_cmap('nipy_spectral', 10))

plt.colorbar()

plt.title('PCA projection');
elbow = []

for i in range(1, 11):

    kmeans = KMeans(n_clusters=i).fit(df)

    elbow.append(np.sqrt(kmeans.inertia_))

    

plt.plot(range(1, 11), elbow, marker='s');

plt.xlabel('Elbow Number')

plt.ylabel('J');