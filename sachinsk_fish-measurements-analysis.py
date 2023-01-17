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
df=pd.read_csv('../input/fish.csv')

df.head()
df.columns = [0,1,2,3,4,5,6]

df.head()
X=df.drop(0, axis=1).values

y=df[0].values

# Perform the necessary imports

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans



# Create scaler: scaler

scaler = StandardScaler()



# Create KMeans instance: kmeans

kmeans = KMeans(n_clusters=4)



# Create pipeline: pipeline

pipeline = make_pipeline(scaler,kmeans)

# Import pandas

import pandas as pd



# Fit the pipeline to samples

pipeline.fit(X)



# Calculate the cluster labels: labels

labels = pipeline.predict(X)



# Create a DataFrame with labels and species as columns: df

df = pd.DataFrame({'labels':labels,'species':y})



# Create crosstab: ct

ct = pd.crosstab(df['labels'],df.species)



# Display ct

print(ct)

# Perform the necessary imports

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline

import matplotlib.pyplot as plt



# Create scaler: scaler

scaler = StandardScaler()



# Create a PCA instance: pca

pca = PCA()



# Create pipeline: pipeline

pipeline = make_pipeline(scaler,pca)



# Fit the pipeline to 'samples'

pipeline.fit(X)



# Plot the explained variances

features = range(pca.n_components_)

plt.bar(features, pca.explained_variance_)

plt.xlabel('PCA feature')

plt.ylabel('variance')

plt.xticks(features)

plt.show()

pca = PCA(n_components=2)

scaled_samples=scaler.fit_transform(X)

# Fit the PCA instance to the scaled samples

pca.fit(scaled_samples)



# Transform the scaled samples: pca_features

pca_features = pca.transform(scaled_samples)



# Print the shape of pca_features

print(pca_features.shape)



