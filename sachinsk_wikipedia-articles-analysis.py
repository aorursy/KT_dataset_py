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
df= pd.read_csv('../input/wikipedia-vectors.csv')

df.head()
articles=df.values

titles= df['Unnamed: 0'].values

# titles= titles.reshape(-1,1)

# Perform the necessary imports

from sklearn.decomposition import TruncatedSVD

from sklearn.cluster import KMeans

from sklearn.pipeline import make_pipeline



# Create a TruncatedSVD instance: svd

svd = TruncatedSVD(n_components=50)



# Create a KMeans instance: kmeans

kmeans = KMeans(n_clusters=6)



# Create a pipeline: pipeline

pipeline = make_pipeline(svd,kmeans)

# # Import pandas

# import pandas as pd



# Fit the pipeline to articles

pipeline.fit(articles)



# Calculate the cluster labels: labels

labels = pipeline.predict(articles)

# labels= labels.reshape(-1,1)

# Create a DataFrame aligning labels and titles: df

df = pd.DataFrame({'label': labels, 'article': titles})



# Display df sorted by cluster label

print(df.sort_values(by='label'))
print(labels.shape)

print(titles.shape)


# Import NMF

from sklearn.decomposition import NMF



# Create an NMF instance: model

model = NMF(n_components=6)



# Fit the model to articles

model.fit(articles)



# Transform the articles: nmf_features

nmf_features = model.transform(articles)



# Print the NMF features

print(nmf_features)

articles.shape