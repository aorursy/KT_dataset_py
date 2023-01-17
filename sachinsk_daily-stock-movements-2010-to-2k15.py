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
df=pd.read_csv('../input/company-stock-movements-2010-2015-incl.csv')

df.head()
df.shape

movements=df.drop('Unnamed: 0',axis=1).values

companies=df['Unnamed: 0'].values
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import Normalizer

from sklearn.cluster import KMeans



# Create a normalizer: normalizer

normalizer = Normalizer()



# Create a KMeans model with 10 clusters: kmeans

kmeans = KMeans(n_clusters=10)



# Make a pipeline chaining normalizer and kmeans: pipeline

pipeline = make_pipeline(normalizer,kmeans)



# Fit pipeline to the daily price movements

pipeline.fit(movements)

# Import pandas

import pandas as pd



# Predict the cluster labels: labels

labels = pipeline.predict(movements)



# Create a DataFrame aligning labels and companies: df

df = pd.DataFrame({'labels': labels, 'companies': companies})



# Display df sorted by cluster label

print(df.sort_values(by='labels'))

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

import matplotlib.pyplot as plt



# Calculate the linkage: mergings

mergings = linkage(movements, method='complete')



# Plot the dendrogram, using varieties as labels

dendrogram(mergings,

           labels=companies,

           leaf_rotation=90,

           leaf_font_size=7

)

plt.show()
labels= fcluster(mergings,20,criterion='distance')

print(labels)
from sklearn.preprocessing import normalize



# Normalize the movements: normalized_movements

normalized_movements = normalize(movements)



# Calculate the linkage: mergings

mergings = linkage(normalized_movements,method='complete')



# Plot the dendrogram

dendrogram(mergings,labels=companies,leaf_rotation=90,leaf_font_size=6)

plt.show()
labels= fcluster(mergings,t=1.2,criterion='distance')

print(labels)
print(df.sort_values('labels'))
# Calculate the linkage: mergings

mergings = linkage(movements,method='single')



# Plot the dendrogram

dendrogram(mergings,labels=companies,leaf_rotation=90,leaf_font_size=6)

plt.show()
# Calculate the linkage: mergings

mergings = linkage(normalized_movements,method='single')



# Plot the dendrogram

dendrogram(mergings,labels=companies,leaf_rotation=90,leaf_font_size=6)

plt.show()
from sklearn.manifold import TSNE

model=TSNE(learning_rate=50)

tsne_features= model.fit_transform(normalized_movements)



# Select the 0th feature: xs

xs = tsne_features[:,0]



# Select the 1th feature: ys

ys = tsne_features[:,1]



# Scatter plot

plt.scatter(xs,ys,alpha=0.5)



# Annotate the points

for x, y, company in zip(xs, ys, companies):

    plt.annotate(company, (x, y), fontsize=5, alpha=0.75)

plt.show()

from scipy.stats import pearsonr

# Import PCA

from sklearn.decomposition import PCA



# Create PCA instance: model

model = PCA()



# Apply the fit_transform method of model to grains: pca_features

pca_features = model.fit_transform(movements)



# Assign 0th column of pca_features: xs

xs = pca_features[:,0]



# Assign 1st column of pca_features: ys

ys = pca_features[:,2]





# Scatter plot xs vs ys

plt.scatter(xs, ys)

plt.axis('equal')

plt.show()



# Calculate the Pearson correlation of xs and ys

correlation, pvalue = pearsonr(xs, ys)



# Display the correlation

print(correlation)

print(model.n_components_)
