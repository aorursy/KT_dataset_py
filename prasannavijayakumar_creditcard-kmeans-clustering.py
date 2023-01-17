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
#import necessary modules

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



sns.set()
#import the dataset

cc= pd.read_csv('/kaggle/input/ccdata/CC GENERAL.csv')

cc.head()
print(cc.shape)
cc.columns
#Look at data using the info() function

cc.info()
#Look at summary statistics of data using the describe() function

cc.describe(include='all')
# Let's get unique values for each category

unique_vals = {

    k: cc[k].unique()

    for k in cc.columns

}



unique_vals
#CUST_ID is a dataset artifact, not something useful for analysis

cc= cc.drop("CUST_ID", axis=1)
cc.isnull().sum()
#CREDIT_LIMIT and MINIMUM_PAYMENTS have some missing value.so fill missing values with median value



cc= cc.fillna(cc.median())



# Checking no more NULLs in the data

all(cc.isna().sum() == 0)
cc.describe(include='all')
#since all the attributes are numerical first we will understand the distributions of the data on each attributes



cc.hist(figsize=(20,15))

plt.title('Data',fontsize=12)

plt.show()
n= len(cc.columns)



plt.figure(figsize=(10,60))

for i in range(n):

    plt.subplot(17,1,i+1)

    sns.boxplot(cc[cc.columns[i]])

    plt.title(cc.columns[i])

plt.tight_layout()
# Create the correlation matrix

corr = cc.corr()

# Generate a mask for the upper triangle

mask = np.triu(np.ones_like(corr, dtype=bool))

plt.figure(figsize=(14,10))

# Add the mask to the heatmap

sns.heatmap(corr, mask=mask, cmap='YlGnBu',center=0, linewidths=1, annot=True, fmt=".2f")

plt.show()
cc.var().sort_values()
from sklearn.preprocessing import StandardScaler



sc= StandardScaler()

cc_scaled= sc.fit_transform(cc)
#checking optimal value of k using elbow method



from sklearn.cluster import KMeans



ks = range(1, 15)

inertias = []

for k in ks:

    # Create a KMeans instance with k clusters: model

    model= KMeans(n_clusters=k)

        # Fit model to samples

    model.fit(cc_scaled)

        # Append the inertia to the list of inertias

    inertias.append(model.inertia_)

    # Plot ks vs inertias

plt.plot(ks, inertias, '-o')

plt.xlabel('number of clusters, k')

plt.ylabel('inertia')

plt.xticks(ks)

plt.show()
clusters_df=pd.DataFrame({'num_clusters':ks,'cluster_errors':inertias})

clusters_df
#choose k = 4 for number of clusters, based on plot above. also after k=4 the slope of the line is almot constant as well.



from sklearn.cluster import KMeans



KM= KMeans(n_clusters=4)

KM.fit(cc_scaled)



KM_labels = KM.fit_predict(cc_scaled)

KM_labels
KM.cluster_centers_.shape
print(KM.inertia_)
cc['cluster_labels'] = KM_labels

cc.head()
plt.figure(figsize=(20,15))

df1= cc[cc.cluster_labels==0]

df2= cc[cc.cluster_labels==1]

df3= cc[cc.cluster_labels==2]

df4= cc[cc.cluster_labels==3]





plt.scatter(df1['PAYMENTS'], df1['PURCHASES'], color='black')

plt.scatter(df2['PAYMENTS'], df2['PURCHASES'], color='orange')

plt.scatter(df3['PAYMENTS'], df3['PURCHASES'], color='purple')

plt.scatter(df4['PAYMENTS'], df4['PURCHASES'], color='blue')



plt.show()
cc['cluster_labels'].value_counts()
cc.groupby('cluster_labels').mean()
for c in cc:

    grid= sns.FacetGrid(cc, col='cluster_labels')

    grid.map(plt.hist, c)
#t-SNE provides great visualizations when the individual samples can be labeled



from sklearn.manifold import TSNE

model = TSNE(learning_rate=200)



# Apply fit_transform to samples: tsne_features

tsne_features = model.fit_transform(cc_scaled)



# Select the 0th feature: xs

xs = tsne_features[:,0]

# Select the 1st feature: ys

ys = tsne_features[:,1]



plt.figure(figsize=(20,15))

# Scatter plot, coloring by variety_numbers

plt.scatter(xs, ys, c=KM_labels)

plt.show()
from scipy.cluster.hierarchy import dendrogram, linkage



#calculate the linkage: mergings

mergings= linkage(cc_scaled, method='ward')



plt.figure(figsize=(20,15))

#Plot the dendrogram, using labels

dendrogram(mergings, labels=KM_labels, p=5, leaf_rotation=90,leaf_font_size=10, truncate_mode='level')



plt.show()
from sklearn.decomposition import PCA



model= PCA()



model.fit_transform(cc_scaled)
# Plot the explained variances

features = range(model.n_components_)

plt.figure(figsize=(20,15))

plt.bar(features, model.explained_variance_)

plt.xlabel('PCA feature')

plt.ylabel('variance')

plt.xticks(features)

plt.show()
pca= PCA(n_components= 2)



pca.fit(cc_scaled)

pca_features=pca.transform(cc_scaled)



print(pca_features.shape)
# Create a dataframe with the two PCA components

pca_df = pd.DataFrame(data=pca_features,columns=['pca1','pca2'])

pca_df.head()
# Concatenate the clusters labels to the dataframe

df = pd.concat([pca_df,pd.DataFrame({'cluster':KM_labels})], axis = 1)

df.head()
plt.figure(figsize=(18,12))

sns.scatterplot(x='pca1', y='pca2', hue='cluster', data=df, palette=['purple','orange','blue','black'])

plt.xlabel('Principal Component 1', fontsize=13)

plt.ylabel('Principal Component 2', fontsize=13)

plt.show()