import warnings

warnings.filterwarnings("ignore")

import numpy as np

import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report

from sklearn.cluster import DBSCAN

from sklearn.datasets.samples_generator import make_blobs

from sklearn.neighbors import NearestNeighbors

import seaborn as sns

import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d

%matplotlib inline
cluster_df = pd.read_csv('../input/cluster_table.csv')

cluster_df.head(10)
print('The shape of the dataframe is - {}'.format(cluster_df.shape))

print('The number of entries - {}'.format(cluster_df.size))

print('-'*60)

print('The basic statistics -\n{}'.format(cluster_df.describe()))

print('-'*60)

print('Unique values per column -\n{}'.format(cluster_df.nunique()))

print('-'*60)

print('Table Info -')

print(cluster_df.info())
sns.pairplot(cluster_df, hue='Cluster label')

plt.show()
sns.distplot(cluster_df.X, label='X values')

sns.distplot(cluster_df.Y, label='Y values')

sns.distplot(cluster_df.Z, label='Z values')

plt.legend()

plt.show()
fig, axes = plt.subplots(2,3, figsize=(10,5))

axes = axes.flatten()  # Converting multidimensional "axes" array into a single dimensional array.

# Get the boxplots of features X,Y and Z

ax = sns.boxplot('X', data=cluster_df, orient='h', ax=axes[0])

ax = sns.boxplot('Y', data=cluster_df, orient='h', ax=axes[1])

ax = sns.boxplot('Z', data=cluster_df, orient='h', ax=axes[2])

# Get the boxplots of features X,Y and Z grouped by 'Cluster label'

ax = sns.boxplot(y='Cluster label', x='X', data=cluster_df, orient='h', ax=axes[3])

ax = sns.boxplot(y='Cluster label', x='Y', data=cluster_df, orient='h', ax=axes[4])

ax = sns.boxplot(y='Cluster label', x='Z', data=cluster_df, orient='h', ax=axes[5])

fig.tight_layout()     # Automatically adjust subplot parameters to give specified padding
X = cluster_df[['X', 'Y', 'Z']]

Y_target = cluster_df['Cluster label']

scaler = StandardScaler()

scaled_X = scaler.fit_transform(X)
# Storing the distance to the 6th nearest neaighbor to the distance variable

neigh = NearestNeighbors(n_neighbors=6)   # Classifier implementing the 6-nearest neighbors vote

nbrs = neigh.fit(scaled_X)

distances, indices = nbrs.kneighbors(scaled_X)

distances = np.sort(distances, axis=0)

distances = distances[:,5]

# Generate the elbow plot

plt.figure(figsize=(10,6))

plt.plot(distances)

plt.title('The Elbow Plot')

plt.xlabel('Core Point')

plt.ylabel('Avg distance from neighbors')

plt.yticks(np.arange(0.2, 1.6, 0.1))      # Increase density of yticks to help meausuring elbow values visually

plt.grid(axis='y')                        # Display gridlines to help meausuring elbow values visually

plt.show()
# Fit model and get labels

dbscan = DBSCAN(eps=0.54, min_samples=6)

dbscan.fit(scaled_X)

label = dbscan.labels_

# Plot a confusion matrix

df = pd.DataFrame({'Actual': Y_target, 'Predicted': label+1})

ct = pd.crosstab(df['Actual'], df['Predicted'])

sns.heatmap(ct, annot=True, cmap="YlGnBu")

plt.show()

# Display the classification report

print('\n', '*'*30, 'Classification Report', '*'*30, '\n')

print(classification_report(df['Actual'], df['Predicted']))
# Creating separate dataframes based on predicted groups or clusters

outliers = cluster_df.iloc[np.where(label==-1)]  # Only contains samples from predicted outlier class

class1 = cluster_df.iloc[np.where(label==0)]     # Only contains samples from predicted class 1

class2 = cluster_df.iloc[np.where(label==1)]     # Only contains samples from predicted class 2

class3 = cluster_df.iloc[np.where(label==2)]     # Only contains samples from predicted class 3

class4 = cluster_df.iloc[np.where(label==3)]     # Only contains samples from predicted class 4

# Generating the 3D figure

fig = plt.figure(figsize=(9,7))

ax = plt.axes(projection='3d')

ax.scatter3D(class1.X, class1.Y, class1.Z, s=50, c='red', label='Cluster 1')

ax.scatter3D(class2.X, class2.Y, class2.Z, s=50, c='blue', label='Cluster 2')

ax.scatter3D(class3.X, class3.Y, class3.Z, s=50, c='green', label='Cluster 3')

ax.scatter3D(class4.X, class4.Y, class4.Z, s=50, c='orange', label='Cluster 4')

ax.scatter3D(outliers.X, outliers.Y, outliers.Z, s=50, c='violet', label='Outliers')

ax.legend()

plt.show()