import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



# Load data 

data = pd.read_csv("../input/breast-cancer-wisconsin-data/data.csv")



# Show dataframe

data
data = data.drop(["id"], axis=1)

data = data.drop(["diagnosis"], axis=1)

data = data.drop(["Unnamed: 32"], axis=1)
import numpy as np



plt.figure(figsize=(14,3))



line1 = np.array([2, 1, 1, 2])

line2 = np.array([1, 2, 2, 1])



plt.subplot(131)

plt.plot(line1, color='royalblue')

plt.plot(line2, color='lightcoral')

plt.title(np.corrcoef(line1, line2)[1,0], fontsize=18)



line1 = np.array([2, 1, 1, 2])

line2 = np.array([1, 2, 2, 3])



plt.subplot(132)

plt.plot(line1, color='royalblue')

plt.plot(line2, color='lightcoral')

plt.title(round(np.corrcoef(line1, line2)[1,0],2), fontsize=18)



line1 = np.array([2, 1, 1, 2])

line2 = 2*line1



plt.subplot(133)

plt.plot(line1, color='royalblue')

plt.plot(line2, color='lightcoral')

plt.title(np.corrcoef(line2, 2*line2)[1,0], fontsize=18);

plt.figure(figsize=(15,10))

correlations = data.corr()

sns.heatmap(round(correlations,2), cmap='RdBu', annot=True, 

            annot_kws={"size": 7}, vmin=-1, vmax=1);
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

from scipy.spatial.distance import squareform



plt.figure(figsize=(12,5))

dissimilarity = 1 - abs(correlations)

Z = linkage(squareform(dissimilarity), 'complete')



dendrogram(Z, labels=data.columns, orientation='top', 

           leaf_rotation=90);
# Clusterize the data

threshold = 0.8

labels = fcluster(Z, threshold, criterion='distance')



# Show the cluster

labels
import numpy as np



# Keep the indices to sort labels

labels_order = np.argsort(labels)



# Build a new dataframe with the sorted columns

for idx, i in enumerate(data.columns[labels_order]):

    if idx == 0:

        clustered = pd.DataFrame(data[i])

    else:

        df_to_append = pd.DataFrame(data[i])

        clustered = pd.concat([clustered, df_to_append], axis=1)
plt.figure(figsize=(15,10))

correlations = clustered.corr()

sns.heatmap(round(correlations,2), cmap='RdBu', annot=True, 

            annot_kws={"size": 7}, vmin=-1, vmax=1);
plt.figure(figsize=(15,10))



for idx, t in enumerate(np.arange(0.2,1.1,0.1)):

    

    # Subplot idx + 1

    plt.subplot(3, 3, idx+1)

    

    # Calculate the cluster

    labels = fcluster(Z, t, criterion='distance')



    # Keep the indices to sort labels

    labels_order = np.argsort(labels)



    # Build a new dataframe with the sorted columns

    for idx, i in enumerate(data.columns[labels_order]):

        if idx == 0:

            clustered = pd.DataFrame(data[i])

        else:

            df_to_append = pd.DataFrame(data[i])

            clustered = pd.concat([clustered, df_to_append], axis=1)

            

    # Plot the correlation heatmap

    correlations = clustered.corr()

    sns.heatmap(round(correlations,2), cmap='RdBu', vmin=-1, vmax=1, 

                xticklabels=False, yticklabels=False)

    plt.title("Threshold = {}".format(round(t,2)))
sns.clustermap(correlations, method="complete", cmap='RdBu', annot=True, 

               annot_kws={"size": 7}, vmin=-1, vmax=1, figsize=(15,12));