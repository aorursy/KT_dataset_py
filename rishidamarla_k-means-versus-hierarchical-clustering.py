# Importing all necessary libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly

import plotly.graph_objects as go
df = pd.read_csv('../input/nba-players-data/all_seasons.csv', index_col=0)
df.head()
df.info()
df.describe()
# Removing Non-integer features.

df2 = df.drop(['team_abbreviation', 'college', 'country', 'draft_year', 'draft_round', 'draft_number', 'season'], axis = 1) 
df2.info()
# Plotting players on a 2D graph - Height vs Weight.

fig = go.Figure(data=go.Scatter(x=df2['player_weight'],

                                y=df2['player_height'],

                                mode='markers',

                                text=df2['player_name'],

                                marker=dict(color='#17408b')

                                ))



fig.update_layout(

    title='NBA Player Height and Weight (for interactive exploration)',

    xaxis_title='Weight (kg)',

    yaxis_title='Height (cm)',

    plot_bgcolor='rgba(0,0,0,0)'

)

fig.show()
# Finding the line of best fit.

plt.figure(figsize=(16, 8))



sns.regplot(x='player_weight', y='player_height', data=df2, color='#17408b')



plt.title('Relationship Between Player Height and Weight', fontsize=16)

plt.ylabel('Height (cm)')

plt.xlabel('Weight (kg)')

sns.despine()



plt.show()
df3 = df2.drop(['player_name'], axis = 1) 
X = df3.values

# Using the standard scaler method to standardize all of the features by converting them into values between -3 and +3.

from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(X)
X
# Using Principal Component Analysis or PCA in short to reduce the dimensionality of the data in order to optimize the result of the clustering.

from sklearn.decomposition import PCA

pca = PCA(n_components=2)

principalComponents1 = pca.fit_transform(X)
principalComponents1
PCA_dataset1 = pd.DataFrame(data = principalComponents1, columns = ['component1', 'component2'] )

PCA_dataset1.head()
principal_component1 = PCA_dataset1['component1']

principal_component2 = PCA_dataset1['component2']
plt.figure()

plt.figure(figsize=(10,10))

plt.xlabel('Component 1')

plt.ylabel('Component 2')

plt.title('2 Component PCA')

plt.scatter(PCA_dataset1['component1'], PCA_dataset1['component2']) #c = y_kmeans, s=10)
# Implementing the K Means Clustering Algorithm and specifying the number of clusters needed.

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 80, init = 'k-means++', random_state = 1)

y_kmeans = kmeans.fit_predict(principalComponents1)
from matplotlib import colors as mcolors
# Plotting the clusters.

plt.scatter(principalComponents1[y_kmeans == 0, 0], principalComponents1[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')

plt.scatter(principalComponents1[y_kmeans == 1, 0], principalComponents1[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')

plt.scatter(principalComponents1[y_kmeans == 2, 0], principalComponents1[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')

plt.scatter(principalComponents1[y_kmeans == 3, 0], principalComponents1[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')

plt.scatter(principalComponents1[y_kmeans == 4, 0], principalComponents1[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')

plt.scatter(principalComponents1[y_kmeans == 5, 0], principalComponents1[y_kmeans == 5, 1], s = 100, c = 'limegreen', label = 'Cluster 6')

plt.scatter(principalComponents1[y_kmeans == 6, 0], principalComponents1[y_kmeans == 6, 1], s = 100, c = 'lavender', label = 'Cluster 7')

plt.scatter(principalComponents1[y_kmeans == 7, 0], principalComponents1[y_kmeans == 7, 1], s = 100, c = 'black', label = 'Cluster 8')

plt.scatter(principalComponents1[y_kmeans == 8, 0], principalComponents1[y_kmeans == 8, 1], s = 100, c = 'dimgray', label = 'Cluster 9')

plt.scatter(principalComponents1[y_kmeans == 9, 0], principalComponents1[y_kmeans == 9, 1], s = 100, c = 'silver', label = 'Cluster 10')

plt.scatter(principalComponents1[y_kmeans == 10, 0], principalComponents1[y_kmeans == 10, 1], s = 100, c = 'gainsboro', label = 'Cluster 11')

plt.scatter(principalComponents1[y_kmeans == 11, 0], principalComponents1[y_kmeans == 11, 1], s = 100, c = 'white', label = 'Cluster 12')

plt.scatter(principalComponents1[y_kmeans == 12, 0], principalComponents1[y_kmeans == 12, 1], s = 100, c = 'whitesmoke', label = 'Cluster 13')

plt.scatter(principalComponents1[y_kmeans == 13, 0], principalComponents1[y_kmeans == 13, 1], s = 100, c = 'rosybrown', label = 'Cluster 14')

plt.scatter(principalComponents1[y_kmeans == 14, 0], principalComponents1[y_kmeans == 14, 1], s = 100, c = 'indianred', label = 'Cluster 15')

plt.scatter(principalComponents1[y_kmeans == 15, 0], principalComponents1[y_kmeans == 15, 1], s = 100, c = 'firebrick', label = 'Cluster 16')

plt.scatter(principalComponents1[y_kmeans == 16, 0], principalComponents1[y_kmeans == 16, 1], s = 100, c = 'red', label = 'Cluster 17')

plt.scatter(principalComponents1[y_kmeans == 17, 0], principalComponents1[y_kmeans == 17, 1], s = 100, c = 'mistyrose', label = 'Cluster 18')

plt.scatter(principalComponents1[y_kmeans == 18, 0], principalComponents1[y_kmeans == 18, 1], s = 100, c = 'salmon', label = 'Cluster 19')

plt.scatter(principalComponents1[y_kmeans == 19, 0], principalComponents1[y_kmeans == 19, 1], s = 100, c = 'darksalmon', label = 'Cluster 20')

plt.scatter(principalComponents1[y_kmeans == 20, 0], principalComponents1[y_kmeans == 20, 1], s = 100, c = 'coral', label = 'Cluster 21')

plt.scatter(principalComponents1[y_kmeans == 21, 0], principalComponents1[y_kmeans == 21, 1], s = 100, c = 'orangered', label = 'Cluster 22')

plt.scatter(principalComponents1[y_kmeans == 22, 0], principalComponents1[y_kmeans == 22, 1], s = 100, c = 'sienna', label = 'Cluster 23')

plt.scatter(principalComponents1[y_kmeans == 23, 0], principalComponents1[y_kmeans == 23, 1], s = 100, c = 'seashell', label = 'Cluster 24')

plt.scatter(principalComponents1[y_kmeans == 24, 0], principalComponents1[y_kmeans == 24, 1], s = 100, c = 'chocolate', label = 'Cluster 25')

plt.scatter(principalComponents1[y_kmeans == 25, 0], principalComponents1[y_kmeans == 25, 1], s = 100, c = 'saddlebrown', label = 'Cluster 26')

plt.scatter(principalComponents1[y_kmeans == 26, 0], principalComponents1[y_kmeans == 26, 1], s = 100, c = 'sandybrown', label = 'Cluster 27')

plt.scatter(principalComponents1[y_kmeans == 27, 0], principalComponents1[y_kmeans == 27, 1], s = 100, c = 'peachpuff', label = 'Cluster 28')

plt.scatter(principalComponents1[y_kmeans == 28, 0], principalComponents1[y_kmeans == 28, 1], s = 100, c = 'peru', label = 'Cluster 29')

plt.scatter(principalComponents1[y_kmeans == 29, 0], principalComponents1[y_kmeans == 29, 1], s = 100, c = 'bisque', label = 'Cluster 30')

plt.scatter(principalComponents1[y_kmeans == 30, 0], principalComponents1[y_kmeans == 30, 1], s = 100, c = 'linen', label = 'Cluster 31')

plt.scatter(principalComponents1[y_kmeans == 31, 0], principalComponents1[y_kmeans == 31, 1], s = 100, c = 'darkorange', label = 'Cluster 32')

plt.scatter(principalComponents1[y_kmeans == 32, 0], principalComponents1[y_kmeans == 32, 1], s = 100, c = 'burlywood', label = 'Cluster 33')

plt.scatter(principalComponents1[y_kmeans == 33, 0], principalComponents1[y_kmeans == 33, 1], s = 100, c = 'antiquewhite', label = 'Cluster 34')

plt.scatter(principalComponents1[y_kmeans == 34, 0], principalComponents1[y_kmeans == 34, 1], s = 100, c = 'tan', label = 'Cluster 35')

plt.scatter(principalComponents1[y_kmeans == 35, 0], principalComponents1[y_kmeans == 35, 1], s = 100, c = 'navajowhite', label = 'Cluster 36')

plt.scatter(principalComponents1[y_kmeans == 36, 0], principalComponents1[y_kmeans == 36, 1], s = 100, c = 'orange', label = 'Cluster 37')

plt.scatter(principalComponents1[y_kmeans == 37, 0], principalComponents1[y_kmeans == 37, 1], s = 100, c = 'oldlace', label = 'Cluster 38')

plt.scatter(principalComponents1[y_kmeans == 38, 0], principalComponents1[y_kmeans == 38, 1], s = 100, c = 'darkgoldenrod', label = 'Cluster 39')

plt.scatter(principalComponents1[y_kmeans == 39, 0], principalComponents1[y_kmeans == 39, 1], s = 100, c = 'goldenrod', label = 'Cluster 40')

plt.scatter(principalComponents1[y_kmeans == 40, 0], principalComponents1[y_kmeans == 40, 1], s = 100, c = 'gold', label = 'Cluster 41')

plt.scatter(principalComponents1[y_kmeans == 41, 0], principalComponents1[y_kmeans == 41, 1], s = 100, c = 'khaki', label = 'Cluster 42')

plt.scatter(principalComponents1[y_kmeans == 42, 0], principalComponents1[y_kmeans == 42, 1], s = 100, c = 'darkkhaki', label = 'Cluster 43')

plt.scatter(principalComponents1[y_kmeans == 43, 0], principalComponents1[y_kmeans == 43, 1], s = 100, c = 'ivory', label = 'Cluster 44')

plt.scatter(principalComponents1[y_kmeans == 44, 0], principalComponents1[y_kmeans == 44, 1], s = 100, c = 'beige', label = 'Cluster 45')

plt.scatter(principalComponents1[y_kmeans == 45, 0], principalComponents1[y_kmeans == 45, 1], s = 100, c = 'olive', label = 'Cluster 46')

plt.scatter(principalComponents1[y_kmeans == 46, 0], principalComponents1[y_kmeans == 46, 1], s = 100, c = 'y', label = 'Cluster 47')

plt.scatter(principalComponents1[y_kmeans == 47, 0], principalComponents1[y_kmeans == 47, 1], s = 100, c = 'olivedrab', label = 'Cluster 48')

plt.scatter(principalComponents1[y_kmeans == 48, 0], principalComponents1[y_kmeans == 48, 1], s = 100, c = 'yellowgreen', label = 'Cluster 49')

plt.scatter(principalComponents1[y_kmeans == 49, 0], principalComponents1[y_kmeans == 49, 1], s = 100, c = 'darkolivegreen', label = 'Cluster 50')

plt.scatter(principalComponents1[y_kmeans == 50, 0], principalComponents1[y_kmeans == 50, 1], s = 100, c = 'greenyellow', label = 'Cluster 51')

plt.scatter(principalComponents1[y_kmeans == 51, 0], principalComponents1[y_kmeans == 51, 1], s = 100, c = 'chartreuse', label = 'Cluster 52')

plt.scatter(principalComponents1[y_kmeans == 52, 0], principalComponents1[y_kmeans == 52, 1], s = 100, c = 'blanchedalmond', label = 'Cluster 53')

plt.scatter(principalComponents1[y_kmeans == 53, 0], principalComponents1[y_kmeans == 53, 1], s = 100, c = 'darkseagreen', label = 'Cluster 54')

plt.scatter(principalComponents1[y_kmeans == 54, 0], principalComponents1[y_kmeans == 54, 1], s = 100, c = 'palegreen', label = 'Cluster 55')

plt.scatter(principalComponents1[y_kmeans == 55, 0], principalComponents1[y_kmeans == 55, 1], s = 100, c = 'forestgreen', label = 'Cluster 56')

plt.scatter(principalComponents1[y_kmeans == 56, 0], principalComponents1[y_kmeans == 56, 1], s = 100, c = 'seagreen', label = 'Cluster 57')

plt.scatter(principalComponents1[y_kmeans == 57, 0], principalComponents1[y_kmeans == 57, 1], s = 100, c = 'mediumseagreen', label = 'Cluster 58')

plt.scatter(principalComponents1[y_kmeans == 58, 0], principalComponents1[y_kmeans == 58, 1], s = 100, c = 'springgreen', label = 'Cluster 59')

plt.scatter(principalComponents1[y_kmeans == 59, 0], principalComponents1[y_kmeans == 59, 1], s = 100, c = 'mintcream', label = 'Cluster 60')

plt.scatter(principalComponents1[y_kmeans == 60, 0], principalComponents1[y_kmeans == 60, 1], s = 100, c = 'mediumaquamarine', label = 'Cluster 61')

plt.scatter(principalComponents1[y_kmeans == 61, 0], principalComponents1[y_kmeans == 61, 1], s = 100, c = 'aquamarine', label = 'Cluster 62')

plt.scatter(principalComponents1[y_kmeans == 62, 0], principalComponents1[y_kmeans == 62, 1], s = 100, c = 'turquoise', label = 'Cluster 63')

plt.scatter(principalComponents1[y_kmeans == 63, 0], principalComponents1[y_kmeans == 63, 1], s = 100, c = 'lightseagreen', label = 'Cluster 64')

plt.scatter(principalComponents1[y_kmeans == 64, 0], principalComponents1[y_kmeans == 64, 1], s = 100, c = 'azure', label = 'Cluster 65')

plt.scatter(principalComponents1[y_kmeans == 65, 0], principalComponents1[y_kmeans == 65, 1], s = 100, c = 'paleturquoise', label = 'Cluster 66')

plt.scatter(principalComponents1[y_kmeans == 66, 0], principalComponents1[y_kmeans == 66, 1], s = 100, c = 'darkslategray', label = 'Cluster 67')

plt.scatter(principalComponents1[y_kmeans == 67, 0], principalComponents1[y_kmeans == 67, 1], s = 100, c = 'teal', label = 'Cluster 68')

plt.scatter(principalComponents1[y_kmeans == 68, 0], principalComponents1[y_kmeans == 68, 1], s = 100, c = 'c', label = 'Cluster 69')

plt.scatter(principalComponents1[y_kmeans == 69, 0], principalComponents1[y_kmeans == 69, 1], s = 100, c = 'cyan', label = 'Cluster 70')

plt.scatter(principalComponents1[y_kmeans == 70, 0], principalComponents1[y_kmeans == 70, 1], s = 100, c = 'darkturquoise', label = 'Cluster 71')

plt.scatter(principalComponents1[y_kmeans == 71, 0], principalComponents1[y_kmeans == 71, 1], s = 100, c = 'cadetblue', label = 'Cluster 72')

plt.scatter(principalComponents1[y_kmeans == 72, 0], principalComponents1[y_kmeans == 72, 1], s = 100, c = 'powderblue', label = 'Cluster 73')

plt.scatter(principalComponents1[y_kmeans == 73, 0], principalComponents1[y_kmeans == 73, 1], s = 100, c = 'deepskyblue', label = 'Cluster 74')

plt.scatter(principalComponents1[y_kmeans == 74, 0], principalComponents1[y_kmeans == 74, 1], s = 100, c = 'steelblue', label = 'Cluster 75')

plt.scatter(principalComponents1[y_kmeans == 75, 0], principalComponents1[y_kmeans == 75, 1], s = 100, c = 'aliceblue', label = 'Cluster 76')

plt.scatter(principalComponents1[y_kmeans == 76, 0], principalComponents1[y_kmeans == 76, 1], s = 100, c = 'dodgerblue', label = 'Cluster 77')

plt.scatter(principalComponents1[y_kmeans == 77, 0], principalComponents1[y_kmeans == 77, 1], s = 100, c = 'slategrey', label = 'Cluster 78')

plt.scatter(principalComponents1[y_kmeans == 78, 0], principalComponents1[y_kmeans == 78, 1], s = 100, c = 'lightsteelblue', label = 'Cluster 79')

plt.scatter(principalComponents1[y_kmeans == 79, 0], principalComponents1[y_kmeans == 79, 1], s = 100, c = 'cornflowerblue', label = 'Cluster 80')

plt.scatter(principalComponents1[y_kmeans == 80, 0], principalComponents1[y_kmeans == 80, 1], s = 100, c = 'navy', label = 'Cluster 81')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 25, c = 'yellow', label = 'Centroids')

plt.title('Clusters of NBA Players')

plt.xlabel('Principal Component 1')

plt.ylabel('Principal Component 2')

plt.show()
# Implementing a dendogram to visualize the euclidean distanced between each compound.

import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(principalComponents1, method = 'ward'))

plt.title('Dendrogram')

plt.xlabel('Compounds')

plt.ylabel('Euclidean distances')

plt.show()
# Implementing the Hierachical Clustering.

from sklearn.cluster import AgglomerativeClustering

hc2 = AgglomerativeClustering(n_clusters = 80, affinity = 'euclidean', linkage = 'ward')

y_hc2 = hc2.fit_predict(principalComponents1)
# Plotting the clusters.

plt.scatter(principalComponents1[y_hc2 == 0, 0], principalComponents1[y_hc2 == 0, 1], s = 100, c = 'red', label = 'Cluster 1')

plt.scatter(principalComponents1[y_hc2 == 1, 0], principalComponents1[y_hc2 == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')

plt.scatter(principalComponents1[y_hc2 == 2, 0], principalComponents1[y_hc2 == 2, 1], s = 100, c = 'green', label = 'Cluster 3')

plt.scatter(principalComponents1[y_hc2 == 3, 0], principalComponents1[y_hc2 == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')

plt.scatter(principalComponents1[y_hc2 == 4, 0], principalComponents1[y_hc2 == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')

plt.scatter(principalComponents1[y_hc2 == 5, 0], principalComponents1[y_hc2 == 5, 1], s = 100, c = 'limegreen', label = 'Cluster 6')

plt.scatter(principalComponents1[y_hc2 == 6, 0], principalComponents1[y_hc2 == 6, 1], s = 100, c = 'lavender', label = 'Cluster 7')

plt.scatter(principalComponents1[y_hc2 == 7, 0], principalComponents1[y_hc2 == 7, 1], s = 100, c = 'black', label = 'Cluster 8')

plt.scatter(principalComponents1[y_hc2 == 8, 0], principalComponents1[y_hc2 == 8, 1], s = 100, c = 'dimgray', label = 'Cluster 9')

plt.scatter(principalComponents1[y_hc2 == 9, 0], principalComponents1[y_hc2 == 9, 1], s = 100, c = 'silver', label = 'Cluster 10')

plt.scatter(principalComponents1[y_hc2 == 10, 0], principalComponents1[y_hc2 == 10, 1], s = 100, c = 'gainsboro', label = 'Cluster 11')

plt.scatter(principalComponents1[y_hc2 == 11, 0], principalComponents1[y_hc2 == 11, 1], s = 100, c = 'white', label = 'Cluster 12')

plt.scatter(principalComponents1[y_hc2 == 12, 0], principalComponents1[y_hc2 == 12, 1], s = 100, c = 'whitesmoke', label = 'Cluster 13')

plt.scatter(principalComponents1[y_hc2 == 13, 0], principalComponents1[y_hc2 == 13, 1], s = 100, c = 'rosybrown', label = 'Cluster 14')

plt.scatter(principalComponents1[y_hc2 == 14, 0], principalComponents1[y_hc2 == 14, 1], s = 100, c = 'indianred', label = 'Cluster 15')

plt.scatter(principalComponents1[y_hc2 == 15, 0], principalComponents1[y_hc2 == 15, 1], s = 100, c = 'firebrick', label = 'Cluster 16')

plt.scatter(principalComponents1[y_hc2 == 16, 0], principalComponents1[y_hc2 == 16, 1], s = 100, c = 'red', label = 'Cluster 17')

plt.scatter(principalComponents1[y_hc2 == 17, 0], principalComponents1[y_hc2 == 17, 1], s = 100, c = 'mistyrose', label = 'Cluster 18')

plt.scatter(principalComponents1[y_hc2 == 18, 0], principalComponents1[y_hc2 == 18, 1], s = 100, c = 'salmon', label = 'Cluster 19')

plt.scatter(principalComponents1[y_hc2 == 19, 0], principalComponents1[y_hc2 == 19, 1], s = 100, c = 'darksalmon', label = 'Cluster 20')

plt.scatter(principalComponents1[y_hc2 == 20, 0], principalComponents1[y_hc2 == 20, 1], s = 100, c = 'coral', label = 'Cluster 21')

plt.scatter(principalComponents1[y_hc2 == 21, 0], principalComponents1[y_hc2 == 21, 1], s = 100, c = 'orangered', label = 'Cluster 22')

plt.scatter(principalComponents1[y_hc2 == 22, 0], principalComponents1[y_hc2 == 22, 1], s = 100, c = 'sienna', label = 'Cluster 23')

plt.scatter(principalComponents1[y_hc2 == 23, 0], principalComponents1[y_hc2 == 23, 1], s = 100, c = 'seashell', label = 'Cluster 24')

plt.scatter(principalComponents1[y_hc2 == 24, 0], principalComponents1[y_hc2 == 24, 1], s = 100, c = 'chocolate', label = 'Cluster 25')

plt.scatter(principalComponents1[y_hc2 == 25, 0], principalComponents1[y_hc2 == 25, 1], s = 100, c = 'saddlebrown', label = 'Cluster 26')

plt.scatter(principalComponents1[y_hc2 == 26, 0], principalComponents1[y_hc2 == 26, 1], s = 100, c = 'sandybrown', label = 'Cluster 27')

plt.scatter(principalComponents1[y_hc2 == 27, 0], principalComponents1[y_hc2 == 27, 1], s = 100, c = 'peachpuff', label = 'Cluster 28')

plt.scatter(principalComponents1[y_hc2 == 28, 0], principalComponents1[y_hc2 == 28, 1], s = 100, c = 'peru', label = 'Cluster 29')

plt.scatter(principalComponents1[y_hc2 == 29, 0], principalComponents1[y_hc2 == 29, 1], s = 100, c = 'bisque', label = 'Cluster 30')

plt.scatter(principalComponents1[y_hc2 == 30, 0], principalComponents1[y_hc2 == 30, 1], s = 100, c = 'linen', label = 'Cluster 31')

plt.scatter(principalComponents1[y_hc2 == 31, 0], principalComponents1[y_hc2 == 31, 1], s = 100, c = 'darkorange', label = 'Cluster 32')

plt.scatter(principalComponents1[y_hc2 == 32, 0], principalComponents1[y_hc2 == 32, 1], s = 100, c = 'burlywood', label = 'Cluster 33')

plt.scatter(principalComponents1[y_hc2 == 33, 0], principalComponents1[y_hc2 == 33, 1], s = 100, c = 'antiquewhite', label = 'Cluster 34')

plt.scatter(principalComponents1[y_hc2 == 34, 0], principalComponents1[y_hc2 == 34, 1], s = 100, c = 'tan', label = 'Cluster 35')

plt.scatter(principalComponents1[y_hc2 == 35, 0], principalComponents1[y_hc2 == 35, 1], s = 100, c = 'navajowhite', label = 'Cluster 36')

plt.scatter(principalComponents1[y_hc2 == 36, 0], principalComponents1[y_hc2 == 36, 1], s = 100, c = 'orange', label = 'Cluster 37')

plt.scatter(principalComponents1[y_hc2 == 37, 0], principalComponents1[y_hc2 == 37, 1], s = 100, c = 'oldlace', label = 'Cluster 38')

plt.scatter(principalComponents1[y_hc2 == 38, 0], principalComponents1[y_hc2 == 38, 1], s = 100, c = 'darkgoldenrod', label = 'Cluster 39')

plt.scatter(principalComponents1[y_hc2 == 39, 0], principalComponents1[y_hc2 == 39, 1], s = 100, c = 'goldenrod', label = 'Cluster 40')

plt.scatter(principalComponents1[y_hc2 == 40, 0], principalComponents1[y_hc2 == 40, 1], s = 100, c = 'gold', label = 'Cluster 41')

plt.scatter(principalComponents1[y_hc2 == 41, 0], principalComponents1[y_hc2 == 41, 1], s = 100, c = 'khaki', label = 'Cluster 42')

plt.scatter(principalComponents1[y_hc2 == 42, 0], principalComponents1[y_hc2 == 42, 1], s = 100, c = 'darkkhaki', label = 'Cluster 43')

plt.scatter(principalComponents1[y_hc2 == 43, 0], principalComponents1[y_hc2 == 43, 1], s = 100, c = 'ivory', label = 'Cluster 44')

plt.scatter(principalComponents1[y_hc2 == 44, 0], principalComponents1[y_hc2 == 44, 1], s = 100, c = 'beige', label = 'Cluster 45')

plt.scatter(principalComponents1[y_hc2 == 45, 0], principalComponents1[y_hc2 == 45, 1], s = 100, c = 'olive', label = 'Cluster 46')

plt.scatter(principalComponents1[y_hc2 == 46, 0], principalComponents1[y_hc2 == 46, 1], s = 100, c = 'y', label = 'Cluster 47')

plt.scatter(principalComponents1[y_hc2 == 47, 0], principalComponents1[y_hc2 == 47, 1], s = 100, c = 'olivedrab', label = 'Cluster 48')

plt.scatter(principalComponents1[y_hc2 == 48, 0], principalComponents1[y_hc2 == 48, 1], s = 100, c = 'yellowgreen', label = 'Cluster 49')

plt.scatter(principalComponents1[y_hc2 == 49, 0], principalComponents1[y_hc2 == 49, 1], s = 100, c = 'darkolivegreen', label = 'Cluster 50')

plt.scatter(principalComponents1[y_hc2 == 50, 0], principalComponents1[y_hc2 == 50, 1], s = 100, c = 'greenyellow', label = 'Cluster 51')

plt.scatter(principalComponents1[y_hc2 == 51, 0], principalComponents1[y_hc2 == 51, 1], s = 100, c = 'chartreuse', label = 'Cluster 52')

plt.scatter(principalComponents1[y_hc2 == 52, 0], principalComponents1[y_hc2 == 52, 1], s = 100, c = 'blanchedalmond', label = 'Cluster 53')

plt.scatter(principalComponents1[y_hc2 == 53, 0], principalComponents1[y_hc2 == 53, 1], s = 100, c = 'darkseagreen', label = 'Cluster 54')

plt.scatter(principalComponents1[y_hc2 == 54, 0], principalComponents1[y_hc2 == 54, 1], s = 100, c = 'palegreen', label = 'Cluster 55')

plt.scatter(principalComponents1[y_hc2 == 55, 0], principalComponents1[y_hc2 == 55, 1], s = 100, c = 'forestgreen', label = 'Cluster 56')

plt.scatter(principalComponents1[y_hc2 == 56, 0], principalComponents1[y_hc2 == 56, 1], s = 100, c = 'seagreen', label = 'Cluster 57')

plt.scatter(principalComponents1[y_hc2 == 57, 0], principalComponents1[y_hc2 == 57, 1], s = 100, c = 'mediumseagreen', label = 'Cluster 58')

plt.scatter(principalComponents1[y_hc2 == 58, 0], principalComponents1[y_hc2 == 58, 1], s = 100, c = 'springgreen', label = 'Cluster 59')

plt.scatter(principalComponents1[y_hc2 == 59, 0], principalComponents1[y_hc2 == 59, 1], s = 100, c = 'mintcream', label = 'Cluster 60')

plt.scatter(principalComponents1[y_hc2 == 60, 0], principalComponents1[y_hc2 == 60, 1], s = 100, c = 'mediumaquamarine', label = 'Cluster 61')

plt.scatter(principalComponents1[y_hc2 == 61, 0], principalComponents1[y_hc2 == 61, 1], s = 100, c = 'aquamarine', label = 'Cluster 62')

plt.scatter(principalComponents1[y_hc2 == 62, 0], principalComponents1[y_hc2 == 62, 1], s = 100, c = 'turquoise', label = 'Cluster 63')

plt.scatter(principalComponents1[y_hc2 == 63, 0], principalComponents1[y_hc2 == 63, 1], s = 100, c = 'lightseagreen', label = 'Cluster 64')

plt.scatter(principalComponents1[y_hc2 == 64, 0], principalComponents1[y_hc2 == 64, 1], s = 100, c = 'azure', label = 'Cluster 65')

plt.scatter(principalComponents1[y_hc2 == 65, 0], principalComponents1[y_hc2 == 65, 1], s = 100, c = 'paleturquoise', label = 'Cluster 66')

plt.scatter(principalComponents1[y_hc2 == 66, 0], principalComponents1[y_hc2 == 66, 1], s = 100, c = 'darkslategray', label = 'Cluster 67')

plt.scatter(principalComponents1[y_hc2 == 67, 0], principalComponents1[y_hc2 == 67, 1], s = 100, c = 'teal', label = 'Cluster 68')

plt.scatter(principalComponents1[y_hc2 == 68, 0], principalComponents1[y_hc2 == 68, 1], s = 100, c = 'c', label = 'Cluster 69')

plt.scatter(principalComponents1[y_hc2 == 69, 0], principalComponents1[y_hc2 == 69, 1], s = 100, c = 'cyan', label = 'Cluster 70')

plt.scatter(principalComponents1[y_hc2 == 70, 0], principalComponents1[y_hc2 == 70, 1], s = 100, c = 'darkturquoise', label = 'Cluster 71')

plt.scatter(principalComponents1[y_hc2 == 71, 0], principalComponents1[y_hc2 == 71, 1], s = 100, c = 'cadetblue', label = 'Cluster 72')

plt.scatter(principalComponents1[y_hc2 == 72, 0], principalComponents1[y_hc2 == 72, 1], s = 100, c = 'powderblue', label = 'Cluster 73')

plt.scatter(principalComponents1[y_hc2 == 73, 0], principalComponents1[y_hc2 == 73, 1], s = 100, c = 'deepskyblue', label = 'Cluster 74')

plt.scatter(principalComponents1[y_hc2 == 74, 0], principalComponents1[y_hc2 == 74, 1], s = 100, c = 'steelblue', label = 'Cluster 75')

plt.scatter(principalComponents1[y_hc2 == 75, 0], principalComponents1[y_hc2 == 75, 1], s = 100, c = 'aliceblue', label = 'Cluster 76')

plt.scatter(principalComponents1[y_hc2 == 76, 0], principalComponents1[y_hc2 == 76, 1], s = 100, c = 'dodgerblue', label = 'Cluster 77')

plt.scatter(principalComponents1[y_hc2 == 77, 0], principalComponents1[y_hc2 == 77, 1], s = 100, c = 'slategrey', label = 'Cluster 78')

plt.scatter(principalComponents1[y_hc2 == 78, 0], principalComponents1[y_hc2 == 78, 1], s = 100, c = 'lightsteelblue', label = 'Cluster 79')

plt.scatter(principalComponents1[y_hc2 == 79, 0], principalComponents1[y_hc2 == 79, 1], s = 100, c = 'cornflowerblue', label = 'Cluster 80')

plt.scatter(principalComponents1[y_hc2 == 80, 0], principalComponents1[y_hc2 == 80, 1], s = 100, c = 'navy', label = 'Cluster 81')

plt.title('Clusters of NBA Players')

plt.xlabel('Principal Component 1')

plt.ylabel('Principal Component 2')

plt.show()
