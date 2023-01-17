import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Load data
df = pd.read_csv('../input/Mall_Customers.csv')
df.head()
# Are there any null values?
df.isna().any()
# Convert gender to 1 and 0
df['Gender'].replace({'Male':0,'Female':1},inplace=True)
df.drop(['CustomerID'],axis=1,inplace=True) # We dont need Customer ID
df.head()
df.shape # How many data points?
# Any features strongly correlated?
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


for col in df.columns:
    X = df.drop([col], axis=1)
    y = df[col]
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42)
    reg = DecisionTreeRegressor()
    reg.fit(X_train,y_train)
    print('Score for {} as dependent variable is {}'.format(col,reg.score(X_test,y_test)))
# Visualize the feature correlation using scatter plot
# Are there any outliers?
pd.scatter_matrix(df, figsize=(16,9), diagonal ='kde')
# Visualize the feature correlation using heatmap
import seaborn as sns
sns.heatmap(df.corr(),xticklabels=df.columns,yticklabels=df.columns)
# Find the principal components!
from sklearn.decomposition import PCA
pca = PCA(n_components=4)
pca.fit(df)
pca.explained_variance_ratio_
pca.components_
dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]

components = pd.DataFrame(pca.components_,columns=df.columns)
components.index = dimensions

variance = pd.DataFrame(pca.explained_variance_ratio_, columns=['Explained Variance'])
variance.index = dimensions

pd.concat([variance,components], axis=1)
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(16,9))
components.plot(kind='bar', ax=ax)
ax.set_xticklabels(dimensions)
for i,variance in enumerate(pca.explained_variance_ratio_):
    ax.text(i,ax.get_ylim()[1]+0.05,'Explained variance {}'.format(np.round(variance,4)))
plt.show()
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(df)
pca.explained_variance_ratio_
transformed_data = pca.transform(df)
transformed_data = pd.DataFrame(transformed_data,columns=['Dimension 1','Dimension 2'])
transformed_data.head()
# Use silhouette score to find the ideal number of clusters.
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

no_of_clusters= range(2,11)
kmeans = [KMeans(n_clusters=i) for i in no_of_clusters]
score = [silhouette_score(transformed_data,kmeans[i].fit(transformed_data).predict(transformed_data),metric='euclidean') for i in range(len(kmeans))]
plt.plot(no_of_clusters,score)
plt.xlabel('No of clusters')
plt.ylabel('Silhouette Score')
plt.show()
kmeans = KMeans(n_clusters=5)
kmeans.fit(transformed_data)
kmeans.predict(transformed_data)

plt.scatter(transformed_data.iloc[:,0],transformed_data.iloc[:,1],c=kmeans.labels_,cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],color='black')
plt.show()
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=5)
gmm.fit(transformed_data)
labels = gmm.predict(transformed_data)

plt.scatter(transformed_data.iloc[:,0],transformed_data.iloc[:,1],c=labels,cmap='rainbow')
plt.show()
cluster_proba_df = pd.DataFrame(gmm.predict_proba(transformed_data), columns = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5'])
cluster_proba_df['Belongs to'] = cluster_proba_df.idxmax(axis=1)
cluster_proba_df.head()