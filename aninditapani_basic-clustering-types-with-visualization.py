# Load csv into data frame
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_csv('../input/Seed_Data.csv')
df.head() # first 5 rows
df.shape #(rows, columns)
# 210 data points
df.describe()
#  Are the features strongly related ? 
# To know this, take each column as dependent variable and try to predict this column
# from other columns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
cols = df.columns

for col in cols:
    X = df.drop([col], axis=1)
    y = pd.DataFrame(df.loc[:, col])
    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42)
    reg = LinearRegression()
    reg.fit(X_train,y_train)
    score = reg.score(X_test,y_test)
    print('Score for {} as dependent variable is {}'.format(col,score))
# Visualize how each feature is related to another feature
pd.scatter_matrix(df, diagonal='kde', figsize=(16,9))
df_A = df[['A','A_Coef']]
df_A.head()
import matplotlib.pyplot as plt
df_A.plot('A','A_Coef',kind='scatter',figsize=(7,5))
from sklearn.cluster import KMeans

no_of_clusters = range(1, 10)
kmeans = [KMeans(n_clusters=i) for i in no_of_clusters]
score = [kmeans[i].fit(df_A).score(df_A) for i in range(len(kmeans))]
plt.plot(no_of_clusters,score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(df_A)
cluster_labels = kmeans.predict(df_A)
kmeans.cluster_centers_
# 3 cluster centres
kmeans.labels_ 
# 0, 1, 2
plt.figure(figsize=(7,5))
plt.scatter(df_A['A'],df_A['A_Coef'],c=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], color='red')
plt.title('3 Means Clustering')
plt.xlabel('A')
plt.ylabel('A_Coef')
plt.show()
from sklearn.cluster import AgglomerativeClustering
hac_clustering = AgglomerativeClustering(n_clusters=3).fit(df_A)
hac_clustering
plt.figure(figsize=(7,5))
plt.scatter(df_A['A'],df_A['A_Coef'],c=hac_clustering.labels_)
plt.title('3 Means Clustering')
plt.xlabel('A')
plt.ylabel('A_Coef')
plt.show()
df.plot.scatter('WK','LKG')
df_LKG_WK = df[['LKG','WK']]
no_of_clusters=range(1,10)
kmeans = [KMeans(n_clusters=i) for i in no_of_clusters]
score = [kmeans[i].fit(df_LKG_WK).score(df_LKG_WK) for i in range(len(kmeans))]
score
plt.plot(no_of_clusters, score)
kmeans = KMeans(n_clusters=2,random_state=42)
kmeans.fit(df_LKG_WK)
kmeans.predict(df_LKG_WK)
kmeans.cluster_centers_
plt.scatter(df_LKG_WK.iloc[:,0],df_LKG_WK.iloc[:,1],c=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], color='red')
plt.title('2 Means Clustering')
plt.xlabel('WK')
plt.ylabel('LKG')
plt.show()
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=2).fit(df_LKG_WK)
gmm_labels = gmm.predict(df_LKG_WK)
plt.scatter(df_LKG_WK.iloc[:,0],df_LKG_WK.iloc[:,1],c=gmm_labels)
plt.title('2 Means Clustering')
plt.xlabel('WK')
plt.ylabel('LKG')
plt.show()
# As we know that this soft clustering, we can find the probability
# with which each data point belongs to the two clusters
y_pred = gmm.predict_proba(df_LKG_WK)
y_pred[50] # Probability that that data point at row 50 belongs to cluster 1 is 0.99674903 and cluster 2 is 0.00325097
y_pred[100] #cluster 1 - 0.01608574, cluster 2 - 0.98391426