import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
data = pd.read_csv("../input/CC GENERAL.csv")
data.head()
data.info()
data.describe()
data.isna().sum()
data = data.fillna(data.mean())
data.isna().sum()
data.drop('CUST_ID', axis=1, inplace=True)
data.head(2)
data.dtypes
data.nunique()
data[['CASH_ADVANCE_TRX','PURCHASES_TRX','TENURE']].nunique()
plt.figure(figsize=(12,12))
sns.heatmap(data.corr(),xticklabels=data.columns, yticklabels=data.columns, annot=True)
sns.pairplot(data)
fig, axes = plt.subplots(ncols=1, nrows=3)
ax0,ax1,ax2 = axes.flatten()

ax0.hist(data['CASH_ADVANCE_TRX'],65,histtype='bar', stacked=True)
ax0.set_title('CASH_ADVANCE_TRX')

ax1.hist(data['PURCHASES_TRX'], 173, histtype='bar', stacked=True)
ax1.set_title('PURCHASES_TRX')

ax2.hist(data['TENURE'],7,histtype='bar', stacked=True)
ax2.set_title('TENURE')

fig.tight_layout()
features = data.copy()
list(features)
cols = ['BALANCE',
        'PURCHASES',
        'ONEOFF_PURCHASES',
        'INSTALLMENTS_PURCHASES',
        'CASH_ADVANCE',
        'CASH_ADVANCE_TRX',
        'PURCHASES_TRX',
        'CREDIT_LIMIT',
        'PAYMENTS',
        'MINIMUM_PAYMENTS']
features[cols] = np.log(1+features[cols])
features.head()
data.head()
features.describe()
# Determining outliers by boxplot
features.boxplot(rot=90, figsize=(30,10))
from sklearn.cluster import KMeans
X = np.array(features)
sumOfSqrdDist = []
K = range(1,15)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans = kmeans.fit(X)
    sumOfSqrdDist.append([k, kmeans.inertia_])
    
plt.plot(K, sumOfSqrdDist, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum of Squared Distances')
plt.title('Elbow method for optimal k')
plt.show();
sumOfSqrdDist
n_cluster = 11
clustering = KMeans(n_clusters=n_cluster, random_state=42)
cluster_labels = clustering.fit_predict(X)

plt.hist(cluster_labels, bins=range(n_cluster+1))
plt.title("# Customers per cluster")
plt.xlabel("Cluster")
plt.ylabel(" # Customers")
plt.show()

features['CLUSTER_INDEX'] = cluster_labels
data['CLUSTER_INDEX'] = cluster_labels
kmeans.cluster_centers_
from scipy.cluster.hierarchy import ward,dendrogram,linkage
np.set_printoptions(precision=4,suppress=True)
distance = linkage(X,'ward')
plt.figure(figsize=(20,10))
plt.title("Hierarchical Clustering Dendogram")
plt.xlabel("Index")
plt.ylabel("Ward's Distance")
dendrogram(distance, leaf_rotation=90, leaf_font_size=9);
plt.axhline(98, c='k')
from scipy.cluster.hierarchy import fcluster

max_d = 97
clusters = fcluster(distance, max_d, criterion='distance')
clusters
k = 11 #K=3
clusters = fcluster(distance, k, criterion='maxclust')

plt.figure(figsize=(10,8))
plt.scatter(X[:,0], X[:,1], c=clusters);
from sklearn.metrics import silhouette_score

sumOfSquaredErrors = []
for k in range(2,30):
    kmeans = KMeans(n_clusters=k).fit(X)
    sumOfSquaredErrors.append([k,silhouette_score(X, kmeans.labels_)])
plt.plot(pd.DataFrame(sumOfSquaredErrors)[0],pd.DataFrame(sumOfSquaredErrors)[1])
data.head()
data['CLUSTER_INDEX'].nunique()
from sklearn.metrics import confusion_matrix,classification_report
print(classification_report(features['CLUSTER_INDEX'], kmeans.labels_))
