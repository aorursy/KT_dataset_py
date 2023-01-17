import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")
warnings.simplefilter('ignore')
data = pd.read_csv("../input/ccdata/CC GENERAL.csv")
data.head()
#checking for Null Values 
data.info()
data.isnull().sum()
CREDIT_LIMIT_mean=data["CREDIT_LIMIT"].mean()

data["CREDIT_LIMIT"].fillna(CREDIT_LIMIT_mean,inplace=True)
MINIMUM_PAYMENTS_mode=data["MINIMUM_PAYMENTS"].mode()[0]
data["MINIMUM_PAYMENTS"].fillna(MINIMUM_PAYMENTS_mode,inplace=True)
# Rechecking again for null Values

data.isnull().sum()
X = data.iloc[:,2:17].values
X
# Modeling using kmeans
from sklearn.cluster import KMeans
wcss =[]
for i in range (1,11):
    kmeans = KMeans(n_clusters = i, max_iter =300)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
# Now i defined that the optimal number of clusters is 6 (K=6)
kmeans=KMeans(n_clusters= 6)
Y_Kmeans = kmeans.fit_predict(X)

plt.scatter(X[Y_Kmeans == 0, 0], X[Y_Kmeans == 0,1],s = 100, c='red', label = 'Cluster 1')

plt.scatter(X[Y_Kmeans == 1, 0], X[Y_Kmeans == 1,1],s = 100, c='blue', label = 'Cluster 2')

plt.scatter(X[Y_Kmeans == 2, 0], X[Y_Kmeans == 2,1],s = 100, c='green', label = 'Cluster 3')
plt.scatter(X[Y_Kmeans == 3, 0], X[Y_Kmeans == 3,1],s = 100, c='orange', label = 'Cluster 4')
plt.scatter(X[Y_Kmeans == 4, 0], X[Y_Kmeans == 4,1],s = 100, c='brown', label = 'Cluster 5')
plt.scatter(X[Y_Kmeans == 5, 0], X[Y_Kmeans == 5,1],s = 100, c='pink', label = 'Cluster 6')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 300, c = 'yellow', label = 'Centroids')






plt.legend()
plt.show()
labels= data.columns.values
labels
Y_Kmeans
data.head()
data["cluster"]=Y_Kmeans
data.head()
data[data["cluster"]==1]
# Clustering Analysis

clustering =  data.iloc[:,2:18]
clustering 
clustering["cluster"]=Y_Kmeans
clustering.head()
for c in clustering:
    grid= sns.FacetGrid(clustering, col='cluster')
    grid.map(plt.hist, c)
#Cluster0 People with average to high credit limit who make all type of purchases

#Cluster1 This group has more people with due payments who take advance cash more often

#Cluster2 Less money spenders with average to high credit limits who purchases mostly in installments

#Cluster3 People with high credit limit who take more cash in advance

#Cluster4 High spenders with high credit limit who make expensive purchases

#Cluster5 People who don't spend much money and who have average to high credit limit