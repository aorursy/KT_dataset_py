import pandas as pd
mall_df = pd.read_csv ("../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv")
mall_df
mall_df.isnull().values.ravel().sum()
mall_df.isna().sum()
mall_df_0 = pd.get_dummies(mall_df)
corr= mall_df_0.corr()
corr
import seaborn as sns
sns.heatmap(corr, linewidths = 0.5, annot=True, center=0, cmap="YlGnBu")
mall_df.describe()
sns.pairplot(mall_df)
def outliers (mall_df):
    import numpy as np
    import statistics as sts

    for i in mall_df.describe().columns:
        No = np.array(mall_df[i])
        p=[]
        Q1 = mall_df[i].quantile(0.25)
        Q3 = mall_df[i].quantile(0.75)
        IQR = Q3 - Q1
        LTV= Q1 - (1.5 * IQR)
        UTV= Q3 + (1.5 * IQR)
        for j in No:
            if j <= LTV or j>=UTV:
                p.append(sts.median(No))
            else:
                p.append(j)
        mall_df[i]=p
    return mall_df
mall_df_out = outliers(mall_df)
mall_df_out
mall = mall_df_out.iloc[:, [3, 4]].values
mall
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    model_km = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    model_km.fit(mall)
    wcss.append(model_km.inertia_)
wcss
wcss_df = pd.DataFrame(wcss)
wcss_df
import matplotlib.pyplot as plt
plt.plot(wcss_df)
plt.title('The Elbow Method', fontsize = 20)
plt.xlabel('No. of Clusters')
plt.ylabel('wcss')
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(mall[0:10], method = 'single'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Ecuclidean Distance')
plt.show()
model_kmns = KMeans(n_clusters= 5, init='k-means++', random_state=0)
predict = model_kmns.fit_predict(mall)
predict
plt.scatter(mall[predict == 0, 0], mall[predict == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(mall[predict == 1, 0], mall[predict == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(mall[predict == 2, 0], mall[predict == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(mall[predict == 3, 0], mall[predict == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(mall[predict == 4, 0], mall[predict == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(model_kmns.cluster_centers_[:, 0], model_kmns.cluster_centers_[:, 1], s = 100, c = 'yellow', label = 'Centroids')
plt.title('Customers clusters')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()