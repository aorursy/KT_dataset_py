# load the churn file and understanding the data
import pandas as pd
data=pd.read_csv("churn.csv")
data.head()
# drop the  customerID column and store churn data sepertely
data=data.drop(['customerID'], axis =1) 
churn=data['Churn']
# Total No of records
data.shape
#Checking the missing values if any
data.isnull().sum()
#Understanding the data types associated with it
data.dtypes
#converting the totalcharges from object type to float  type
data.TotalCharges = pd.to_numeric(data.TotalCharges, errors='coerce')
#checking the datatypes again
data.dtypes
# Understanding the Total No of records
data.shape
# Decide which categorical variables you want to use in model
for col_name in data.columns:
    if data[col_name].dtypes == 'object':# in pandas it is object
        unique_cat = len(data[col_name].unique())
        print("Feature '{col_name}' has {unique_cat} unique categories".format(col_name=col_name, unique_cat=unique_cat))
        print(data[col_name].value_counts())
        print()
data.head()
#Removing the churn column
data=data.drop(['Churn'], axis =1)
#checking the shape after removing churn column
data.shape
#generating dummy variables for categorical variables by using get_dummies
data = pd.get_dummies(data = data)
# Lets understand the genererted columns
data.columns
#data view
data.head()
# removing unwandted rows
data.dropna(axis=0,inplace=True)
data.head()
data
# lets see the sample of first 5 customers
data.head()


#reduced no. of rows from 7043 to 7032
data.shape
# storeing the generated values in a seperate sheet for future revenue and segmentation calculations
data.to_csv('data2.csv')
#Using the elbow method to find the optimum number of clusters
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
wcss = []
for i in range(1,11):
    km=KMeans(n_clusters=i,init='k-means++', max_iter=300, n_init=10, random_state=0)
    km.fit(data)
    wcss.append(km.inertia_)
plt.plot(range(1,11),wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('wcss')
plt.show()
#applying K means clustering
kmeans_1=KMeans(n_clusters=4)
kmeans_1.fit(data)
cluster_pred=kmeans_1.predict(data)
cluster_pred_2=kmeans_1.labels_
cluster_center=kmeans_1.cluster_centers_ # shows the 4 centroids
#storing in a sperate file for calculations 
clusterss = pd.DataFrame(cluster_pred_2)
# reading the stored cluster file
dfc=pd.read_csv("clusterss.csv", sep=",")

#performing group by on cluster number
dfc1=dfc.groupby('Cluster Number')
dfc1
#number of clusters and assigned customers for each cluster
print(dfc1.count())
#percentage calculations
c0=1691
c1=1183
c2=3185
c3=973
T_C=1691+1183+3185+973
c0_Per=c0/T_C
c1_Per=c1/T_C
c2_Per=c2/T_C
c3_Per=c3/T_C

      
print(' Segemnet0 percentage is equal to ', c0_Per)
print(' Segemnet1 percentage is equal to ', c1_Per)
print(' Segemnet2 percentage is equal to ', c2_Per)
print(' Segemnet3 percentage is equal to ', c3_Per)
print(dfc1.describe())
# revenue is calculated by using excel pivot operations, by mapping cluster numbers to the total charges of individual customers
c0rev=3285503
c1rev=4939820
c2rev=1355779
c3rev=6475065

print(' Segemnet0 Revenue is equal to ', c0rev)
print(' Segemnet1 Revenue  is equal to ', c1rev)
print(' Segemnet2 Revenue  is equal to ', c2rev)
print(' Segemnet3 Revenue  is equal to ', c3rev)
#from the above sheet, sheet will be attached along with Jupyter PDF
print(' Segemnet0 Customers at risk of leaving', '23%')
print(' Segemnet1 Customers at risk of leaving', '17%')
print(' Segemnet2 Customers at risk of leaving', '36%')
print(' Segemnet3 Customers at risk of leaving', '13%')
#from the above sheet,
print(' Segemnet0 Revenue at risk ', '784985$')
print(' Segemnet1 Revenue at risk', '840772$')
print(' Segemnet2 Revenue at risk', '1355779$')
print(' Segemnet3 Revenue at risk', '837483$')
X1= data.iloc[:, [1,3]].values
X1[1]
# Visualising the clusters
from sklearn.cluster import KMeans
plt.scatter(X1[y_kmeans == 0, 0], X1[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 0')
plt.scatter(X1[y_kmeans == 1, 0], X1[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 1')
plt.scatter(X1[y_kmeans == 2, 0], X1[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 2')
plt.scatter(X1[y_kmeans == 3, 0], X1[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 3')

plt.title('Clusters of customers')
plt.xlabel('tenure (k$)')
plt.ylabel('Total charges (1-100)')
plt.legend()
plt.show()

X2= data.iloc[:, [1,2]].values
X2[1]
# Visualising the clusters
plt.scatter(X2[y_kmeans == 0, 0], X2[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 0')
plt.scatter(X2[y_kmeans == 1, 0], X2[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 1')
plt.scatter(X2[y_kmeans == 2, 0], X2[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 2')
plt.scatter(X2[y_kmeans == 3, 0], X2[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 3')
plt.title('Clusters of customers')
plt.xlabel('Tenure')
plt.ylabel('Monthly Charges (1-100)')
plt.legend()
plt.show()
#forming dendrogram
import scipy
from scipy.cluster import hierarchy
dendro=hierarchy.dendrogram(hierarchy.linkage(data,method='ward'))
plt.axhline(y=45000)# cut at 45000 to get 4 clusters
#performing Agglomerative Clustering
from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')  #took n=4 from elbow curve 
Agg=cluster.fit_predict(data)
# creating a new pandas dataframe for the output
Aggclusterss = pd.DataFrame(Agg)
#creating a new sample csv file
Aggclusterss.to_csv('Aggclusterss.csv') 
#read the file
Aggdfc=pd.read_csv("Aggclusterss.csv", sep=",")

# group by cluster numbers
Aggdfc1=Aggdfc.groupby('0')
Aggdfc1
# no of clusters and customers assigned to each cluster
print(Aggdfc1.count())
#calculating segment percentages
c0_HI=1651
c1_HI=2399
c2_HI=1089
c3_HI=1893
T_C_HI=c0+c1+c2+c3
print(T_C_HI)
c0_Per_HI=c0_HI/T_C_HI
c1_Per_HI=c1_HI/T_C_HI
c2_Per_HI=c2_HI/T_C_HI
c3_Per_HI=c3_HI/T_C_HI

      
print(' Segemnet0 percentage is equal to ', c0_Per_HI)
print(' Segemnet1 percentage is equal to ', c1_Per_HI)
print(' Segemnet2 percentage is equal to ', c2_Per_HI)
print(' Segemnet3 percentage is equal to ', c3_Per_HI)
# from the above sheet
print(' Segemnet0 Customers at risk of leaving', '21%')
print(' Segemnet1 Customers at risk of leaving', '23%')
print(' Segemnet2 Customers at risk of leaving', '13%')
print(' Segemnet3 Customers at risk of leaving', '42%')
# from the above sheet
print(' Segemnet0 Revenue at risk ', '1172149$')
print(' Segemnet1 Revenue at risk', '620952$')
print(' Segemnet2 Revenue at risk', '931800$')
print(' Segemnet3 Revenue at risk', '125979$')
X4= data.iloc[:, [1,3]].values
X4[1]
plt.figure(figsize=(10, 7))  
plt.scatter(X4[:,0], X4[:,1], c=cluster.labels_, cmap='rainbow') 
plt.title('Clusters of customers')
plt.xlabel('tenure (k$)')
plt.ylabel('Total charges (1-100)')

data.shape