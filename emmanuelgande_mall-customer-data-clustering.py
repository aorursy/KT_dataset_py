import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set_style('whitegrid')
df = pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
df.head(2)
df.shape
df.info()
df.describe()
100*df.isnull().sum()/df.shape[0]
sns.countplot(df['Gender'])
plt.xlabel('Gender')
plt.ylabel('Number of Customers')
plt.title('Customer Distribution by Gender')
sns.distplot(df['Age'],kde=False,bins=30)
plt.ylabel('Frequency')
plt.title('Customer Distribution by Age')
sns.distplot(df[df['Gender'] == 'Male']['Age'],kde=False,bins=30,color='blue',label='Male')
sns.distplot(df[df['Gender'] == 'Female']['Age'],kde=False,bins=30,color='red',label='Female')
plt.title('Customer Distribution by Age and Gender')
plt.ylabel('Number of Customers')
plt.legend()
sns.boxplot(df['Age'])
sns.distplot(df['Annual Income (k$)'],kde=False,bins=25)
plt.title('Customer Distribution by Annual Income (k$)')
plt.ylabel('Frequency')
sns.distplot(df[df['Gender'] == 'Male']['Annual Income (k$)'],kde=False,bins=30,color='blue',label='Male')
sns.distplot(df[df['Gender'] == 'Female']['Annual Income (k$)'],kde=False,bins=30,color='red',label='Female')
plt.title('Customer Distribution by Annual Income (k$) and Gender')
plt.ylabel('Number of Customers')
plt.legend()
sns.boxplot(df['Annual Income (k$)'])
sns.distplot(df['Spending Score (1-100)'])
sns.distplot(df[df['Gender'] == 'Male']['Spending Score (1-100)'],color='blue',kde=False,label='Male')
sns.distplot(df[df['Gender'] == 'Female']['Spending Score (1-100)'],color='red',kde=False,label='Female')
plt.legend()
sns.boxplot(df['Spending Score (1-100)'])
sns.pairplot(data=df,hue='Gender',palette='coolwarm')
plt.figure(figsize=(12,6))
sns.heatmap(df.corr(),cmap='viridis',annot=True)
sns.lmplot(x='Annual Income (k$)',y='CustomerID',data=df)
plt.title('CustomerID vs. Annual Income (k$)')
df.dtypes
df['Gender'].unique()
df.Gender.value_counts()
dmap = {'Male':1,'Female':0}
df['Gender'] = df['Gender'].map(dmap)
df.head(5)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df)
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(data_scaled)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss,marker='*')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
kmeans = KMeans(n_clusters=5,init='k-means++',max_iter=300,tol=0.0001).fit(data_scaled)
kmeans
kmeans.labels_
kmeans.cluster_centers_
clusters_data = pd.DataFrame(data_scaled,columns=['CustomerID', 'Genre', 'Age', 'Annual Income (k$)',
       'Spending Score (1-100)'])
clusters_data['Cluster'] = kmeans.labels_
clusters_data.head()
clusters_data.groupby('Cluster').count()['Spending Score (1-100)']
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(clusters_data['Cluster'],kmeans.labels_))
print(classification_report(clusters_data['Cluster'],kmeans.labels_))