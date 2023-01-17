import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
% matplotlib inline
df=pd.read_csv('../input/who_suicide_statistics.csv')
df.head()
new_df=df.copy()
new_df.head()
df.info()
df.fillna(0,inplace=True)
df.head()
df.info()
df.country.unique()
df.year.unique()
df.age.unique()
df.suicides_no.unique()
df=df.replace(['male','female'],[0,1])
df=df.replace(['5-14 years','15-24 years','25-34 years','35-54 years','55-74 years','75+ years'],[0,1,2,3,4,5])
df.head()
df['population']=df['population'].astype(int)
df.head()
df['suicides_no']=df['suicides_no'].astype(int)
df.head()
df.suicides_no.unique()
pd.scatter_matrix(df,color='red',figsize=(10,10),diagonal='hist')
sns.barplot(x='age',y='suicides_no',data=df,ci=0)
plt.title(" total suicides in age group ")

#(['5-14 years','15-24 years','25-34 years','35-54 years','55-74 years','75+ years'],[0,1,2,3,4,5])
sns.barplot(x='sex',y='suicides_no',data=df,ci=0)
plt.title("total suicides committed by male and female")


#(['male','female'],[0,1])
sns.barplot(x='age',y='suicides_no',data=df,hue='sex',palette='spring',ci=0)
plt.title(" total suicides in age group classified by gender")
sns.barplot(x='year',y='suicides_no',data=df,ci=0)
plt.title(" total suicides committed over the years ")
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn import preprocessing
from scipy.spatial.distance import cdist
df.head()
cluster=df[['year','sex','age','population']]
cluster_s=cluster.copy()
cluster_s['population']=preprocessing.scale(cluster_s['population'].astype('float64'))
cluster_s['year']=preprocessing.scale(cluster_s['year'].astype('float64'))
cluster_s['sex']=preprocessing.scale(cluster_s['sex'].astype('float64'))
cluster_s['age']=preprocessing.scale(cluster_s['age'].astype('float64'))
cluster_train,cluster_test=train_test_split(cluster_s,test_size=0.3,random_state=7)
cluster_train.shape,cluster_test.shape
cluster=range(1,11)
mean_dist=[]
for k in cluster:
    model=KMeans(n_clusters=k)
    model.fit(cluster_train)
    mean_dist.append(sum(np.min(cdist(cluster_train,model.cluster_centers_,'euclidean'),axis=1))/cluster_train.shape[0])
plt.figure(figsize=(10,8))
plt.plot(cluster,mean_dist)
plt.xlabel("Number of clusters")
plt.ylabel("Average distance")
plt.title("Elbow Curve")
plt.xticks(range(1,11))
from sklearn.decomposition import PCA
model1=KMeans(n_clusters=2)
model1.fit(cluster_train)
pca_2=PCA(2)
plt.figure(figsize=(10,6))
plot_columns=pca_2.fit_transform(cluster_train)
plt.scatter(x=plot_columns[:,0],y=plot_columns[:,1],c=model1.labels_,)
plt.xlabel("Canonical Varialbe 1")
plt.ylabel("Canonical variable 2")
plt.title("2 canonical variable for 2 cluster")
plt.show()