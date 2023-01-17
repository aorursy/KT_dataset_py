# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load



# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np # useful python library for linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # data visualization
%matplotlib notebook

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from mpl_toolkits import mplot3d

import warnings
warnings.filterwarnings('ignore')
pd.set_option('max_rows',200)
df=pd.read_csv('/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv') # importing data
df.set_index('CustomerID',inplace=True) # setting customer id as index
df.head(10) # quick glimpse of data
df.info() # getting to know the data
df.describe() # various statistical analysis of the data
"""
Visualization of Gender distribution in data
Right figure- bar plot
Left figure- pie chart

"""

plt.figure(figsize = (15 , 6))

plt.subplot(1,2,1)
sns.countplot(y='Gender',data=df,palette='colorblind')
ax=plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


plt.subplot(1,2,2)
plt.pie(x=df['Gender'].value_counts(),labels=['Female','Male'],autopct='%1.2f%%')
ax=plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
# Distribution of Numerical data with the help of histograms

plt.figure(figsize = (15 , 6))
n = 0 
for x in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:
    n += 1
    plt.subplot(1 , 3 , n)
    plt.subplots_adjust(hspace =0.5 , wspace = 0.5)
    sns.distplot(df[x] , bins = 20)
    plt.title('Distplot of {}'.format(x))
    ax=plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()
cols=df.drop('Gender',axis=1).columns
x=sns.PairGrid(data=df,hue='Gender',vars=cols,palette='colorblind',layout_pad=True)
x.map_offdiag(sns.scatterplot)
x.add_legend()
x.fig.set_size_inches(15,15)
x
plt.figure(figsize=(15,8))
plt.subplot(2,1,1)
sns.barplot(x=df['Age'], y=df['Annual Income (k$)'], hue=df['Gender'], ci=0)
plt.title('Income by Age')
plt.xlabel('')

plt.subplot(2,1,2)
sns.barplot(x=df['Age'], y=df['Spending Score (1-100)'], hue=df['Gender'], ci=0)
plt.title('Score by Age')

plt.show()
plt.figure(figsize=(10,10))
sns.heatmap(df.corr(), cmap = 'Wistia', annot = True)
plt.title('Heatmap for the Data', fontsize = 20)
plt.show()
#Segmentation using Age and Spending Score Data

X1=df[['Age' , 'Spending Score (1-100)']]

scaler=MinMaxScaler()
X1=scaler.fit_transform(X1)

error1=[]
for i in range(1,16):
    clf1=KMeans(n_clusters = i ,init='k-means++',n_init = 10,max_iter=300,tol=0.0001,random_state=0,algorithm='auto')
    clf1.fit(X1)
    error1.append(clf1.inertia_)
"""
Elbow Method to find optimal number of clusters
"""

plt.figure(figsize=(15,7))
sns.lineplot(x=range(1,16),y=error1,ci=None)
plt.xticks(ticks=range(1,16),labels=range(1,16))
ax=plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.title('Segmentation using Age and Spending Score (Elbow Method)')
# Density Distribution of data
sns.jointplot(x='Spending Score (1-100)',y='Age',data=df,kind='kde')
clf1=KMeans(n_clusters = 2 ,init='k-means++',n_init = 10,max_iter=300,tol=0.0001,random_state=0,algorithm='auto')
clf1.fit(X1)

labels1=clf1.labels_
centroids1=clf1.cluster_centers_
df['labels1']=labels1

plt.figure(figsize=(12,12))
sns.scatterplot(x='Age',y='Spending Score (1-100)',data=df,hue='labels1',s=100,palette='colorblind',legend=False)
ax=plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.title('Segmentation using Age and Spending Score (Cluster Visualisation)')
#Segmentation using Annual Income (k$) and Spending Score

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler,MinMaxScaler
X2=df[['Annual Income (k$)' , 'Spending Score (1-100)']]

scaler=MinMaxScaler()
X2=scaler.fit_transform(X2)

error2=[]
for i in range(1,16):
    clf2=KMeans(n_clusters = i ,init='k-means++',n_init = 10,max_iter=300,tol=0.0001,random_state=0,algorithm='auto')
    clf2.fit(X2)
    error2.append(clf2.inertia_)
plt.figure(figsize=(15,7))
sns.lineplot(x=range(1,16),y=error2,ci=None)
plt.xticks(ticks=range(1,16),labels=range(1,16))
ax=plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.title('Segmentation using Annual Income (k$) and Spending Score (Elbow Method)')
# Density Distribution of data
sns.jointplot(x='Spending Score (1-100)',y='Annual Income (k$)',data=df,kind='kde')
clf2=KMeans(n_clusters = 5 ,init='k-means++',n_init = 10,max_iter=300,tol=0.0001,random_state=0,algorithm='auto')
clf2.fit(X2)

labels2=clf2.labels_
centroids2=clf2.cluster_centers_
df['labels2']=labels2

plt.figure(figsize=(12,12))
sns.scatterplot(x='Annual Income (k$)',y='Spending Score (1-100)',data=df,hue='labels2',palette=sns.color_palette("hls", 5),s=100,legend='full')
ax=plt.gca()
ax.spines['top'].set_visible(False)  
ax.spines['right'].set_visible(False)
plt.title('Segmentation using Annual Income (k$) and Spending Score (Cluster Visualisation)')
#Segmentation using Annual Income (k$) and Age

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler,MinMaxScaler
X3=df[['Annual Income (k$)' , 'Age']]

scaler=MinMaxScaler()
X3=scaler.fit_transform(X3)

error3=[]
for i in range(1,16):
    clf3=KMeans(n_clusters = i ,init='k-means++',n_init = 10,max_iter=300,tol=0.0001,random_state=0,algorithm='auto')
    clf3.fit(X3)
    error3.append(clf3.inertia_)
plt.figure(figsize=(15,7))
sns.lineplot(x=range(1,16),y=error3,ci=None)
plt.xticks(ticks=range(1,16),labels=range(1,16))
ax=plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.title('Segmentation using Annual Income (k$) and Age (Elbow Method)')
# Density Distribution of data
sns.jointplot(x='Age',y='Annual Income (k$)',data=df,kind='kde')
clf3=KMeans(n_clusters = 3 ,init='k-means++',n_init = 10,max_iter=300,tol=0.0001,random_state=0,algorithm='auto')
clf3.fit(X3)

labels3=clf3.labels_
centroids3=clf3.cluster_centers_
df['labels3']=labels3

plt.figure(figsize=(12,12))
sns.scatterplot(y='Annual Income (k$)',x='Age',data=df,hue='labels3',s=100,palette='colorblind',legend='full')
ax=plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.title('Segmentation using Annual Income (k$) and Age (Cluster Visualisation)')
#Segmentation using Annual Income (k$),Age and Spending Score (1-100)   

X4=df[['Annual Income (k$)' , 'Age','Spending Score (1-100)']]

scaler=MinMaxScaler()     
X4=scaler.fit_transform(X4)

error4=[]
for i in range(1,16):
    clf4=KMeans(n_clusters = i ,init='k-means++',n_init = 10,max_iter=300,tol=0.0001,random_state=0,algorithm='auto')
    clf4.fit(X4)
    error4.append(clf4.inertia_)
plt.figure(figsize=(15,7))
sns.lineplot(x=range(1,16),y=error4,ci=None)
plt.xticks(ticks=range(1,16),labels=range(1,16))
ax=plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.title('Segmentation using Annual Income (k$), Age and Spending Score (1-100) (Elbow Method)')
clf4=KMeans(n_clusters = 4 ,init='k-means++',n_init = 10,max_iter=300,tol=0.0001,random_state=0,algorithm='auto')
clf4.fit(X4)

labels4=clf4.labels_
centroids4=clf4.cluster_centers_
df['labels4']=labels4

fig=plt.figure(figsize=(15,15))
ax=plt.axes(projection='3d')
z=np.array(df['Age'])
x=np.array(df['Spending Score (1-100)'])
y=np.array(df['Annual Income (k$)'])

ax.set_xlabel('Spending Score (1-100)')
ax.set_ylabel('Annual Income (k$)')
ax.set_zlabel('Age')

ax.scatter3D(x,y,z,c=np.array(df['labels4']),cmap='rainbow',s=100)