# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/Mall_Customers.csv")
df.info()
import seaborn as sns
gend_count=df['Gender'].value_counts()
gend_count_scr=pd.DataFrame(df.groupby(['Gender'])['Spending Score (1-100)'].mean())
sns.countplot(x='Gender',data=df)
plt.title("Count of Customers on the basis of Gender")
gend_count_scr=gend_count_scr.reset_index()
gend_count_scr.plot.bar(x='Gender',y='Spending Score (1-100)')
plt.title("Spending Score of Customers on the basis of Gender")
gend_count_sal=pd.DataFrame(df.groupby(['Gender'])['Annual Income (k$)'].mean())
gend_count_sal=gend_count_sal.reset_index()
gend_count_sal.plot.bar(x='Gender',y='Annual Income (k$)')
plt.title("Salary of the customers based on gender")
age_count=pd.DataFrame(df['Age'].value_counts())
age_count=age_count.reset_index()
df['Age'].plot.hist(by='Age')
plt.title("Distribution of Age of customers")
age_count_scr=pd.DataFrame(df.groupby('Age')['Spending Score (1-100)'].mean())
age_count_scr=age_count_scr.reset_index()
plt.figure(figsize=(10,5))
plt.bar(age_count_scr['Age'],age_count_scr['Spending Score (1-100)'])
plt.xlabel("Age")
plt.ylabel("Average Scoring Score")
age_count_sal=pd.DataFrame(df.groupby('Age')['Annual Income (k$)'].mean())
age_count_sal=age_count_sal.reset_index()
plt.figure(figsize=(10,5))
plt.bar(age_count_sal['Age'],age_count_sal['Annual Income (k$)'])
plt.xlabel("Age")
plt.ylabel("Average Annual Income (k$)")
fig,axes=plt.subplots(1,2,figsize=(15,10))
colors = { 'Male' : 'blue', 'Female': 'red'}
axes[0].scatter(df['Age'],df['Annual Income (k$)'],c=df['Gender'].apply(lambda x:colors[x]))
axes[0].set_xlabel("Age")
axes[0].set_ylabel("Annual Income (k$)")
axes[1].scatter(df['Age'],df['Spending Score (1-100)'],c=df['Gender'].apply(lambda x:colors[x]))
axes[1].set_xlabel('Age')
axes[1].set_ylabel('Spending Score (1-100)')
plt.scatter(df['Annual Income (k$)'],df['Spending Score (1-100)'])
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
from sklearn.cluster import KMeans
squared_dist=[]
for i in range(1,21):
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df[['Annual Income (k$)','Spending Score (1-100)']])
    squared_dist.append(kmeans.inertia_)
plt.plot(list(range(1,21)),squared_dist)
plt.xlabel("no of clusters")
plt.ylabel("sum of Squared distances from the clusters")
kmeans=KMeans(n_clusters=5)
y_predict=kmeans.fit_predict(df[['Annual Income (k$)','Spending Score (1-100)']])
X=df[['Annual Income (k$)','Spending Score (1-100)']]
plt.scatter(X.iloc[y_predict==0,0],X.iloc[y_predict==0,1],c='red',label='Cluster 1')
plt.scatter(X.iloc[y_predict==1,0],X.iloc[y_predict==1,1],c='blue',label='Cluster 2')
plt.scatter(X.iloc[y_predict==2,0],X.iloc[y_predict==2,1],c='green',label='Cluster 3')
plt.scatter(X.iloc[y_predict==3,0],X.iloc[y_predict==3,1],c='yellow',label='Cluster 4')
plt.scatter(X.iloc[y_predict==4,0],X.iloc[y_predict==4,1],c='violet',label='Cluster 5')
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.legend()
plt.scatter(df.iloc[y_predict==0,2],df.iloc[y_predict==0,3],c='red',label='Cluster 1')
plt.scatter(df.iloc[y_predict==1,2],df.iloc[y_predict==1,3],c='blue',label='Cluster 2')
plt.scatter(df.iloc[y_predict==2,2],df.iloc[y_predict==2,3],c='green',label='Cluster 3')
plt.scatter(df.iloc[y_predict==3,2],df.iloc[y_predict==3,3],c='yellow',label='Cluster 4')
plt.scatter(df.iloc[y_predict==4,2],df.iloc[y_predict==4,3],c='violet',label='Cluster 5')
plt.xlabel("Age")
plt.ylabel("Annual Income")
plt.legend()
plt.scatter(df.iloc[y_predict==0,2],df.iloc[y_predict==0,4],c='red',label='Cluster 1')
plt.scatter(df.iloc[y_predict==1,2],df.iloc[y_predict==1,4],c='blue',label='Cluster 2')
plt.scatter(df.iloc[y_predict==2,2],df.iloc[y_predict==2,4],c='green',label='Cluster 3')
plt.scatter(df.iloc[y_predict==3,2],df.iloc[y_predict==3,4],c='yellow',label='Cluster 4')
plt.scatter(df.iloc[y_predict==4,2],df.iloc[y_predict==4,4],c='violet',label='Cluster 5')
plt.xlabel("Age")
plt.ylabel("Spending Score")
plt.legend()
