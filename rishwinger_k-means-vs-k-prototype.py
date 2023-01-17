# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, normalize,MinMaxScaler

from sklearn.cluster import KMeans

from kmodes.kprototypes import KPrototypes

import warnings

warnings.filterwarnings("ignore") 

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')

df.head()
missing_cols=df.isnull().sum()/df.shape[0]

missing_cols=missing_cols[missing_cols>0]

missing_cols
df.set_index('CustomerID',inplace=True)

df.head()
num_cols=df.select_dtypes(include=['int64']).columns

ctg_cols=df.select_dtypes(include=['object']).columns



print('Numerical Cols=',num_cols)

print('Categorical Cols=',ctg_cols)
cols_val=2

fig, ax = plt.subplots(len(num_cols),cols_val,figsize=(12, 5))

colours_val=['c','b','r','g','y','p','m']

did_not_ran=True

for i,col in enumerate(num_cols):

    for j in range(cols_val):

        if did_not_ran==True:

            sns.boxplot(df[col],ax=ax[i,j],color=colours_val[i+j])

            ax[i,j].set_title(col)

            did_not_ran=False

        else:

            sns.distplot(df[col],ax=ax[i,j],color=colours_val[i+j])

            ax[i,j].set_title(col)

            did_not_ran=True

            

            

plt.suptitle("EDA")

plt.tight_layout()

plt.show()
plt.figure(figsize=(12,5))

sns.scatterplot(df['Annual Income (k$)'] ,df['Spending Score (1-100)'])

plt.title('Scatterplot')

plt.show()
print('Min Age =',df.Age.min())

print('Max Age =',df.Age.max())
df['Age_bins']=pd.cut(df.Age,bins=(17,35,50,70),labels=["18-35","36-50","50+"])

df[['Age','Age_bins']].drop_duplicates(subset=['Age_bins']).reset_index(drop=True)
df1=df[['Annual Income (k$)', 'Spending Score (1-100)']]

df1.shape
std=MinMaxScaler()

arr1=std.fit_transform(df1)

%%time

kmeans_cluster=KMeans(n_clusters=2,random_state=7)

result_cluster=kmeans_cluster.fit_predict(arr1)
df1['Clusters']=result_cluster

df1['Clusters'].value_counts()
ax=sns.countplot(x=df1.Clusters)

for index, row in pd.DataFrame(df1['Clusters'].value_counts()).iterrows():

    ax.text(index,row.values[0], str(round(row.values[0])),color='black', ha="center")

    #print(index,row.values[0])

plt.title('Cluster Count')

plt.show()
plt.figure(figsize=(12,5))

sns.scatterplot(x=df1['Annual Income (k$)'],y=df1['Spending Score (1-100)'],hue=df1.Clusters,palette="Set2",)

plt.title('2 Clusters')

plt.show()
fig,ax=plt.subplots(1,2,figsize=(12,5))

sns.heatmap(df1.loc[df1.Clusters==0,['Annual Income (k$)', 'Spending Score (1-100)']].describe().round(),annot=True,fmt='g',ax=ax[0])

ax[0].set_title("Cluster-0")

sns.heatmap(df1.loc[df1.Clusters==1,['Annual Income (k$)', 'Spending Score (1-100)']].describe().round(),annot=True,fmt='g',ax=ax[1])

ax[1].set_title("Cluster-1")

plt.suptitle("Cluster Analysis")

plt.show()
%%time

SSE = []

for i in range(1, 11):

    kmeans = KMeans(n_clusters = i, random_state = 7)

    kmeans.fit(arr1)

    SSE.append(kmeans.inertia_)
plt.figure(figsize=(12,5))

sns.lineplot(range(1, 11), SSE,marker='o')

plt.title('Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('SSE')

plt.show()
kmeans_cluster=KMeans(n_clusters=5,random_state=7)

result_cluster=kmeans_cluster.fit_predict(arr1)
df1['Clusters']=result_cluster

df1['Clusters'].value_counts()
d1=df[['Gender','Age_bins']].reset_index(drop=True)

df1_comb=pd.concat([df1.reset_index(drop=True),d1],axis=1)

df1_comb.head()
ax=sns.countplot(x=df1_comb.Clusters)

for index, row in pd.DataFrame(df1_comb['Clusters'].value_counts()).iterrows():

    ax.text(index,row.values[0], str(round(row.values[0])),color='black', ha="center")

    #print(index,row.values[0])

plt.title('Cluster Count')

plt.show()
plt.figure(figsize=(12,7))

sns.scatterplot(x=df1_comb['Annual Income (k$)'],y=df1_comb['Spending Score (1-100)'],hue=df1_comb.Clusters,palette="Set2",)

plt.title('5 Clusters')

plt.show()
fig,ax=plt.subplots(1,5,figsize=(15,5))

#cbar_ax = fig.add_axes([1.03, .3, .03, .4])

for cluster_val in sorted(df1_comb.Clusters.unique()):

    #print(cluster_val)

    sns.heatmap(df1_comb.loc[df1_comb.Clusters==cluster_val,['Annual Income (k$)', 'Spending Score (1-100)']].describe().round(),annot=True,fmt='g',ax=ax[cluster_val],\

               cbar=i == 0,vmin=0, vmax=130)

    titl='Cluster-'+str(cluster_val)

    ax[cluster_val].set_title(titl)

    

plt.suptitle('Clustering Analysis')



#plt.tight_layout()

plt.show()
fig,ax=plt.subplots(1,5,figsize=(16,5))

#cbar_ax = fig.add_axes([1.03, .3, .03, .4])

for cluster_val in sorted(df1_comb.Clusters.unique()):

    #print(cluster_val)

    sns.heatmap(df1_comb.loc[df1_comb.Clusters==cluster_val,:].groupby('Age_bins').agg({'Clusters':'size','Annual Income (k$)':'mean','Spending Score (1-100)':'mean'}).\

    rename(columns={'Clusters':'Count','Annual Income (k$)':'IncomeMean','Spending Score (1-100)':'SpendScoreMean'})\

                .fillna(0).round(),annot=True,fmt='g',ax=ax[cluster_val],cbar=i == 0,vmin=0, vmax=130)

    titl='Cluster-'+str(cluster_val)+' Analysis'

    ax[cluster_val].set_title(titl)

    



plt.suptitle('Clustering Age wise Analysis')



#plt.tight_layout()

plt.show()
fig,ax=plt.subplots(1,5,figsize=(16,5))

#cbar_ax = fig.add_axes([1.03, .3, .03, .4])

for cluster_val in sorted(df1_comb.Clusters.unique()):

    #print(cluster_val)

    sns.heatmap(df1_comb.loc[df1_comb.Clusters==cluster_val,:].groupby('Gender').agg({'Clusters':'size','Annual Income (k$)':'mean','Spending Score (1-100)':'mean'}).\

    rename(columns={'Clusters':'Count','Annual Income (k$)':'IncomeMean','Spending Score (1-100)':'SpendScoreMean'})\

                .fillna(0).round(),annot=True,fmt='g',ax=ax[cluster_val],cbar=i == 0,vmin=0, vmax=130)

    titl='Cluster-'+str(cluster_val)+' Analysis'

    ax[cluster_val].set_title(titl)

    



plt.suptitle('Clustering Gender Wise Analysis')



#plt.tight_layout()

plt.show()
plt.figure(figsize=(12,5))



sns.boxplot(x='Clusters',y='value',hue='variable',\

            data=pd.melt(df1,id_vars=['Clusters'],value_vars=['Annual Income (k$)','Spending Score (1-100)']),\

           palette="Set2")

plt.xlabel("Clusters")

plt.title("Boxplot-Annual Income - Spending Score")

plt.show()
df_proto=pd.DataFrame(arr1,columns=['AnnualIcome','SpendingScore'])

df_proto.head()
d2=pd.concat([df_proto,d1],axis=1)

d2.head()
%%time

kproto_clusters=KPrototypes(n_clusters=5,random_state=7,init="Cao")

result_cluster=kproto_clusters.fit_predict(d2,categorical=[2,3])
d2['Clusters']=result_cluster

d2['Clusters'].value_counts()
ax=sns.countplot(x=d2.Clusters)

for index, row in pd.DataFrame(d2['Clusters'].value_counts()).iterrows():

    ax.text(index,row.values[0], str(round(row.values[0])),color='black', ha="center")

    #print(index,row.values[0])

plt.title('Cluster Count')

plt.show()
kproto_clusters.cluster_centroids_
df1.drop(['Clusters'],axis=1,inplace=True)

d3=pd.concat([df1.reset_index(drop=True),d2],axis=1)

d3.head()
plt.figure(figsize=(12,5))

sns.scatterplot(x=d3['Annual Income (k$)'],y=d3['Spending Score (1-100)'],hue=d3.Clusters,palette="Set2",)

plt.title('5 Clusters')

plt.show()
fig,ax=plt.subplots(1,5,figsize=(15,5))

#cbar_ax = fig.add_axes([1.03, .3, .03, .4])

for cluster_val in sorted(d3.Clusters.unique()):

    #print(cluster_val)

    sns.heatmap(d3.loc[d3.Clusters==cluster_val,['Annual Income (k$)', 'Spending Score (1-100)']].describe().round(),annot=True,fmt='g',ax=ax[cluster_val],\

               cbar=i == 0,vmin=0, vmax=130)

    titl='Cluster-'+str(cluster_val)

    ax[cluster_val].set_title(titl)

    



plt.suptitle('Clustering Analysis')



#plt.tight_layout()

plt.show()
fig,ax=plt.subplots(1,5,figsize=(16,5))

#cbar_ax = fig.add_axes([1.03, .3, .03, .4])

for cluster_val in sorted(d3.Clusters.unique()):

    #print(cluster_val)

    sns.heatmap(d3.loc[d3.Clusters==cluster_val,:].groupby('Age_bins').agg({'Clusters':'size','Annual Income (k$)':'mean','Spending Score (1-100)':'mean'}).\

    rename(columns={'Clusters':'Count','Annual Income (k$)':'IncomeMean','Spending Score (1-100)':'SpendScoreMean'})\

                .fillna(0).round(),annot=True,fmt='g',ax=ax[cluster_val],cbar=i == 0,vmin=0, vmax=130)

    titl='Cluster-'+str(cluster_val)+' Analysis'

    ax[cluster_val].set_title(titl)

    



plt.suptitle('Clustering Age wise Analysis')



#plt.tight_layout()

plt.show()
fig,ax=plt.subplots(1,5,figsize=(16,5))

#cbar_ax = fig.add_axes([1.03, .3, .03, .4])

for cluster_val in sorted(d3.Clusters.unique()):

    #print(cluster_val)

    sns.heatmap(d3.loc[d3.Clusters==cluster_val,:].groupby('Gender').agg({'Clusters':'size','Annual Income (k$)':'mean','Spending Score (1-100)':'mean'}).\

    rename(columns={'Clusters':'Count','Annual Income (k$)':'IncomeMean','Spending Score (1-100)':'SpendScoreMean'})\

                .fillna(0).round(),annot=True,fmt='g',ax=ax[cluster_val],cbar=i == 0,vmin=0, vmax=130)

    titl='Cluster-'+str(cluster_val)+' Analysis'

    ax[cluster_val].set_title(titl)

    



plt.suptitle('Clustering Gender Wise Analysis')



#plt.tight_layout()

plt.show()
plt.figure(figsize=(12,5))



sns.boxplot(x='Clusters',y='value',hue='variable',\

            data=pd.melt(d3,id_vars=['Clusters'],value_vars=['Annual Income (k$)','Spending Score (1-100)']),\

           palette="Set2")

plt.xlabel("Clusters")

plt.title("Boxplot-Annual Income - Spending Score")

plt.show()