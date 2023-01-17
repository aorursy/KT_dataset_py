import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set_style("ticks")
df=pd.read_csv("../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv")

df.head()
df.drop(["CustomerID"],axis=1,inplace=True)
df["Gender"]=pd.get_dummies(df["Gender"],drop_first=True)

#where 0=female and 1=male
df.info()
df.describe().T
df.isnull().sum()

#no missing data, which is good
df.head()
fig=plt.figure(figsize=(12,8))

sns.distplot(a=df["Spending Score (1-100)"],bins=40,color="#AE9CCD")

plt.title("Customer Spending Distribution",size=16)
fig=plt.figure(figsize=(12,8))



ax1=plt.subplot2grid((2,3),(0,0),colspan=3)

ax2=plt.subplot2grid((2,3),(1,0))

ax3=plt.subplot2grid((2,3),(1,1))

ax4=plt.subplot2grid((2,3),(1,2))



sns.boxplot(x=df["Spending Score (1-100)"],y=df["Gender"],orient="h",palette={0:"pink",1:"skyblue"},ax=ax1)



ax2.pie(df["Gender"].value_counts(),colors=["pink","skyblue"],labels=["Females","Males"])



sns.distplot(a=df["Age"].loc[(df["Gender"]==0)],bins=20,color="pink",ax=ax3)

sns.distplot(a=df["Age"].loc[(df["Gender"]==1)],bins=20,color="skyblue",ax=ax3)



sns.distplot(a=df["Annual Income (k$)"].loc[(df["Gender"]==0)],bins=20,color="pink",ax=ax4)

sns.distplot(a=df["Annual Income (k$)"].loc[(df["Gender"]==1)],bins=20,color="skyblue",ax=ax4)



plt.tight_layout(pad=1,h_pad=1,w_pad=1)

plt.suptitle("Gender Distribution",y=1.01,size=16)
fig=plt.figure(figsize=(12,7))



ax1=plt.subplot2grid((2,3),(0,0),colspan=2,rowspan=2)

ax2=plt.subplot2grid((2,3),(0,2))

ax3=plt.subplot2grid((2,3),(1,2))



sns.scatterplot(x=df["Age"],y=df["Spending Score (1-100)"],hue=df["Gender"],palette={0:"pink",1:"skyblue"},ax=ax1)



sns.distplot(a=df["Age"],bins=20,color="lightsalmon",ax=ax2)



sns.scatterplot(x=df["Age"],y=df["Annual Income (k$)"],color="lightgreen",ax=ax3)



plt.suptitle("Age Distribution",y=1.01,size=16)

plt.tight_layout(pad=1,h_pad=1,w_pad=1)
fig=plt.figure(figsize=(12,5))



ax1=fig.add_subplot(1,2,1)

ax2=fig.add_subplot(1,2,2)



sns.scatterplot(x=df["Annual Income (k$)"],y=df["Spending Score (1-100)"],color="#AE9CCD",ax=ax1)



sns.distplot(a=df["Annual Income (k$)"],bins=30,color="lightgreen",ax=ax2)



plt.suptitle("Annual Income Distribution",y=0.93,size=16)
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

scaled_df=pd.DataFrame(scaler.fit_transform(df),columns=df.columns)

scaled_df.head()
from sklearn.cluster import KMeans
kinertia=[]



for i in range(1,11):

    kmodel=KMeans(n_clusters=i)

    kmodel.fit(scaled_df.drop(["Gender"],axis=1))

    kinertia.append(kmodel.inertia_)



kinertia_df=pd.DataFrame({"No of Cluster":range(1,11),"Inertia":kinertia})



sns.lineplot(x=kinertia_df["No of Cluster"],y=kinertia_df["Inertia"],color="powderblue",linewidth=2,marker="s",markersize=8,markeredgecolor="whitesmoke",markeredgewidth=1)
from sklearn.metrics import silhouette_score
ksilhouette=[]



for j in range(2,11):

    kmodel=KMeans(n_clusters=j)

    kpredict=kmodel.fit_predict(scaled_df.drop(["Gender"],axis=1))

    kscores=silhouette_score(scaled_df.drop(["Gender"],axis=1),kpredict)

    ksilhouette.append(kscores)



silhouette_df=pd.DataFrame({"no_k":list(range(2,11)),"silhoutte":ksilhouette})



sns.barplot(x=silhouette_df["no_k"],y=silhouette_df["silhoutte"],color="mistyrose")

plt.xlabel("Number of Clusters")

plt.ylabel("Silhouette Score")
model=KMeans(n_clusters=6,random_state=7)

model.fit(scaled_df)

df["Cluster"]=model.labels_
cluster_df=df.groupby(["Cluster"]).mean().T

cluster_df["Average"]=df.apply(lambda x:x.mean(),axis=0)

cluster_df
cluster_df_graphing=cluster_df.drop(["Gender"])



fig,ax=plt.subplots(3,2,figsize=(12,6))



for i,ax in enumerate(ax.flat):

    sns.scatterplot(x=cluster_df_graphing["Average"],y=cluster_df_graphing.index,marker="X",color="k",ax=ax)

    sns.barplot(x=cluster_df_graphing[i],y=cluster_df_graphing.index,palette={"Age":"lightsalmon","Annual Income (k$)":"lightgreen","Spending Score (1-100)":"#AE9CCD"},ax=ax)

    ax.set_xlim(1,90)

    plt.setp(ax.collections,zorder=100)

    ax.set_xlabel("Cluster {}".format(i))

    

plt.tight_layout(pad=1,h_pad=1,w_pad=1)    
from sklearn.decomposition import PCA

pca=PCA(n_components=2)



pca_df=pd.DataFrame(pca.fit_transform(scaled_df),columns=["PC1","PC2"])

pca_df["Cluster"]=model.labels_



plt.figure(figsize=(12,8))

sns.scatterplot(x=pca_df["PC1"],y=pca_df["PC2"],hue=pca_df["Cluster"],palette="Pastel1")