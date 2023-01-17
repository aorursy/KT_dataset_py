# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import colorsys
import seaborn as sns
import warnings
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from kmodes.kprototypes import KPrototypes
import matplotlib.cm as cm

from sklearn.metrics import silhouette_samples, silhouette_score
df=pd.read_csv("../input/results.csv")

print("Data have number of row     :",df.shape[0])
print("Data have number of columns :",df.shape[1])
df.head(5)
df.columns
df.info()
numeric = ['10k','25k','age','official','35k','overall','pace','30k','5k','half','20k','40k']
df[numeric] = df[numeric].apply(pd.to_numeric, errors = 'coerce', axis=1)
print(df.dtypes)
df.describe()
plt.figure(figsize=(8,6))
hage = sns.distplot(df.age, color='g')
hage.set_xlabel('Ages',fontdict= {'size':14})
hage.set_ylabel(u'Distribution',fontdict= {'size':14})
hage.set_title(u'Distribution for Ages',fontsize=18)
plt.show()
list_feature_number=['5k','10k','20k','25k','30k','35k','40k','official','overall','pace','half','division']

fig,ax = plt.subplots(6,2, figsize=(12,12)) 
i=0 
for x in range(6):
    for y in range(2):
        sns.distplot(df[list_feature_number[i]], ax = ax[x,y])
        i+=1
plt.tight_layout()
plt.show()
feature_number = df.dtypes[df.dtypes != "object"].index
print(feature_number)
sns.pairplot(df, x_vars=["division"],y_vars=["official"],height=8, aspect=.8, kind="reg")
sns.pairplot(df, x_vars=["age"],y_vars=["official"],height=8, aspect=.8, kind="reg")
sns.pairplot(df, x_vars=["pace"],y_vars=["official"],height=8, aspect=.8, kind="reg")
feature_category = df.dtypes[df.dtypes == "object"].index
print(feature_category)
plt.figure(figsize=(8,6))
hage = sns.countplot(df.gender, palette={'F':'r','M':'b'})
hage.set_xlabel('Gender',fontdict= {'size':14})
hage.set_ylabel(u'Count',fontdict= {'size':14})
hage.set_title(u'count of Genner',fontsize=18)
plt.show()
plt.figure(figsize=(25,25))
d = sns.countplot(x='age', hue='gender', data=df, palette={'F':'r','M':'b'}, saturation=0.6)
d.set_title('Number of Finishers for Age and Gender', fontsize=25)
d.set_xlabel('Ages',fontdict={'size':20})
d.set_ylabel('Number of Finishers',fontdict={'size':20})
d.legend(fontsize=16)
plt.show()
sns.catplot(x='gender', y='age', data=df,height=8,kind='box')

df_1 = df.copy()
bins = [17, 25, 40,70,80, 90]
df_1['Ranges'] = pd.cut(df_1['age'],bins,labels=["18-25", "26-40", "40-70", "70-80","> 80"]) 

df_2 = pd.crosstab(df_1.Ranges,df_1['gender']).apply(lambda r: (r/r.sum()) * 100 , axis=1)

ax1 = df_2.plot(kind = "bar", stacked = True, color = ['r','b'], figsize=(9,6),
                      fontsize=12, position=0.5)
ax1.get_legend_handles_labels
ax1.legend(bbox_to_anchor = (1.3, 1))
ax1.set_xlabel('Age Ranges', fontdict={'size':14})
ax1.set_ylabel('Percentages (%)', fontdict={'size':14})
ax1.set_title('Gender x Age Ranges', fontsize=18)
plt.show()
df["country"].value_counts()
country_counts=df["country"].value_counts()<30
country_index=country_counts[country_counts==True].index
print(len(country_index))
print(country_index)
df["city"].value_counts()
df.isnull().sum()
df2=df.drop(columns=["ctz"])
feature_missing=["10k","25k","35k","state","30k","5k","20k","40k","city"]
row_missing=df2[df2.half.isnull()==True].index
for col in feature_missing:
    row_missing=row_missing.append(df2[df2[col].isnull()==True].index)
df2=df2.drop(list(set(row_missing)))
df2.shape
df2.isnull().sum().sum()

df2_corr=df2.corr()

mask = np.zeros_like(df2_corr)
mask[np.triu_indices_from(mask)] = True
df2_corr=df2_corr[df2_corr > 0.95]
with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(20, 12))
    ax = sns.heatmap(df2_corr, mask=mask ,vmax=1, square=True,annot=True)

df2=df2.drop(columns=["5k","10k","20k","25k","30k","35k","40k","half"])
df2_corr=df2.corr()

mask = np.zeros_like(df2_corr)
mask[np.triu_indices_from(mask)] = True
df2_corr=df2_corr[df2_corr > 0.9]
with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(20, 12))
    ax = sns.heatmap(df2_corr, mask=mask ,vmax=1, square=True,annot=True)
df2=df2.drop(columns=["pace","overall"])
df2_corr=df2.corr()

mask = np.zeros_like(df2_corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(20, 12))
    ax = sns.heatmap(df2_corr, mask=mask ,vmax=1, square=True,annot=True)
df2.shape
df2.head(5)
df2["bib"].value_counts()
df2=df2.drop(columns=["name","bib","city"])
df_number=pd.get_dummies(df2)
df_number.head(5)
scaler=MinMaxScaler(feature_range=(0, 1), copy=True)

X=scaler.fit_transform(df_number)
print(X)
# Using the elbow method to find the optimal number of clusters

wcss = []
for i in range(1, 12):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    # inertia method returns wcss for that model
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(10,5))
sns.lineplot(range(1, 12), wcss,marker='o',color='red')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)
# Visualising the clusters
plt.figure(figsize=(15,7))
sns.scatterplot(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], color = 'yellow', label = 'Cluster 1',s=50)
sns.scatterplot(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], color = 'blue', label = 'Cluster 2',s=50)
sns.scatterplot(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], color = 'green', label = 'Cluster 3',s=50)
sns.scatterplot(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], color = 'grey', label = 'Cluster 4',s=50)
sns.scatterplot(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], color = 'orange', label = 'Cluster 5',s=50)
sns.scatterplot(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color = 'red', 
                label = 'Centroids',s=300,marker=',')
plt.grid(False)
plt.title('Clusters')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

#Calculate the average of silhouette scores
silhouette_avg = silhouette_score(X,y_kmeans)

#Calculate the silhouette score for each data
each_silhouette_score = silhouette_samples(X,y_kmeans,metric="euclidean")
print(silhouette_avg)
fig =plt.figure()
ax = fig.add_subplot(1,1,1)
y_lower =10
n_clusters=5
colorlist =["tomato","antiquewhite","blueviolet","cornflowerblue","darkgreen","seashell","skyblue","mediumseagreen"]

for i in range(n_clusters):
    ith_cluster_silhouette_values = each_silhouette_score[y_kmeans == i]
    ith_cluster_silhouette_values.sort()
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i
    
    color = colorlist[i]
    ax.fill_betweenx(np.arange(y_lower,y_upper),0,ith_cluster_silhouette_values,facecolor=color,edgecolor=color,alpha=0.3)
    
    #label the silhouse plots with their cluster numbers at the middle
    ax.text(-0.05,y_lower + 0.5 * size_cluster_i,str(i))
    
    #compute the new y_lower for next plot
    y_lower = y_upper +10 
    
ax.set_title("Silhuoette plot")
ax.set_xlabel("silhouette score")
ax.set_ylabel("Cluster label")
    
#the vertical line for average silhouette score of all the values
ax.axvline(x=silhouette_avg,color="red",linestyle="--")
    
ax.set_yticks([])
ax.set_xticks([-0.2,0,0.2,0.4,0.6,0.8,1])
cols=list(df_number.columns)
print(cols)
# Silhouette score

for n_clusters in range(2,11):
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

plt.show()
df2.head(5)
X_Num=df2.drop(columns=["gender","state","country"])
X_Cat=df2.drop(columns=["division","age","official","genderdiv"])
X_Num.head(5)
X_Cat.head(5)
# Running K-Prototype clustering
kproto = KPrototypes(n_clusters=3, init='Huang', verbose=0, random_state=42,max_iter=20, n_init=50,n_jobs=-2,gamma=.25) 
y_kprototypes = kproto.fit_predict(df2, categorical=[1,5,6])
labels=kproto.labels_
silScoreKmeans=silhouette_score(X_Num,labels,metric="euclidean")
print("Silhouette Score with number : ",silScoreKmeans)
#silScoreKmodes=silhouette_score(X_Cat,labels,metric="hamming")
#print("Silhouette Score with category : ",silScoreKmodes)
#print("Silhouette Score",(silScoreKmeans+silScoreKmodes)/2)
scaler=LabelEncoder()
X_Cat_labelendcoding=X_Cat.copy()
X_Cat_labelendcoding["gender"]=scaler.fit_transform(X_Cat["gender"])
X_Cat_labelendcoding["state"]=scaler.fit_transform(X_Cat["state"])
X_Cat_labelendcoding["country"]=scaler.fit_transform(X_Cat["country"])
X_Cat_labelendcoding.head(5)
X_Cat_onehot=pd.get_dummies(X_Cat)
X_Cat_onehot.head(5)
kproto1 = KPrototypes(n_clusters=3, init='Cao', verbose=0, random_state=42,max_iter=20, n_init=50,n_jobs=-2,gamma=.25) 
y_kprototypes1 = kproto.fit_predict(df2, categorical=[1,5,6])
print(y_kprototypes1)
labels1=y_kprototypes1

silScoreKmeans1=silhouette_score(X_Num,labels1,metric="euclidean")
print("Silhouette Score with number : ",silScoreKmeans)
#silScoreKmodes1=silhouette_score(X_labelendcoding,labels1,metric="euclidean")
#print("Silhouette Score with category : ",silScoreKmodes)
#print("Silhouette Score",(silScoreKmeans+silScoreKmodes)/2)
#Visualize K-Prototype clustering on the PCA projected Data
df3=df2.copy()
df3['Cluster_id']=y_kprototypes
print(df3['Cluster_id'].value_counts())
sns.pairplot(df3,hue='Cluster_id',palette='Dark2',diag_kind='kde')
plt.subplots(figsize = (15,5))
sns.countplot(x=df2['age'],order=df2['age'].value_counts().index,hue=df3["Cluster_id"])
plt.show()
df_3 = df3.copy()
bins = [17, 25, 40,70,80, 90]
df_3['group_age'] = pd.cut(df_3['age'],bins,labels=["18-25", "26-40", "40-70", "70-80","> 80"]) 
plt.subplots(figsize = (15,5))
sns.countplot(x=df_3['group_age'],order=df_3['group_age'].value_counts().index,hue=df3["Cluster_id"])
plt.show()
plt.subplots(figsize = (5,5))
sns.countplot(x=df3['gender'],order=df3['gender'].value_counts().index,hue=df3['Cluster_id'])
plt.show()