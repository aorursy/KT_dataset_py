# Import libray

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
import pandas as pd

data = pd.read_csv("../input/Mall_Customers.csv")
data.head()

# We'll see the 1st 5 lines.
data[['Genre','Age']].head()
data.tail()

# We'll see the last 5 lines
data.sample(10)

# Random data
data.describe()

# It is the function that show the analysis of numerical values.
data.dtypes

# It shows the data type in the data set.
data.columns

#Show data's columns
#Rename data's columns

data.rename(columns={'Genre':'Gender','Annual Income (k$)':'AnnualIncome','Spending Score (1-100)':'SpendingScore'},inplace=True)

data.head()
data.info()

# Shows the property value in the data set.
data.shape

# data row and columns count
data.isnull().sum()

# Count null values
data[data['Gender'].isnull()]
data.shape
data.dropna(how='all').shape
data.dropna(how='any').shape
data.isnull().sum()
print(list(data.isnull().any()))

#Every feature control check null value in this data
#data control null values

data.isnull().values.any()
# Show data gender unique

data.Gender.unique()
# Show gender value counts

data.Gender.value_counts()
# Show graph data gender

sns.countplot(data.Gender)

plt.title('Gender Count')

plt.show()
#Linear relationship among the independent attributs

sns.scatterplot(x="AnnualIncome" , y = "SpendingScore",data = data)
data.AnnualIncome.fillna(data.AnnualIncome.mean(),inplace  = True) 
data.isnull().sum()
data.SpendingScore.fillna(data.SpendingScore .mean(),inplace  = True)
data.isnull().sum()
data['Gender'].astype(str)

data.dtypes
data1=data.dropna()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder

le_gend=LabelEncoder()

data1.Gender=le_gend.fit_transform(data1.Gender)
data1.head(50)

data1.shape
data1.isnull().sum()
#data correlation

data1.corr()
data1.iloc[:,1:].corr()
sns.heatmap(data1.corr(),annot=True,fmt='.1f')

plt.show()
#need to drop customerID

data1.drop('CustomerID',axis=1,inplace=True)
#show data gender unique

data1.Gender.unique()
#show gender value counts

data1.Gender.value_counts()
plt.figure(figsize=(15,6))

plt.subplot(1,2,1)

sns.boxplot(y=data1["SpendingScore"], color="yellow")

plt.subplot(1,2,2)

sns.boxplot(y=data1["AnnualIncome"])

plt.show()
ss1_20 = data1["SpendingScore"][(data1["SpendingScore"] >= 1) & (data1["SpendingScore"] <= 20)]

ss21_40 = data1["SpendingScore"][(data1["SpendingScore"] >= 21) & (data1["SpendingScore"] <= 40)]

ss41_60 = data1["SpendingScore"][(data1["SpendingScore"] >= 41) & (data1["SpendingScore"] <= 60)]

ss61_80 = data1["SpendingScore"][(data1["SpendingScore"] >= 61) & (data1["SpendingScore"] <= 80)]

ss81_100 = data1["SpendingScore"][(data1["SpendingScore"] >= 81) & (data1["SpendingScore"] <= 100)]



ssx = ["1-20", "21-40", "41-60", "61-80", "81-100"]

ssy = [len(ss1_20.values), len(ss21_40.values), len(ss41_60.values), len(ss61_80.values), len(ss81_100.values)]



plt.figure(figsize=(15,6))

sns.barplot(x=ssx, y=ssy, palette="twilight")

plt.title("SpendingScores")

plt.xlabel("Score")

plt.ylabel("Number of Customer Having the Score")

plt.show()
ai0_30 = data1["AnnualIncome"][(data1["AnnualIncome"] >= 0) & (data1["AnnualIncome"] <= 30)]

ai31_60 = data1["AnnualIncome"][(data1["AnnualIncome"] >= 31) & (data1["AnnualIncome"] <= 60)]

ai61_90 = data1["AnnualIncome"][(data1["AnnualIncome"] >= 61) & (data1["AnnualIncome"] <= 90)]

ai91_120 = data1["AnnualIncome"][(data1["AnnualIncome"] >= 91) & (data1["AnnualIncome"] <= 120)]

ai121_150 = data1["AnnualIncome"][(data1["AnnualIncome"] >= 121) & (data1["AnnualIncome"] <= 150)]



aix = ["$ 0 - 30,000", "$ 30,001 - 60,000", "$ 60,001 - 90,000", "$ 90,001 - 120,000", "$ 120,001 - 150,000"]

aiy = [len(ai0_30.values), len(ai31_60.values), len(ai61_90.values), len(ai91_120.values), len(ai121_150.values)]



plt.figure(figsize=(15,6))

sns.barplot(x=aix, y=aiy, palette="rainbow")

plt.title("Annual Incomes")

plt.xlabel("Income")

plt.ylabel("Number of Customer")

plt.show()
# Show graph data gender

age18_25 = data1.Age[(data1.Age <= 25) & (data1.Age >= 18)]

age26_35 = data1.Age[(data1.Age <= 35) & (data1.Age >= 26)]

age36_45 = data1.Age[(data1.Age <= 45) & (data1.Age >= 36)]

age46_55 = data1.Age[(data1.Age <= 55) & (data1.Age >= 46)]

age55above = data1.Age[data1.Age >= 56]



x = ["18-25","26-35","36-45","46-55","55+"]

y = [len(age18_25.values),len(age26_35.values),len(age36_45.values),len(age46_55.values),len(age55above.values)]



plt.figure(figsize=(15,6))

sns.barplot(x=x, y=y, palette="plasma")

plt.title("Number of Customer and Ages")

plt.xlabel("Age")

plt.ylabel("Number of Customer")

plt.show()
#show graph data gender

sns.countplot(data1.Gender)

plt.title('Gender Count')

plt.show()
labels=data1.Gender.unique()

colors=['gray','red']

explode=[0,0.1]

values=data1.Gender.value_counts().values



#visualization

plt.figure(figsize=(7,7))

plt.pie(values,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%')

plt.title('Race/Ethnicity According Analysis',color='black',fontsize=10)

plt.show()
sns.boxplot(x="Gender",y = "SpendingScore", data=data1)
plt.figure(figsize=(25,10))

sns.barplot(x = "AnnualIncome", y = "SpendingScore", hue = "Gender", data = data1)

plt.xticks(rotation=45)

plt.show()
data1.hist(figsize=(18,12))

plt.title('All Data Show Histogram System')

plt.show()
data1.Age.unique()
print(len(data1.Age.unique()))
data1.Age.value_counts()[:10]
plt.figure(figsize=(15,7))

sns.barplot(x=data1.Age.value_counts().index,y=data1.Age.value_counts().values)

plt.xlabel('Age')

plt.ylabel('Rate')

plt.title('Age vs Rate State')

plt.show()
pd.plotting.scatter_matrix(data1,figsize=(10,10))

plt.figure()

plt.show()
sns.violinplot(x=data1['AnnualIncome'],y=data1['Gender'])

plt.title('AnnualIncome & Gender')

plt.show()
pd.crosstab(index=data1["Age"],columns=data1["Gender"],normalize="index")
sns.violinplot(y=data1['SpendingScore'],x=data1['Gender'])

plt.title('SpendingScore & Gender')

plt.show()
plt.figure(figsize=(10,8))

for gender in data1.Gender.unique():

    plt.scatter(x='Age',y='SpendingScore',data=data1[data1['Gender']==gender],s=100,alpha=.7)

    plt.xlabel('Age')

    plt.ylabel('SpendingScore')

    plt.title('Age & SpendingScore')

plt.show()
plt.figure(figsize=(10,8))

for gender in data1.Gender.unique():

    plt.scatter(x='Age',y='AnnualIncome',data=data1[data1['Gender']==gender],s=100,alpha=.7)

    plt.xlabel('Age')

    plt.ylabel('AnnualIncome')

    plt.title('Age & AnnualIncome')

plt.show()
plt.figure(figsize=(10,8))

for gender in data1.Gender.unique():

    plt.scatter(x='AnnualIncome',y='SpendingScore',data=data1[data1['Gender']==gender],s=100,alpha=.7)

    plt.xlabel('AnnualIncome')

    plt.ylabel('SpendingScore')

    plt.title('AnnualIncome & SpendingScore')

plt.show()
sns.pairplot(data1[['Age','AnnualIncome','SpendingScore']])

plt.show()

sns.factorplot(x="Gender", y="AnnualIncome", kind='violin',data=data1)

plt.show()
pd.plotting.scatter_matrix(data1,figsize=(10,10))

plt.figure()
data.info()
######new scaling data

import   sklearn.preprocessing  as StandardScaler

features = ['Age','AnnualIncome','SpendingScore']

data2= data1.loc[:, features];data2
#KDE Plot described as Kernel Density Estimate is used for visualizing the Probability Density of a continuous variable



sns.pairplot(data1, size = 6, aspect = 0.5, diag_kind = 'kde')   # Kernel Density Estimation plot



plt.show()
from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

normaliser= preprocessing.Normalizer()

df_normalised=normaliser.fit_transform(data2);df_normalised
from sklearn.decomposition import PCA

from sklearn import preprocessing
pca=PCA().fit(data2)

pca_data = pca.transform(df_normalised)
plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.show()
# Eigen Values or the variance determined by each PCs



pca.explained_variance_ratio_
percent_variance=np.round(pca.explained_variance_ratio_*100, decimals=2)

columns=['PC1','PC2','PC3']

y = np.arange(len(columns))

plt.bar(columns, percent_variance)

plt.ylabel('Percentage of Variance')

plt.xlabel('Principal Components')

plt.title('PCA Scree Plot')

plt.show()
###selecting pcs which have significant effect & data transforming 

pca_new=PCA(n_components=2)

prin_comps=pca_new.fit_transform(df_normalised);prin_comps
updated_pcs=pd.DataFrame(data=prin_comps,columns=['PC1','PC2'])

updated_df=pd.concat([updated_pcs,data1['Gender']], axis=1);updated_df
dataset_standardized = preprocessing.scale(updated_pcs);dataset_standardized
# Find the appropriate cluster number



plt.figure(figsize=(10,8))

from sklearn.cluster import KMeans



wcss=[]

for i in range(2,11):

    kmeans=KMeans(n_clusters=i,init="k-means++",random_state=42)

    kmeans.fit(dataset_standardized )

    wcss.append(kmeans.inertia_)

    

plt.plot(range(2,11),wcss)

plt.title("the Elbow Method")

plt.xlabel("Number of clusters")

plt.ylabel("wcss")

plt.show()
from sklearn.metrics import silhouette_score



for i in range(2,11):

    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state=99)

    pred = kmeans.fit_predict(dataset_standardized)

    score = silhouette_score (dataset_standardized, pred, metric='euclidean')

    wcss = kmeans.inertia_

    print('For n_clusters = {}, silhouette score is {} and WCSS is {}'.format(i, score, wcss))
####Clustering

#It is time to cluster the data so that we can extract information from them related to the customer annual spending behaviors.

####K-Means

#I will run K-Means starting from k=2 to k=10. I will collect the silhouette scores for each of the results. So that I can determine the best number of clusters.

from matplotlib.colors import ListedColormap, LinearSegmentedColormap



cmap=LinearSegmentedColormap.from_list('BlRd',['blue','red','cyan'])



silhouette_scores=[]

for i in range(2,11):

    cl=KMeans(n_clusters=i,random_state=0)

    result=cl.fit_predict(updated_pcs)

    silhouette=silhouette_score(updated_pcs ,result)

    silhouette_scores.append(silhouette)

    plt.subplot(5,2,i-1)

    plt.scatter(updated_pcs.PC1.values,updated_pcs.PC2.values,c=result,cmap=cmap)

    plt.title(str(i)+' Clusters, Silhouette score :'+ str(silhouette)[:5])

    fig,ax=plt.gcf(),plt.gca()

    fig.set_size_inches(10,10)

    plt.tight_layout()

plt.show()
km = KMeans(n_clusters=5)

clusters = km.fit_predict(data2.iloc[:,1:])

data2["label"] = clusters



from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

 

fig = plt.figure(figsize=(20,10))

ax = fig.add_subplot(111, projection='3d')

ax.scatter(data2.Age[data2.label == 0], data2["AnnualIncome"][data2.label == 0], data2["SpendingScore"][data2.label == 0], c='blue', s=60)

ax.scatter(data2.Age[data2.label == 1], data2["AnnualIncome"][data2.label == 1], data2["SpendingScore"][data2.label == 1], c='red', s=60)

ax.scatter(data2.Age[data2.label == 2], data2["AnnualIncome"][data2.label == 2], data2["SpendingScore"][data2.label == 2], c='green', s=60)

ax.scatter(data2.Age[data2.label == 3], data2["AnnualIncome"][data2.label == 3], data2["SpendingScore"][data2.label == 3], c='orange', s=60)

ax.scatter(data2.Age[data2.label == 4], data2["AnnualIncome"][data2.label == 4], data2["SpendingScore"][data2.label == 4], c='purple', s=60)

ax.view_init(30, 185)

plt.xlabel("Age")

plt.ylabel("Annual Income (k$)")

ax.set_zlabel('Spending Score (1-100)')

plt.show()
##Here are the results of running hierarchical clustering on the data set. I will try all linkage methods possible to see the differences. Then I will plot dendrograms and clusters side by side.

from scipy.cluster.hierarchy import linkage, dendrogram

from scipy.cluster.hierarchy import fcluster



methods=['ward','single','complete','average','weighted','centroid','median']



plot_id=0

for method in methods:

    cl=linkage(updated_pcs,method=method)

    

    for sw in ['dendrogram','clusters']:

        if sw=='dendrogram':

            plot_id+=1

            plt.subplot(7,2,plot_id)

            plt.title(method)

            fig,ax=plt.gcf(),plt.gca()

            dn=dendrogram(cl,truncate_mode='level',p=15)

            plt.tight_layout()

            fig.set_size_inches(10,15)

        else:

            plot_id+=1

            labels=fcluster(cl,2,criterion='maxclust')

            plt.subplot(7,2,plot_id)

            plt.title(method)

            plt.scatter(updated_pcs.PC1.values.tolist(),

                       updated_pcs.PC2.values.tolist(),

                       cmap=cmap,

                       c=labels)

plt.show()   
cl=linkage(updated_pcs,method='ward')

fig,ax=plt.gcf(),plt.gca()

dn=dendrogram(cl,truncate_mode='level',p=15)

plt.tight_layout()

fig.set_size_inches(10,8)

plt.axhline(y=8,c='k')

plt.axhline(y=12,c='k')

plt.show()
##This maximum of 20 seems to be a good distance for clustering. Doing so, we should have 6 clusters. I am saving the plot:
cl=linkage(updated_pcs,method='ward')

labels=fcluster(cl,6,criterion='maxclust')

plt.scatter(updated_pcs.PC1.values.tolist(),

           updated_pcs.PC2.values.tolist(),

           cmap=cmap,

           c=labels)

plt.show()

#plt.savefig('img/hierarchical_fav.png')
cl=linkage(updated_pcs,method='weighted')

labels=fcluster(cl,6,criterion='maxclust')

plt.scatter(updated_pcs.PC1.values.tolist(),

           updated_pcs.PC2.values.tolist(),

           cmap=cmap,

           c=labels)

plt.show()
# Find the appropriate cluster number



plt.figure(figsize=(10,8))

from sklearn.cluster import KMeans



wcss=[]

for i in range(2,11):

    kmeans=KMeans(n_clusters=i,init="k-means++",random_state=42)

    kmeans.fit(data1)

    wcss.append(kmeans.inertia_)

    

plt.plot(range(2,11),wcss)

plt.title("the Elbow Method")

plt.xlabel("Number of clusters")

plt.ylabel("wcss")

plt.show()
from sklearn.metrics import silhouette_score



for i in range(2,11):

    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state=99)

    pred = kmeans.fit_predict(data1)

    score = silhouette_score (data1, pred, metric='euclidean')

    wcss = kmeans.inertia_

    print('For n_clusters = {}, silhouette score is {} and WCSS is {}'.format(i, score, wcss))
####hierarchical.

import scipy.cluster.hierarchy as shc

plt.figure(figsize=(10, 7))  

plt.title("Dendrograms")  

dend = shc.dendrogram(shc.linkage(data1, method='ward'))