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
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
import scipy.cluster.hierarchy as sch
mall=mall = pd.read_csv("../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv")
mall.head()
print(mall.shape)
mall.describe()
# Distributions - Age, Income and Spending Score

# General fig
fig, axs =plt.subplots(2,2,figsize=(10,10))

#Age
axs[0,0].hist(mall['Age'],bins=15,color='#e6a340')
axs[0,0].set_title('Age Histogram')
axs[0,0].set_xlabel('Age')
axs[0,0].set_ylabel('Freq')

# Income
axs[0,1].hist(mall["Annual Income (k$)"],bins=15,color='#8be640')
axs[0,1].set_title('Income Histogram')
axs[0,1].set_xlabel('Income')
axs[0,1].set_ylabel('Freq')

# Spending
axs[1,0].hist(mall["Spending Score (1-100)"],bins=15,color='#40a1e6')
axs[1,0].set_title('Spending Histogram')
axs[1,0].set_xlabel('Spending')
axs[1,0].set_ylabel('Freq')

# Gender
# First creating a table to summarize information
gender_sum=mall.groupby(['Gender']).Gender.count().to_frame('Count').reset_index()
axs[1,1].pie(gender_sum.Count,labels=gender_sum.Gender,autopct='%1.1f%%',colors=('#e64040','#40a1e6'))
axs[1,1].set_title('Gender')

# General Title
fig.suptitle('Variables Distribution',size=20)
plt.show()
# The big figure
fig, axs =plt.subplots(3,2,figsize=(10,17))

# Age
# Age - histogram by gender
metric="Age"
axs[0,0].hist(mall.query("Gender == 'Male'")[metric],alpha=0.4, bins=20, label='M',color='#40a1e6')
axs[0,0].hist(mall.query("Gender == 'Female'")[metric],alpha=0.4, bins=20,label='F',color='#e64040')
axs[0,0].set_title('Age Histogram by Gender')
axs[0,0].set_xlabel(metric)
axs[0,0].set_ylabel('Freq')
axs[0,0].legend()

# Age - Boxplot by gender
age_boxplot=[mall.query("Gender == 'Male'")[metric],
             mall.query("Gender == 'Female'")[metric]]
axs[0,1].boxplot(age_boxplot)
axs[0,1].set_xticklabels(['Male','Female'])
axs[0,1].set_ylabel(metric)
axs[0,1].set_title('Age Boxplot by Gender')


#############################################################
# Income
#Income - histogram by gender
metric="Annual Income (k$)"
axs[1,0].hist(mall.query("Gender == 'Male'")[metric],alpha=0.4, bins=20, label='M',color='#40a1e6')
axs[1,0].hist(mall.query("Gender == 'Female'")[metric],alpha=0.4, bins=20,label='F',color='#e64040')
axs[1,0].set_title('Income Histogram by Gender')
axs[1,0].set_xlabel(metric)
axs[1,0].set_ylabel('Freq')
axs[1,0].legend()

# Income - boxplot by gender
income_boxplot=[mall.query("Gender == 'Male'")[metric],
             mall.query("Gender == 'Female'")[metric]]
axs[1,1].boxplot(income_boxplot)
axs[1,1].set_xticklabels(['Male','Female'])
axs[1,1].set_ylabel(metric)
axs[1,1].set_title('Income Boxplot by Gender')
##############################################################
# Spending Score
# Spending Score - histogram by gender
metric="Spending Score (1-100)"
axs[2,0].hist(mall.query("Gender == 'Male'")[metric],alpha=0.4, bins=20, label='M',color='#40a1e6')
axs[2,0].hist(mall.query("Gender == 'Female'")[metric],alpha=0.4, bins=20,label='F',color='#e64040')
axs[2,0].set_title('Spending Histogram by Gender')
axs[2,0].set_xlabel(metric)
axs[2,0].set_ylabel('Freq')
axs[2,0].legend()

# Spending Score - boxplot by gender
spend_boxplot=[mall.query("Gender == 'Male'")[metric],
             mall.query("Gender == 'Female'")[metric]]
axs[2,1].boxplot(spend_boxplot)
axs[2,1].set_xticklabels(['Male','Female'])
axs[2,1].set_ylabel(metric)
axs[2,1].set_title('Spend Boxplot by Gender')

#plot the figure
plt.show()
sns.pairplot(mall,kind='scatter',hue='Gender',palette=('#40a1e6','#e64040'))
plt.show()
# First, just guessing a number of clusters

# Including gender as a dummy variable
mall['Gender_2']=mall['Gender'].apply(lambda x: 1 if x=='Male' else 0)

#running kmeans with 3 clusters
variables=mall[['Age','Annual Income (k$)','Spending Score (1-100)','Gender_2']]
model_kmeans=KMeans(n_clusters=3)
model_kmeans.fit(variables)
# getting a 2D approximation for the clusters
# plot the clusters
tsne=TSNE()
visualization=tsne.fit_transform(variables)
sns.set(rc={'figure.figsize':(5,5)})
sns.scatterplot(x=visualization[:,0],y=visualization[:,1],
               hue=model_kmeans.labels_,
               palette=sns.color_palette('Set1',3))
plt.show()
# Figuring out the optimal number of clusters
# The idea will be running KMeans with different numbers of clusters
# and compute the error associated with it
# We will be using the elbow method to determine the exact number of clusters

# First define a function that returns the number of clusters and error

def k_means_elbow(n_clust,variables):
    model_kmeans=KMeans(n_clusters=n_clust)
    model_kmeans.fit(variables)
    return [n_clust,model_kmeans.inertia_]
#plot Inertia by number of clusters
elbow=[k_means_elbow(n_cluster,variables) for n_cluster in range (1,20)]
elbow=pd.DataFrame(elbow,columns=['n_clusters','Inertia'])
plt.figure(figsize=(5,5))
elbow['Inertia'].plot()
plt.title('Inertia by Number of Clusters',size=15)
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()
# repeat the previous process again but with n_cluters=5
variables=mall[['Age','Annual Income (k$)','Spending Score (1-100)','Gender_2']]
n_clust=6
model_kmeans=KMeans(n_clusters=n_clust)
model_kmeans.fit(variables)

# Including the cluster classification in the dataframe
predict=model_kmeans.predict(variables)
mall['kmeans_6']=pd.Series(predict,index=mall.index)

tsne=TSNE()
visualization=tsne.fit_transform(variables)
sns.set(rc={'figure.figsize':(5,5)})
sns.scatterplot(x=visualization[:,0],y=visualization[:,1],
               hue=model_kmeans.labels_,
               palette=sns.color_palette('Set1',n_clust))
plt.show()
dendrogram=sch.dendrogram(sch.linkage(variables,method='ward'))
variables=mall[['Age','Annual Income (k$)','Spending Score (1-100)','Gender_2']]
n_clust=6
model_HC=AgglomerativeClustering(n_clusters=n_clust, affinity='euclidean',linkage='ward')
groups_HC=model_HC.fit_predict(variables)
groups_HC
mall['HC_6']=pd.Series(groups_HC,index=mall.index)
tsne=TSNE()
visualization=tsne.fit_transform(variables)
sns.set(rc={'figure.figsize':(5,5)})
sns.scatterplot(x=visualization[:,0],y=visualization[:,1],
               hue=model_HC.labels_,
               palette=sns.color_palette('Set1',n_clust))
plt.show()
fig, axs =plt.subplots(1,2,figsize=(10,5))

#kmeans plot
a=axs[0].scatter(x=mall["Annual Income (k$)"],y=mall["Spending Score (1-100)"],c=mall['kmeans_6'])
axs[0].legend(*a.legend_elements())
axs[0].set_title('KMeans 6 Clusters')
axs[0].set_xlabel('Income')
axs[0].set_ylabel('Spending')

#HC plot
b=axs[1].scatter(x=mall["Annual Income (k$)"],y=mall["Spending Score (1-100)"],c=mall['HC_6'])
axs[1].legend(*b.legend_elements())
axs[1].set_title('Hierarchical 6 Clusters')
axs[1].set_xlabel('Income')
axs[1].set_ylabel('Spending')

plt.show()
#Pair plot
sns.pairplot(mall[['Age','Annual Income (k$)','Spending Score (1-100)','kmeans_6']],kind='scatter',hue='kmeans_6')
plt.show()
variables=mall[['Age','Annual Income (k$)','Spending Score (1-100)','Gender_2']]
summary_centroids=pd.DataFrame(model_kmeans.cluster_centers_,columns=variables.columns)
summary_centroids