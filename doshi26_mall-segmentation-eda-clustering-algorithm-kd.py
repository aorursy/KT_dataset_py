



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # pandas and matplotlib for data visualization

import seaborn as sns

%matplotlib inline

import matplotlib.animation as animation

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df= pd.read_csv('../input/Mall_Customers.csv')
df.head()
df.info()
# to check any 'null' values present in dataset

df.isnull().sum()
df=df.drop(['CustomerID'], axis=1)
fig,ax = plt.subplots(figsize=(8, 8))

sns.heatmap(df.corr(), ax=ax, annot=True, linewidths=0.05, fmt= '.2f',cmap="magma")

plt.show()
#The two important features which affects the sepnding score are Age and Annual Income
plt.style.use('fivethirtyeight')
plt.figure(1 , figsize = (15 , 6))

n = 0 

for x in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:

    n += 1

    plt.subplot(1 , 3 , n)

    plt.subplots_adjust(hspace =0.5 , wspace = 0.5)

    sns.distplot(df[x] , bins = 20)

    plt.title('Distplot of {}'.format(x))

plt.show()
plt.figure(1 , figsize = (15 , 5))

sns.countplot(y = 'Gender' , data = df)

plt.show()
labels = ['Female', 'Male']

size = df['Gender'].value_counts()

colors = ['red', 'orange']

explode = [0, 0.1]



plt.rcParams['figure.figsize'] = (4, 4)

plt.pie(size, colors = colors, explode = explode, labels = labels, shadow = True, autopct = '%.2f%%')

plt.title('Gender', fontsize = 20)

plt.axis('off')

plt.legend()

plt.show()
plt.figure(figsize=(6,6))

plt.scatter(df['Annual Income (k$)'],df['Spending Score (1-100)'])

plt.title('Annual Income (k$) vs Spending Score (1-100)')

plt.xlabel('Annual Income (k$)')

plt.ylabel('Spending Score (1-100)')
plt.figure(figsize=(6,6))

plt.scatter(df['Annual Income (k$)'],df['Age'])

plt.title('Annual Income (k$) vs Age')

plt.xlabel('Annual Income (k$)')

plt.ylabel('Age')
plt.figure(1 , figsize = (8, 6))

for gender in ['Male' , 'Female']:

    plt.scatter(x = 'Age' , y = 'Annual Income (k$)' , data = df[df['Gender'] == gender] ,

                s = 200 , alpha = 0.5 , label = gender)

plt.xlabel('Age'), plt.ylabel('Annual Income (k$)') 

plt.title('Age vs Annual Income w.r.t Gender')

plt.legend()

plt.show()
plt.figure(1 , figsize = (10 , 6))

for gender in ['Male' , 'Female']:

    plt.scatter(x = 'Annual Income (k$)',y = 'Spending Score (1-100)' ,

                data = df[df['Gender'] == gender] ,s = 200 , alpha = 0.5 , label = gender)

plt.xlabel('Annual Income (k$)'), plt.ylabel('Spending Score (1-100)') 

plt.title('Annual Income vs Spending Score w.r.t Gender')

plt.legend()

plt.show()
plt.figure(1 , figsize = (15 , 7))

n = 0 

for cols in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:

    n += 1 

    plt.subplot(1 , 3 , n)

    plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)

    sns.violinplot(x = cols , y = 'Gender' , data = df , palette = 'vlag')

    sns.swarmplot(x = cols , y = 'Gender' , data = df)

    plt.ylabel('Gender' if n == 1 else '')

    plt.title('Boxplots & Swarmplots' if n == 2 else '')

plt.show()
#Distribution of values in Age , Annual Income and Spending Score according to Gender
df.head()
### Feature sleection for the model

#Considering only 2 features (Annual income and Spending Score) and no Label available

X= df.iloc[:, [2,3]].values
#Building the Model

#KMeans Algorithm to decide the optimum cluster number , KMeans++ using Elbow Mmethod

#to figure out K for KMeans, I will use ELBOW Method on KMEANS++ Calculation

from sklearn.cluster import KMeans

wcss=[]



#we always assume the max number of cluster would be 10

#you can judge the number of clusters by doing averaging

###Static code to get max no of clusters



for i in range(1,11):

    kmeans = KMeans(n_clusters= i, init='k-means++', random_state=0)

    kmeans.fit(X)

    wcss.append(kmeans.inertia_)



    #inertia_ is the formula used to segregate the data points into clusters
#Visualizing the ELBOW method to get the optimal value of K 

plt.plot(range(1,11), wcss)

plt.title('The Elbow Method')

plt.xlabel('no of clusters')

plt.ylabel('wcss')

plt.show()
#If you zoom out this curve then you will see that last elbow comes at k=5

#no matter what range we select ex- (1,21) also i will see the same behaviour but if we chose higher range it is little difficult to visualize the ELBOW

#that is why we usually prefer range (1,11)

##Finally we got that k=5



#Model Build

kmeansmodel = KMeans(n_clusters= 5, init='k-means++', random_state=0)

y_kmeans= kmeansmodel.fit_predict(X)



#For unsupervised learning we use "fit_predict()" wherein for supervised learning we use "fit_tranform()"

#y_kmeans is the final model . Now how and where we will deploy this model in production is depends on what tool we are using.

#This use case is very common and it is used in BFS industry(credit card) and retail for customer segmenattion.
#Visualizing all the clusters 



plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')

plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')

plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')

plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')

plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')

plt.title('Clusters of customers')

plt.xlabel('Annual Income (k$)')

plt.ylabel('Spending Score (1-100)')

plt.legend()

plt.show()
###Model Interpretation 

#Cluster 1 (Red Color) -> earning high but spending less

#cluster 2 (Blue Colr) -> average in terms of earning and spending 

#cluster 3 (Green Color) -> earning high and also spending high [TARGET SET]

#cluster 4 (cyan Color) -> earning less but spending more

#Cluster 5 (magenta Color) -> Earning less , spending less
df.head()
### Feature sleection for the model

#Considering only 2 features (Annual income and Spending Score) and no Label available

x = df.iloc[:, [1, 3]].values

x.shape
from sklearn.cluster import KMeans



wcss = []

for i in range(1, 11):

    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

    kmeans.fit(x)

    wcss.append(kmeans.inertia_)



plt.rcParams['figure.figsize'] = (15, 5)

plt.plot(range(1, 11), wcss)

plt.title('K-Means Clustering(The Elbow Method)', fontsize = 20)

plt.xlabel('Age')

plt.ylabel('Count')

plt.show()
kmeans = KMeans(n_clusters = 4, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

ymeans = kmeans.fit_predict(x)



plt.rcParams['figure.figsize'] = (8, 8)

plt.title('Cluster of Ages', fontsize = 30)



plt.scatter(x[ymeans == 0, 0], x[ymeans == 0, 1], s = 100, c = 'pink', label = 'Usual Customers' )

plt.scatter(x[ymeans == 1, 0], x[ymeans == 1, 1], s = 100, c = 'orange', label = 'Priority Customers')

plt.scatter(x[ymeans == 2, 0], x[ymeans == 2, 1], s = 100, c = 'lightgreen', label = 'Target Customers(Young)')

plt.scatter(x[ymeans == 3, 0], x[ymeans == 3, 1], s = 100, c = 'red', label = 'Target Customers(Old)')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 50, c = 'black')



plt.xlabel('Age')

plt.ylabel('Spending Score (1-100)')

plt.legend()

plt.show()