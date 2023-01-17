import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.mixture import GaussianMixture

from sklearn import metrics

from sklearn.cluster import KMeans
file_path = '../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv'

mail_customers = pd.read_csv(file_path)
mail_customers.head()
# check whether there is any null value in each column

mail_customers.isnull().any()
# Describe the data

mail_customers.describe()
male_percentage = round(len(mail_customers.Gender[mail_customers.Gender == 'Male'])/len(mail_customers.Gender)*100,2)

female_percentage = round(len(mail_customers.Gender[mail_customers.Gender == 'Female'])/len(mail_customers.Gender)*100,2)

list = [male_percentage,female_percentage]

plt.figure(figsize=(6,6))

plt.pie(list,labels = ['Male','Female'],autopct='%2.1f%%',shadow=True,explode = [0.05,0.05])

plt.title('Customer gender ratio')

plt.legend(loc="upper right")
plt.figure(figsize=(10,6))

sns.distplot(a = mail_customers['Age'], color = 'red')

plt.title('Distribution of Age', fontsize = 15)

plt.xlabel('Range of Age')

plt.ylabel('Count')
bins = [min(mail_customers.Age)-1, 20, 30, 40, 50, 60, 70, max(mail_customers.Age)+1]

labels = ['below 20','20-30','30-40','40-50','50-60','60-70','above 70']

mail_customers['Age group'] = pd.cut(mail_customers.Age,bins,labels = labels) 
aggResult1 = mail_customers.groupby(by=['Age group'])['Age group'].count()

sns.set(style="whitegrid")

plt.figure(figsize=(10,6))

sns.barplot(x = aggResult1.index,y = aggResult1).set_ylabel('Number of People')

plt.figure(figsize=(10,6))

sns.distplot(a = mail_customers['Annual Income (k$)'])

plt.title('Distribution of Annual Income', fontsize = 15)

plt.xlabel('Range of Annual Income')

plt.ylabel('Count')


plt.figure(figsize=(10,6))

sns.boxplot(x='Age group',y='Annual Income (k$)',hue='Gender',data=mail_customers,palette='pastel')

sns.pairplot(mail_customers.iloc[:,1:5],hue='Gender')
column = ['Age','Annual Income (k$)','Spending Score (1-100)']

data = mail_customers.loc[0:,column]

plt.figure(figsize=(14,6))

sns.heatmap(data=data.corr(),cmap='viridis', annot=True)
silhouette_all=[]

for k in range(2,11):

    kmeans_model = KMeans(n_clusters=k, random_state=1).fit(data)

    labels = kmeans_model.labels_

    a = metrics.silhouette_score(data, labels, metric='euclidean')

    silhouette_all.append(a)

    #print(a)

    print('This is the silhouette score when k equals',k,': ',a)
plt.figure(figsize=(10,6))

plt.plot(range(2,11), silhouette_all, marker='o')

plt.xlabel('Number of clusters')

plt.ylabel('silhouette score')

plt.annotate('max score', xy=(6, 0.4523443947724053),arrowprops=dict(facecolor='black'))
km = KMeans(n_clusters = 6, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

y_means = km.fit_predict(data)

plt.figure(figsize=(10,6))

sns.countplot(y_means)
x = data.values

fig = plt.figure(figsize = (10,10))

fig = fig.add_subplot(111, projection='3d')

plt.scatter(x[y_means == 0,0], x[y_means == 0,1], x[y_means == 0,2], c = 'green')

plt.scatter(x[y_means == 1,0], x[y_means == 1,1],x[y_means == 1,2], c = 'yellow')

plt.scatter(x[y_means == 2, 0], x[y_means == 2, 1], x[y_means == 2,2], c = 'cyan')

plt.scatter(x[y_means == 3, 0], x[y_means == 3, 1], x[y_means == 3,2],c = 'magenta')

plt.scatter(x[y_means == 4, 0], x[y_means == 4, 1], x[y_means == 4,2],c = 'orange')

plt.scatter(x[y_means == 5, 0], x[y_means == 5, 1],x[y_means == 5,2], c = 'red')

plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1],km.cluster_centers_[:, 2], c = 'blue' , label = 'centeroid')

fig.set_xlabel('Age of a customer')

fig.set_ylabel('Anual Income')

fig.set_zlabel('Spending Score')

fig.set_title('Clusters of Customers')

column2 = ['Age','Spending Score (1-100)']

data2 = mail_customers.loc[0:,column2]

silhouette_all2=[]

for k in range(2,11):

    kmeans_model = KMeans(n_clusters=k, random_state=1).fit(data)

    labels = kmeans_model.labels_

    a = metrics.silhouette_score(data2, labels, metric='euclidean')

    silhouette_all2.append(a)

    #print(a)

    print('This is the silhouette score when k equals',k,': ',a)
plt.figure(figsize=(10,6))

plt.plot(range(2,11), silhouette_all2, marker='o')

plt.xlabel('Number of clusters')

plt.ylabel('silhouette score')

plt.annotate('max score', xy=(2, 0.4692341232501655),arrowprops=dict(facecolor='black'))
km = KMeans(n_clusters = 2, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

y_means = km.fit_predict(data2)

plt.figure(figsize=(10,6))

sns.countplot(y_means)

plt.xlabel('clusters')

plt.ylabel('counts in each cluster')
x = data2.values

fig = plt.figure(figsize = (16,10))

plt.scatter(x[y_means == 0,0], x[y_means == 0,1], s = 50, c = 'green', marker = 'o')

plt.scatter(x[y_means == 1,0], x[y_means == 1,1], s = 50, c = 'yellow', marker = 'v')

plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 100,c = 'red' , label = 'centeroid')

plt.xlabel('Age of a customer')

plt.ylabel('Spending Score (1-100)')

plt.title('Clusters of Customers')
column3 = ['Annual Income (k$)','Spending Score (1-100)']

data3 = mail_customers.loc[0:,column3]

silhouette_all3=[]

for k in range(2,11):

    kmeans_model = KMeans(n_clusters=k, random_state=1).fit(data)

    labels = kmeans_model.labels_

    a = metrics.silhouette_score(data3, labels, metric='euclidean')

    silhouette_all3.append(a)

    #print(a)

    print('This is the silhouette score when k equals',k,': ',a)
plt.figure(figsize=(10,6))

plt.plot(range(2,11), silhouette_all3, marker='o')

plt.xlabel('Number of clusters')

plt.ylabel('silhouette score')

plt.annotate('max score', xy=(5, 0.5503719213912603),arrowprops=dict(facecolor='black'))
km = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

y_means = km.fit_predict(data3)

plt.figure(figsize=(10,6))

sns.countplot(y_means)

plt.xlabel('clusters')

plt.ylabel('counts in each cluster')
plt.style.use('ggplot')

x = data3.values

fig = plt.figure(figsize = (16,10))

plt.scatter(x[y_means == 0,0], x[y_means == 0,1], s = 50, c = 'green', marker = 'o')

plt.scatter(x[y_means == 1,0], x[y_means == 1,1], s = 50, c = 'yellow', marker = 'v')

plt.scatter(x[y_means == 2, 0], x[y_means == 2, 1],s = 50, c = 'cyan',  marker = 's')

plt.scatter(x[y_means == 3, 0], x[y_means == 3, 1],s = 50, c = 'magenta',  marker = 'p')

plt.scatter(x[y_means == 4, 0], x[y_means == 4, 1],s = 50, c = 'orange', marker = 'x')

plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 100,c = 'red' , label = 'centeroid')

plt.xlabel('Annual Income')

plt.ylabel('Spending Score (1-100)')

plt.title('Clusters of Customers')