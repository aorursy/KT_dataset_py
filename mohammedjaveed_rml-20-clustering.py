



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# importing the dataset

data = pd.read_csv('/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')

data.head(10)
data.isnull().any().any()
data.info()
print(data['gender'].value_counts())

print("--------------------------------------------")

print(data['SeniorCitizen'].value_counts())

print("--------------------------------------------")

print(data['tenure'].value_counts())

print("--------------------------------------------")

print(data['MonthlyCharges'].value_counts())

print("--------------------------------------------")

print(data['Churn'].value_counts())

print("--------------------------------------------")

print(data['TotalCharges'].value_counts())

print("--------------------------------------------")

print(data['Contract'].value_counts())
data['gender'] = data['gender'].apply({'Male':0, 'Female':1}.get)

data['Churn'] = data['Churn'].apply({'No':0, 'Yes':1}.get)

data['Contract'] = data['Contract'].apply({'Month-to-month':0,'One year':1, 'Two year':2}.get)

data['PhoneService'] = data['PhoneService'].apply({'No':0, 'Yes':1}.get)

data = data.replace('No',0)

data = data.replace('Yes',1)

data['TotalCharges'] = data.TotalCharges.replace(" ",0,regex = True)

data['TotalCharges'] = pd.to_numeric(data['TotalCharges'])

sns.pairplot(data)
labels = ['Female', 'Male']

size = data['gender'].value_counts()

colors = ['lightgray', 'indigo']

explode = [0, 0.1]



plt.rcParams['figure.figsize'] = (5, 5)

plt.pie(size, colors = colors, explode = explode, labels = labels, shadow = True, autopct = '%.2f%%')

plt.title('Gender', fontsize = 20)

plt.axis('off')

plt.legend()

plt.show()
plt.rcParams['figure.figsize'] = (15, 8)

sns.countplot(data['Contract'])

plt.title('Distribution of Contract', fontsize = 20)

plt.show()
plt.rcParams['figure.figsize'] = (15, 8)

sns.heatmap(data.corr(),  annot = True)

plt.title('Heatmap for the Data', fontsize = 20)

plt.show()
#  Gender vs Spendscore



plt.rcParams['figure.figsize'] = (18, 7)

sns.boxenplot(data['gender'], data['MonthlyCharges'], palette = 'Blues')

plt.title('Gender vs MonthlyCharges', fontsize = 20)

plt.show()
plt.rcParams['figure.figsize'] = (18, 7)

sns.stripplot(data['gender'], data['MonthlyCharges'], palette = 'Purples', size = 10)

plt.title('Gender vs MonthlyCharges', fontsize = 20)

plt.show()
plt.rcParams['figure.figsize'] = (18, 7)

sns.stripplot(data['gender'], data['TotalCharges'], palette = 'Purples', size = 10)

plt.title('Gender vs TotalCharges', fontsize = 20)

plt.show()


x = data.iloc[:, [5, 19]].values

print(x.shape)
from sklearn.cluster import KMeans



wcss = []

for i in range(1, 11):

    km = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

    km.fit(x)

    wcss.append(km.inertia_)

    

plt.plot(range(1, 11), wcss)

plt.title('The Elbow Method', fontsize = 20)

plt.xlabel('No. of Clusters')

plt.ylabel('wcss')

plt.show()

km = KMeans(n_clusters = 4, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

y_means = km.fit_predict(x)



plt.scatter(x[y_means == 0, 0], x[y_means == 0, 1], s = 100, c = 'blue', label = 'low/new users')

plt.scatter(x[y_means == 1, 0], x[y_means == 1, 1], s = 100, c = 'yellow', label = 'High-attracted users')

plt.scatter(x[y_means == 2, 0], x[y_means == 2, 1], s = 100, c = 'cyan', label = 'low-medium users')

plt.scatter(x[y_means == 3, 0], x[y_means == 3, 1], s = 100, c = 'red', label = 'Medium-high users')



plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 200 , c =  'grey' , label = 'centeroid')



plt.style.use('fivethirtyeight')

plt.title('K Means Clustering', fontsize = 20)

plt.xlabel('Tenure')

plt.ylabel('TotalCharges')

plt.legend()

plt.grid()

plt.show()
x = data.iloc[:, [5, 18]].values

print(x)
from sklearn.cluster import KMeans



wcss = []

for i in range(1, 11):

    km = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

    km.fit(x)

    wcss.append(km.inertia_)

    

plt.plot(range(1, 11), wcss)

plt.title('The Elbow Method', fontsize = 20)

plt.xlabel('No. of Clusters')

plt.ylabel('wcss')

plt.show()

km = KMeans(n_clusters = 4, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

y_means = km.fit_predict(x)



plt.scatter(x[y_means == 0, 0], x[y_means == 0, 1], s = 100, c = 'blue', label = 'new high usage users')

plt.scatter(x[y_means == 1, 0], x[y_means == 1, 1], s = 100, c = 'yellow', label = 'new low usage users')

plt.scatter(x[y_means == 2, 0], x[y_means == 2, 1], s = 100, c = 'cyan', label = 'old high usage users')

plt.scatter(x[y_means == 3, 0], x[y_means == 3, 1], s = 100, c = 'red', label = 'old low usage users')



plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 200 , c =  'grey' , label = 'centeroid')



plt.style.use('fivethirtyeight')

plt.title('K Means Clustering', fontsize = 20)

plt.xlabel('Tenure')

plt.ylabel('MonthlyCharges')

plt.legend()

plt.grid()

plt.show()
import scipy.cluster.hierarchy as sch



dendrogram = sch.dendrogram(sch.linkage(x, method = 'ward'))

plt.title('Dendrogam', fontsize = 20)

plt.xlabel('Customers')

plt.ylabel('Ecuclidean Distance')

plt.show()
from sklearn.cluster import AgglomerativeClustering



hc = AgglomerativeClustering(n_clusters = 4, affinity = 'euclidean', linkage = 'ward')

y_hc = hc.fit_predict(x)



plt.scatter(x[y_means == 0, 0], x[y_means == 0, 1], s = 100, c = 'blue', label = 'new high usage users')

plt.scatter(x[y_means == 1, 0], x[y_means == 1, 1], s = 100, c = 'yellow', label = 'new low usage users')

plt.scatter(x[y_means == 2, 0], x[y_means == 2, 1], s = 100, c = 'cyan', label = 'old high usage users')

plt.scatter(x[y_means == 3, 0], x[y_means == 3, 1], s = 100, c = 'red', label = 'old low usage users')



plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 50, c = 'Green' , label = 'centeroid')



plt.style.use('fivethirtyeight')

plt.title('Hierarchial Clustering', fontsize = 20)

plt.xlabel('Tenure')

plt.ylabel('MonthlyCharges')

plt.legend()

plt.grid()

plt.show()