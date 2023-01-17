#Import all the required libraries



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from sklearn.preprocessing import LabelEncoder

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv', index_col=0)

df.head()
df.shape    # to see number of rows and features
df.info()   #to get information about each of the feature
df.dtypes
df['Gender']=df['Gender'].replace("Female",1)

df['Gender']=df['Gender'].replace("Male",0)

df.head()
# checking if there is any NULL data



df.isnull().sum()
sns.pairplot(df.iloc[:,0:4])
plt.figure(1 , figsize = (15 , 6))

n = 0 

for x in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:

    n += 1

    plt.subplot(1 , 3 , n)

    plt.subplots_adjust(hspace =0.5 , wspace = 0.5)

    sns.distplot(df[x] , bins = 20)

    plt.title('Distplot of {}'.format(x))

plt.show()
labels = ['Female', 'Male']

size = df['Gender'].value_counts()

colors = ['lightgreen', 'orange']

explode = [0, 0.1]



plt.rcParams['figure.figsize'] = (9, 9)

plt.pie(size, colors = colors, explode = explode, labels = labels, shadow = True, autopct = '%.2f%%')

plt.title('Gender', fontsize = 20)

plt.axis('off')

plt.legend()

plt.show()
plt.figure(1 , figsize = (15 , 7))

n = 0 

for x in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:

    for y in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:

        n += 1

        plt.subplot(3 , 3 , n)

        plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)

        sns.regplot(x = x , y = y , data = df)

        plt.ylabel(y.split()[0]+' '+y.split()[1] if len(y.split()) > 1 else y )

plt.show()
hm=sns.heatmap(df.iloc[:,1:5].corr(), annot = True, linewidths=.5, cmap='Blues')

hm.set_title(label='Heatmap of dataset', fontsize=20)

hm
plt.rcParams['figure.figsize'] = (18, 7)

sns.violinplot(df['Gender'], df['Annual Income (k$)'], palette = 'rainbow')

plt.title('Gender vs Spending Score', fontsize = 20)

plt.show()
plt.rcParams['figure.figsize'] = (18, 7)

sns.stripplot(df['Gender'], df['Age'], palette = 'Purples', size = 10)

plt.title('Gender vs Spending Score', fontsize = 20)

plt.show()
x = df['Annual Income (k$)']

y = df['Age']

z = df['Spending Score (1-100)']



sns.lineplot(x, y, color = 'blue')

sns.lineplot(x, z, color = 'pink')

plt.title('Annual Income vs Age and Spending Score', fontsize = 20)

plt.show()
x = df.iloc[:, [2, 3]].values



# let's check the shape of x

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
km = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

y_means = km.fit_predict(x)



plt.scatter(x[y_means == 0, 0], x[y_means == 0, 1], s = 100, c = 'pink', label = 'miser')

plt.scatter(x[y_means == 1, 0], x[y_means == 1, 1], s = 100, c = 'yellow', label = 'general')

plt.scatter(x[y_means == 2, 0], x[y_means == 2, 1], s = 100, c = 'cyan', label = 'target')

plt.scatter(x[y_means == 3, 0], x[y_means == 3, 1], s = 100, c = 'magenta', label = 'spendthrift')

plt.scatter(x[y_means == 4, 0], x[y_means == 4, 1], s = 100, c = 'orange', label = 'careful')

plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 250, c = 'blue' , label = 'centeroid')



plt.style.use('fivethirtyeight')

plt.title('K Means Clustering', fontsize = 20)

plt.xlabel('Annual Income')

plt.ylabel('Spending Score')

plt.legend()

plt.grid()

plt.show()
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

plt.grid()

plt.show()
kmeans = KMeans(n_clusters = 4, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

ymeans = kmeans.fit_predict(x)



plt.rcParams['figure.figsize'] = (10, 10)

plt.title('Cluster of Ages', fontsize = 30)



plt.scatter(x[ymeans == 0, 0], x[ymeans == 0, 1], s = 100, c = 'pink', label = 'Usual Customers' )

plt.scatter(x[ymeans == 1, 0], x[ymeans == 1, 1], s = 100, c = 'orange', label = 'Priority Customers')

plt.scatter(x[ymeans == 2, 0], x[ymeans == 2, 1], s = 100, c = 'lightgreen', label = 'Target Customers(Young)')

plt.scatter(x[ymeans == 3, 0], x[ymeans == 3, 1], s = 100, c = 'red', label = 'Target Customers(Old)')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 200, c = 'black' , label = 'centeroid')



plt.style.use('fivethirtyeight')

plt.xlabel('Age')

plt.ylabel('Spending Score (1-100)')

plt.legend()

plt.grid()

plt.show()