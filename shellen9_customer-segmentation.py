import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns

from sklearn.cluster import KMeans

customers = pd.read_csv(r'../input/Mall_Customers.csv')
customers.head(5)
customers.info()
customers.isnull().sum()
customers[['Age','Annual Income (k$)','Spending Score (1-100)']].describe()
col = ['Age','Annual Income (k$)','Spending Score (1-100)']

n = 0

plt.figure(1,figsize = (15,6))

for x in col:

    n += 1

    plt.subplot(1,3,n)

    plt.subplots_adjust(hspace = 0.5,wspace = 0.5)

    sns.distplot(customers[x],bins = 20)

    plt.title('Displot of {}'.format(x))

plt.show()
customers.Gender.value_counts().plot(kind = 'bar')
plt.figure(1,figsize = (15,7))

n = 0

for y in col:

    n += 1

    plt.subplot(1,3,n)

    plt.subplots_adjust(wspace = 0.5,hspace = 0.5)

    sns.violinplot(x = 'Gender',y = y ,data = customers,palette='vlag')

    sns.swarmplot(x = 'Gender',y = y ,data = customers)

    plt.xlabel('Gender')

    plt.ylabel(y)

plt.show()

    

        
plt.figure(1,figsize=(15,6))

n = 0

for x in col:

    for y in col:

        n += 1

        plt.subplot(3,3,n)

        plt.subplots_adjust(wspace = 0.5,hspace = 0.5)

        sns.regplot(x = x,y = y,data = customers)

plt.show()
plt.figure(1,figsize=(15,6))

n = 0

for x in col:

    for y in col:

        n += 1

        plt.subplot(3,3,n)

        plt.subplots_adjust(wspace = 0.5,hspace = 0.5)

        sns.scatterplot(x = x ,y = y , hue = 'Gender',data = customers)

plt.show()
X1 = customers[['Age','Spending Score (1-100)']]

inertia1 = []

for n in range(1,11):

    algorithm1 = KMeans(n_clusters=n)

    algorithm1.fit(X1)

    inertia1.append(algorithm1.inertia_)

plt.figure(1,figsize = (15,6))

plt.plot(range(1,11),inertia1,'o')

plt.plot(range(1,11),inertia1,'-',alpha = 0.5)

plt.xlabel('Number of Cluster')

plt.ylabel('inertia')

plt.show()
algorithm1 = KMeans(n_clusters=4)

algorithm = algorithm1.fit_predict(X1)

plt.scatter(X1['Age'],X1['Spending Score (1-100)'],c = algorithm)
X2 = customers[['Age','Annual Income (k$)']]

inertia2 = []

for n in range(1,11):

    algorithm2 = KMeans(n_clusters=n)

    algorithm2.fit(X2)

    inertia2.append(algorithm2.inertia_)

plt.figure(1,figsize = (15,5))

plt.plot(range(1,11),inertia2,'o')

plt.plot(range(1,11),inertia2,'-',alpha = 0.5)

plt.xlabel('Number of Cluster')

plt.ylabel('inertia')

plt.show()
algorithm2 = KMeans(n_clusters=5)

algorithm = algorithm2.fit_predict(X2)

plt.scatter(X2['Age'],X2['Annual Income (k$)'],c = algorithm)

plt.show()
X3 = customers[['Annual Income (k$)','Spending Score (1-100)']]

inertia3 = []

for n in range(1,11):

    algorithm3 = KMeans(n_clusters=n)

    algorithm3.fit(X3)

    inertia3.append(algorithm3.inertia_)

plt.figure(1,figsize = (15,5))

plt.plot(range(1,11),inertia3,'o')

plt.plot(range(1,11),inertia3,'-',alpha = 0.5)

plt.xlabel('Numbers of Cluster')

plt.ylabel('inertia')

plt.show()
algorithm3 = KMeans(n_clusters=5)

algorithm = algorithm3.fit_predict(X3)

plt.scatter(X3['Annual Income (k$)'],X3['Spending Score (1-100)'],c = algorithm)

plt.xlabel('Annual Income')

plt.ylabel('Spending Score')

plt.show()
customers['Classify'] = algorithm3.predict(X3)
sns.scatterplot(x = 'Annual Income (k$)',y = 'Spending Score (1-100)',hue = 'Classify',data = customers)
sns.boxplot(x = 'Classify',y = 'Age',data = customers)
customers.Classify.value_counts()