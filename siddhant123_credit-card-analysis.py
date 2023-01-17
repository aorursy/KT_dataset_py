# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/ccdata/CC GENERAL.csv')

df.head()
df.info()
df.describe()
df.isnull().any()
df.isnull().sum()
df.size
df.dropna(inplace = True)
df.columns
df.corr()
import seaborn as sns

import scipy 

import matplotlib.pyplot as plt

%matplotlib inline
plt.figure(figsize = (15,12))

sns.heatmap(df.corr(),cmap = 'inferno',annot = True)

plt.show()
slope,intercept,r_value,p_value,std_err = scipy.stats.linregress(x = df['PURCHASES'],y = df['ONEOFF_PURCHASES'])

reg1 = sns.regplot(x = 'PURCHASES', y = 'ONEOFF_PURCHASES', data = df, line_kws = 

                   {'label':"y = {0:f}x + {1:f}".format(slope,intercept)})

reg1.legend()

print("The correlation coefficient is " + str(r_value))
slope,intercept,r_value,p_value,std_err = scipy.stats.linregress(x = df['PURCHASES'],y = df['PURCHASES_TRX'])

reg1 = sns.regplot(x = 'PURCHASES', y = 'PURCHASES_TRX', data = df, line_kws = 

                   {'label':"y = {0:f}x + {1:f}".format(slope,intercept)})

reg1.legend()

print("The correlation coefficient is " + str(r_value))
slope,intercept,r_value,p_value,std_err = scipy.stats.linregress(x = df['CASH_ADVANCE'],y = df['BALANCE'])

reg1 = sns.regplot(x = 'CASH_ADVANCE', y = 'BALANCE', data = df, line_kws = 

                   {'label':"y = {0:f}x + {1:f}".format(slope,intercept)})

reg1.legend()

print("The correlation coefficient is " + str(r_value))
slope,intercept,r_value,p_value,std_err = scipy.stats.linregress(x = df['CASH_ADVANCE_FREQUENCY'],y = df['PURCHASES_FREQUENCY'])

reg1 = sns.regplot(x = 'CASH_ADVANCE_FREQUENCY', y = 'PURCHASES_FREQUENCY', data = df, line_kws = 

                   {'label':"y = {0:f}x + {1:f}".format(slope,intercept)})

reg1.legend()

print("The correlation coefficient is " + str(r_value))
sns.distplot(df['BALANCE'],bins = 20)
sns.distplot(df['BALANCE_FREQUENCY'],bins = 20)
sns.distplot(df['PURCHASES'],bins = 20)
sns.distplot(df['ONEOFF_PURCHASES'],bins = 20)
sns.distplot(df['INSTALLMENTS_PURCHASES'],bins = 20)
sns.distplot(df['CASH_ADVANCE'],bins = 20)
X = df.iloc[:,[2,3]].values

from sklearn.cluster import KMeans

inertia = []

for i in range(1,5):

    kmeans = KMeans(n_clusters=i,init = 'k-means++',max_iter = 300,n_init = 10,random_state = 0)

    kmeans.fit(X)

    inertia.append(kmeans.inertia_)

plt.plot(range(1,5),inertia)

plt.title('The Elbow Method')

plt.xlabel('Number of Clusters')

plt.ylabel('Inertia')

plt.show()
#Applying KMeans to the Dataset



kmeans = KMeans(n_clusters=3,init = 'k-means++',max_iter = 300,n_init = 10,random_state = 0)

y_kmeans = kmeans.fit_predict(X)
plt.figure(figsize = (12,8))

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'cyan', label = 'Low balance frequency and low purchases')

plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'yellow', label = 'Medium balance frequency and medium purchases')

plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'blue', label = 'High balance frequency and high purchases')

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:, 1], s = 50, c = 'red' , label = 'centroids')

plt.title('BALANCE_FREQUENCY vs PURCHASES')

plt.xlabel('BALANCE_FREQUENCY')

plt.ylabel('PURCHASES')

plt.legend()

plt.show()
X = df.iloc[:,[1,2]].values

from sklearn.cluster import KMeans

inertia = []

for i in range(1,5):

    kmeans = KMeans(n_clusters=i,init = 'k-means++',max_iter = 300,n_init = 10,random_state = 0)

    kmeans.fit(X)

    inertia.append(kmeans.inertia_)

plt.plot(range(1,5),inertia)

plt.title('The Elbow Method')

plt.xlabel('Number of Clusters')

plt.ylabel('Inertia')

plt.show()
#Applying KMeans to the Dataset



kmeans = KMeans(n_clusters=3,init = 'k-means++',max_iter = 300,n_init = 10,random_state = 0)

y_kmeans = kmeans.fit_predict(X)
plt.figure(figsize = (12,8))

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'purple', label = 'Low balance and balance frequency')

plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Medium balance and balance frequency')

plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'High balance and balance frequency')

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:, 1], s = 50, c = 'red' , label = 'centroids')

plt.title('Balance vs Balance Frequency')

plt.xlabel('Balance')

plt.ylabel('Balance Frequency')

plt.legend()

plt.show()
#'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE','PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY',

#'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY','CASH_ADVANCE_TRX', 'PURCHASES_TRX',

#'CREDIT_LIMIT', 'PAYMENTS','MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT', 'TENURE'
X = df.iloc[:,[5,6]].values

from sklearn.cluster import KMeans

inertia = []

for i in range(1,5):

    kmeans = KMeans(n_clusters=i,init = 'k-means++',max_iter = 300,n_init = 10,random_state = 0)

    kmeans.fit(X)

    inertia.append(kmeans.inertia_)

plt.plot(range(1,5),inertia)

plt.title('The Elbow Method')

plt.xlabel('Number of Clusters')

plt.ylabel('Inertia')

plt.show()
#Applying KMeans to the Dataset



kmeans = KMeans(n_clusters=3,init = 'k-means++',max_iter = 300,n_init = 10,random_state = 0)

y_kmeans = kmeans.fit_predict(X)
plt.figure(figsize = (12,8))

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'magenta', label = 'Low CASH_ADVANCE and INSTALLMENTS_PURCHASES')

plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'pink', label = 'Medium CASH_ADVANCE and INSTALLMENTS_PURCHASES')

plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'High CASH_ADVANCE and INSTALLMENTS_PURCHASES')

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:, 1], s = 50, c = 'red' , label = 'centroids')

plt.title('INSTALLMENTS_PURCHASES vs CASH_ADVANCE')

plt.xlabel('INSTALLMENTS_PURCHASES')

plt.ylabel('CASH_ADVANCE')

plt.legend()

plt.show()
X = df.iloc[:,[12,13]].values

from sklearn.cluster import KMeans

inertia = []

for i in range(1,5):

    kmeans = KMeans(n_clusters=i,init = 'k-means++',max_iter = 300,n_init = 10,random_state = 0)

    kmeans.fit(X)

    inertia.append(kmeans.inertia_)

plt.plot(range(1,5),inertia)

plt.title('The Elbow Method')

plt.xlabel('Number of Clusters')

plt.ylabel('Inertia')

plt.show()
#Applying KMeans to the Dataset



kmeans = KMeans(n_clusters=3,init = 'k-means++',max_iter = 300,n_init = 10,random_state = 0)

y_kmeans = kmeans.fit_predict(X)
plt.figure(figsize = (12,8))

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'magenta', label = 'Low CREDIT_LIMIT and PAYMENTS')

plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'purple', label = 'Medium CREDIT_LIMIT and PAYMENTS')

plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'yellow', label = 'High CREDIT_LIMIT and PAYMENTS')

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:, 1], s = 50, c = 'red' , label = 'centroids')

plt.title('CREDIT_LIMIT vs PAYMENTS')

plt.xlabel('CREDIT_LIMIT')

plt.ylabel('PAYMENTS')

plt.legend()

plt.show()