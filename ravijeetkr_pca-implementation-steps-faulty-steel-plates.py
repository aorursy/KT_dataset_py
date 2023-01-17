# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()

sns.set_style('darkgrid')

import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/faulty-steel-plates/faults.csv')

train.head()
train.shape
train.info()
train['Other_Faults'].value_counts()
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

df_scaled = sc.fit_transform(train.iloc[:,:-1])

df_scaled = pd.DataFrame(df_scaled,columns = train.columns[:-1])

df_scaled.head()
from sklearn.cluster import KMeans

cluster_range = range(1,15)

cluster_error = []

for i in cluster_range:

    model = KMeans(n_clusters=i)

    model.fit(df_scaled)

    cluster_error.append(model.inertia_)

cluster_df = pd.DataFrame({'cluster_range':cluster_range,'cluster_error':cluster_error})

cluster_df.head(9)

plt.figure(figsize=(10,6))

plt.plot(cluster_range,cluster_error,marker = 'o')

plt.xlabel('range')

plt.ylabel('error')
kmeans = KMeans(n_clusters=2,n_init=15,random_state=0)

kmeans.fit(df_scaled)

df_k = df_scaled.copy(deep = True)

df_k['labels'] = kmeans.labels_
centroids = kmeans.cluster_centers_

centroids = pd.DataFrame(centroids, columns = df_scaled.columns)

centroids
print('Before Clustering:',train['Other_Faults'].value_counts(),sep='\n')

print('*'*30)

print('After Clustering:',df_k['labels'].value_counts(),sep = '\n')
fault_clusters = df_k.groupby(by = 'labels')

df0 = fault_clusters.get_group(0)

df1 = fault_clusters.get_group(1)

c0 = centroids.iloc[0,:]

c1 = centroids.iloc[1,:]
Inertia_group_0 = 0

for i in range(df0.shape[0]):

    Inertia_group_0 = Inertia_group_0 + np.sum((df0.iloc[i,:-1]-c0)**2)

print(Inertia_group_0)



Inertia_group_1 = 0

for i in range(df1.shape[0]):

    Inertia_group_1 = Inertia_group_1 + np.sum((df1.iloc[i,:-1]-c1)**2)

print(Inertia_group_1)

train.corr()
df_scaled.head()
cov_matrix = np.cov(df_scaled.T)

cov_matrix
eigenvalues,eigenvectors = np.linalg.eig(cov_matrix)

print('eigenvalues:',eigenvalues,sep='\n')

print('eigenectors:',eigenvectors,sep = '\n')
total = np.sum(eigenvalues)

var_exp = [(i/total)*100 for i in sorted(eigenvalues,reverse = True)]

cum_var_exp = np.cumsum(var_exp)

cum_var_exp
plt.figure(figsize=(12 , 8))

plt.bar(range(33), var_exp, alpha = 0.5, align = 'center', label = 'Individual explained variance')

plt.step(range(33), cum_var_exp, where='mid', label = 'Cumulative explained variance')

plt.ylabel('Explained Variance Ratio')

plt.xlabel('Principal Components')

plt.legend(loc = 'best')

plt.tight_layout()

plt.show()
eig_pairs = [(eigenvalues[index],eigenvectors[:,index]) for index in range(len(eigenvalues))]

eig_pairs
eig_pairs.sort()

eig_pairs.reverse()
eig_value_sort = [eig_pairs[index][0] for index in range(len(eigenvalues))]

eig_vector_sort = [eig_pairs[index][1] for index in range(len(eigenvalues))]
p_reduce = np.array(eig_vector_sort[0:17]).transpose()

p_reduce.shape
proj_data_17d = np.dot(df_scaled,p_reduce)

proj_data_17d.shape
kmeans_pca = KMeans(n_clusters=2,n_init=15,random_state = 0)

kmeans_pca.fit(proj_data_17d)

df_k_pca = df_scaled.copy()

df_k_pca['label'] = kmeans.labels_
kmeans_pca.inertia_
print('Inertia using KMeans clustering Before PCA:',kmeans.inertia_)

print('Inertia using KMeans clustering After PCA:',kmeans_pca.inertia_)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score,KFold
y_train = train['Other_Faults']

X_train = train.drop('Other_Faults',axis = 1)

X_train.shape,y_train.shape
model = LogisticRegression()

kfold = KFold(n_splits=10,shuffle=True,random_state=42).get_n_splits(train.values)

cv_results = cross_val_score(model,X_train,y_train,cv = kfold,scoring='roc_auc')

print('accuracy_mean:',np.mean(cv_results),'acuracy_std:',np.std(cv_results,ddof =1),sep='\n')
model = LogisticRegression()

kfold = KFold(n_splits=10,shuffle=True,random_state=42).get_n_splits(train.values)

cv_results = cross_val_score(model,proj_data_17d,y_train,cv = kfold,scoring='roc_auc')

print('accuracy_mean:',np.mean(cv_results),'acuracy_std:',np.std(cv_results,ddof =1),sep='\n')