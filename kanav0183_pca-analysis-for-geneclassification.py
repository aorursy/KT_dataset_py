# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline
df= pd.read_csv('../input/data_set_ALL_AML_train.csv')

df.head()
df1 = [col for col in df.columns if "call" not in col]

df = df[df1]

df.head()
df.T.head()
df = df.T

df2 = df.drop(['Gene Description','Gene Accession Number'],axis=0)

df2.index = pd.to_numeric(df2.index)

df2.sort_index(inplace=True)

df2.head()
df2['cat'] = list(pd.read_csv('../input/actual.csv')[:38]['cancer'])

dic = {'ALL':0,'AML':1}

df2.replace(dic,inplace=True)

df2.head(3)
from sklearn.preprocessing import StandardScaler

X_std = StandardScaler().fit_transform(df2.drop('cat',axis=1))



from sklearn.decomposition import PCA as sklearnPCA

sklearn_pca = sklearnPCA(n_components=30)

Y_sklearn = sklearn_pca.fit_transform(X_std)
cum_sum = sklearn_pca.explained_variance_ratio_.cumsum()



sklearn_pca.explained_variance_ratio_[:10].sum()



cum_sum = cum_sum*100



fig, ax = plt.subplots(figsize=(8,8))

plt.bar(range(30), cum_sum, label='Cumulative _Sum_of_Explained _Varaince', color = 'b',alpha=0.5)

plt.title("Around 95% of variance is explained by the Fisrt 30 colmns ");
X_reduced2 = Y_sklearn
df2.cat.values
train = pd.DataFrame(X_reduced2)

train['cat'] =  df2['cat'].reset_index().cat

train.head(3)
from sklearn.decomposition import PCA as sklearnPCA

sklearn_pca = sklearnPCA(n_components=3)

X_reduced  = sklearn_pca.fit_transform(X_std)

Y=train['cat']

from mpl_toolkits.mplot3d import Axes3D

plt.clf()

fig = plt.figure(1, figsize=(10,6 ))

ax = Axes3D(fig, elev=-150, azim=110,)

ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=Y,cmap=plt.cm.Paired,linewidths=10)

ax.set_title("First three PCA directions")

ax.set_xlabel("1st eigenvector")

ax.w_xaxis.set_ticklabels([])

ax.set_ylabel("2nd eigenvector")

ax.w_yaxis.set_ticklabels([])

ax.set_zlabel("3rd eigenvector")

ax.w_zaxis.set_ticklabels([])

import matplotlib.pyplot as plt

%matplotlib inline

fig = plt.figure(1, figsize=(10,6))

plt.scatter(X_reduced[:, 0],  X_reduced[:, 1], c=df2['cat'],cmap=plt.cm.Paired,linewidths=10)

plt.annotate('See The Brown Cluster',xy=(20,-20),xytext=(9,8),arrowprops=dict(facecolor='black', shrink=0.05))

#plt.scatter(test_reduced[:, 0],  test_reduced[:, 1],c='r')

plt.title("This The 2D Transformation of above graph ")
test = pd.read_csv('../input/data_set_ALL_AML_independent.csv')



test.head(3)
test1 = [col for col in test.columns if "call" not in col]

test = test[test1]

test = test.T

test2 = test.drop(['Gene Description','Gene Accession Number'],axis=0)

test2.index = pd.to_numeric(test2.index)

test2.sort_index(inplace=True)

#test2['cat'] = list(pd.read_csv('actual.csv')[39:63]['cancer'])

#dic = {'ALL':0,'AML':1}

#test2.replace(dic,inplace=True)

#test2
from sklearn.preprocessing import StandardScaler

Y_std = StandardScaler().fit_transform(test2)



from sklearn.decomposition import PCA as sklearnPCA

sklearn_pca = sklearnPCA(n_components=30)

test_reduced = sklearn_pca.fit_transform(Y_std)
test_set = pd.DataFrame(test_reduced)



test_set.head(3)
train.drop('cat',axis=1).plot(kind='hist',figsize=(8,10))
from sklearn.neighbors import KNeighborsClassifier

clf= KNeighborsClassifier(n_neighbors=10,)

clf.fit(train.drop('cat',axis=1),train['cat'])
pred = clf.predict(test_set)



pateint = pd.read_csv('../input/actual.csv')['cancer'][38:]



true = pateint.replace(dic)



import sklearn

sklearn.metrics.confusion_matrix(true, pred)
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(min_samples_split=2)

clf.fit(train.drop('cat',axis=1),train['cat'])

pred = clf.predict(test_set)

true = pateint.replace(dic)

print(sklearn.metrics.confusion_matrix(true, pred))

print()
from sklearn import svm



clf=svm.SVC(kernel='linear')

clf.fit(train.drop('cat',axis=1),train['cat'])

pred = clf.predict(test_set)



pateint = pd.read_csv('../input/actual.csv')['cancer'][38:]



true = pateint.replace(dic)



print(sklearn.metrics.confusion_matrix(true, pred))

print()
fig = plt.figure(1, figsize=(14,6))

plt.scatter(X_reduced[:, 0],  X_reduced[:, 1], c=df2['cat'],cmap=plt.cm.Paired,alpha=0.7,linewidths=7)

plt.scatter(test_reduced[:, 0],  test_reduced[:, 1],c='r',linewidths=10)