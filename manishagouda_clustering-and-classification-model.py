import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df= pd.read_csv('../input/glass/glass.csv')

df.head()
df.shape
df.info()
df.describe()
df.isnull().sum()
from scipy.cluster.hierarchy import linkage,cophenet,dendrogram
### Checking the number of classes in the actual data

df['Type'].value_counts()
df1=df.drop('Type',axis=1)
df1.head()
#### Standardizing the data before clustering

from scipy.stats import zscore

df1=df1.apply(zscore)
from sklearn.cluster import KMeans
cluster_no = range(1,15)

wcss=[]

for no in cluster_no:

    km = KMeans(no,random_state=1)

    km.fit(df1)

    wcss.append(km.inertia_)

    
## Elbow curve to identify the appropriate no of clusters
plt.figure(figsize=(10,5))

plt.plot(cluster_no,wcss,marker='o')
km = KMeans(n_clusters=6,random_state=1)

km.fit(df1)
km.labels_
df1['Class']= km.labels_
df1.head()
df1['Class'].value_counts()
 # plot of the clusters using two features

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,8))



ax1 = plt.subplot(1,2,1)

plt.title('Original Classes')

sns.scatterplot(x='Ca', y='Fe', hue='Type', style='Type', data=df, ax=ax1)



ax2 = plt.subplot(1,2,2)

plt.title('Predicted Classes')

sns.scatterplot(x='Ca', y='Fe', hue='Class', style='Class', data=df1, ax=ax2)

plt.show()
df2=df.drop('Type',axis=1)

df2=df2.apply(zscore)
from scipy.spatial.distance import pdist
Z = linkage(df2, method='complete')

c, coph_dists = cophenet(Z , pdist(df2))

c
Z = linkage(df2, method='single')

c, coph_dists = cophenet(Z , pdist(df2))

c
Z = linkage(df2, method='ward')

c, coph_dists = cophenet(Z , pdist(df2))

c
Z = linkage(df2, method='average')

c, coph_dists = cophenet(Z , pdist(df2))

c
from scipy.cluster.hierarchy import linkage, dendrogram

plt.figure(figsize=[10,10])

merg = linkage(df2, method='average')

dendrogram(merg, leaf_rotation=90)

plt.title('Dendrogram')

plt.xlabel('Data Points')

plt.ylabel('Euclidean Distances')

plt.show()
from scipy.cluster.hierarchy import linkage, dendrogram

plt.figure(figsize=[10,10])

merg = linkage(df2, method='ward')

dendrogram(merg, leaf_rotation=90)

plt.title('Dendrogram')

plt.xlabel('Data Points')

plt.ylabel('Euclidean Distances')

plt.show()
from sklearn.cluster import AgglomerativeClustering

ac = AgglomerativeClustering(n_clusters=6, affinity ='euclidean',linkage='ward')

ac.fit(df2)
df2['label']=ac.labels_
df2.head()
df2['label'].value_counts()
 # plot of the clusters using two features

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,8))



ax1 = plt.subplot(1,2,1)

plt.title('Original Classes')

sns.scatterplot(x='Ca', y='Fe', hue='Type', style='Type', data=df, ax=ax1)



ax2 = plt.subplot(1,2,2)

plt.title('Predicted Classes')

sns.scatterplot(x='Ca', y='Fe', hue='label', style='label', data=df2, ax=ax2)

plt.show()
plt.title('Original Classes')

sns.scatterplot(x='Mg', y='Al', hue='Type', style='Type', data=df)

plt.show()

plt.title('K-Means Classes')

sns.scatterplot(x='Mg', y='Al', hue='Class', style='Class', data=df1)

plt.show()

plt.title('Hierarchical Classes')

sns.scatterplot(x='Mg', y='Al', hue='label', style='label', data=df2)

plt.show()
print('Original Data Classes:')

print(df.Type.value_counts())

print('-' * 30)

print('K-Means Predicted Data Classes:')

print(df1.Class.value_counts())

print('-' * 30)

print('Hierarchical Predicted Data Classes:')

print(df2.label.value_counts())
###### Calculating cohen_kappa_score to check the aggrement
#### For KMeans Label

from sklearn.metrics import cohen_kappa_score

cohen_kappa_score(df['Type'],df1['Class'] )
##### For Agglomerative Label

cohen_kappa_score(df['Type'],df2['label'])
df3=df1.drop('Class',1)

df3.head()
##### Finding Silhouette Score 
from sklearn.metrics import silhouette_score

silhouette_score ( df3 , df1['Class']  )
silhouette_score ( df3 , df2['label']  )
## Decision Tree



from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score,roc_auc_score

dt = DecisionTreeClassifier()

x= df1.drop('Class',axis=1)

y=df1['Class']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

dt.fit(x_train,y_train)

y_pred=dt.predict(x_test)

y_prob = dt.predict_proba(x_test)[:,1]

print('accuracy_score fot test:',accuracy_score(y_test,y_pred))

## KNN model



from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier()

kn.fit(x_train,y_train)

y_pred=kn.predict(x_test)

y_prob = kn.predict_proba(x_test)[:,1]

print('accuracy_score fot test:',accuracy_score(y_test,y_pred))

## Logistic regression



from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_train,y_train)

y_pred=lr.predict(x_test)

y_prob = lr.predict_proba(x_test)[:,1]

print('accuracy_score fot test:',accuracy_score(y_test,y_pred))

## SVC



from sklearn.svm import SVC

svc= SVC(probability=True)

svc.fit(x_train,y_train)

y_pred=svc.predict(x_test)

y_prob = svc.predict_proba(x_test)[:,1]

print('accuracy_score fot test:',accuracy_score(y_test,y_pred))

###### Using PC for model building 
from sklearn.decomposition import PCA
df.head()

df4=df.drop('Type',axis=1)
from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(df4)
# covariance matrix

cov_matrix = np.cov(X_std.T)

print('Covariance Matrix \n', cov_matrix)
eig_values,eig_vect = np.linalg.eig(cov_matrix)
print('Eigen Vectors \n', eig_vect)

print('\n Eigen Values \n', eig_values)
tot = sum(eig_values)

var_exp = [( i /tot ) * 100 for i in sorted(eig_values, reverse=True)]

cum_var = np.cumsum(var_exp)

print("Cumulative Variance Explained", cum_var)
pca1 = PCA(n_components=6).fit_transform(X_std)
## KNN model



from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier()

kn.fit(pca1,y)

y_pred=kn.predict(pca1)

print('accuracy_score fot test:',accuracy_score(y,y_pred))

# Logistic regression

lr = LogisticRegression()

lr.fit(pca1,y)

y_pred=lr.predict(pca1)

print('accuracy_score fot test:',accuracy_score(y,y_pred))
## Decision Tree

dt =DecisionTreeClassifier()

dt.fit(pca1,y)

y_pred=dt.predict(pca1)

print('accuracy_score fot test:',accuracy_score(y,y_pred))
## SVM

svc= SVC(probability=True)

svc.fit(pca1,y)

y_pred=svc.predict(pca1)

print('accuracy_score fot test:',accuracy_score(y,y_pred))



df2.head()
x= df2.drop('label',axis=1)

y=df2['label']
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

## KNN model



from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier()

kn.fit(x_train,y_train)

y_pred=kn.predict(x_test)

y_prob = kn.predict_proba(x_test)[:,1]

print('accuracy_score fot test:',accuracy_score(y_test,y_pred))

## Logistic model



lr = LogisticRegression()

lr.fit(x_train,y_train)

y_pred=lr.predict(x_test)

y_prob = lr.predict_proba(x_test)[:,1]

print('accuracy_score fot test:',accuracy_score(y_test,y_pred))

## Decision tree



dt =DecisionTreeClassifier()

dt.fit(x_train,y_train)

y_pred=dt.predict(x_test)

y_prob = dt.predict_proba(x_test)[:,1]

print('accuracy_score fot test:',accuracy_score(y_test,y_pred))

## SVM



svc= SVC(probability=True)

svc.fit(x_train,y_train)

y_pred=svc.predict(x_test)

print('accuracy_score fot test:',accuracy_score(y_test,y_pred))

### with pca 
sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)
x_train
pca2 = PCA()

x_train_2 = pca2.fit_transform(x_train)

x_test_2 = pca2.transform(x_test)
## Decision tree

dt=DecisionTreeClassifier()

dt.fit(x_train_2,y_train)

y_pred = dt.predict(x_test_2)

print("Accuracy Score:",accuracy_score(y_test, y_pred) )
## SVM



svc=SVC()

svc.fit(x_train_2,y_train)

y_pred = svc.predict(x_test_2)

print("Accuracy Score:",accuracy_score(y_test, y_pred) )
## logistic

lr=LogisticRegression()

lr.fit(x_train_2,y_train)

y_pred = lr.predict(x_test_2)

print("Accuracy Score:",accuracy_score(y_test, y_pred) )
## KNN

kn=KNeighborsClassifier()

kn.fit(x_train_2,y_train)

y_pred = kn.predict(x_test_2)

print("Accuracy Score:",accuracy_score(y_test, y_pred) )
import pandas as pd

glass = pd.read_csv("../input/glass/glass.csv")