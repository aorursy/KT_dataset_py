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

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

import collections

from collections import Counter as count
data= pd.read_csv("../input/uci-turkiye-student-evaluation-data-set/turkiye-student-evaluation_generic.csv")

data.head()
data.columns
data.info()
data.shape
data.isnull().sum()
data.describe()
data.shape
sns.countplot(x='class',data=data)

plt.show()
plt.figure(figsize=(15,6))

sns.boxplot(data=data.loc[:,'Q1':'Q25'])

plt.show()
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

data = pd.DataFrame(sc.fit_transform(data),columns=data.columns)
from sklearn.cluster import KMeans
cluster_range = range(1,10)

cluster_errors = []



for num_clusters in cluster_range:

    clusters = KMeans(num_clusters, n_init=10, max_iter=100)

    clusters.fit(data)

    

    cluster_errors.append(clusters.inertia_)

    

pd.DataFrame({'num_clusters':cluster_range, 'Error': cluster_errors})
plt.figure(figsize=(10,5))

plt.plot(cluster_range, cluster_errors, marker = "o" )

plt.title('Elbow Plot')

plt.xlabel('Number of Clusters')

plt.ylabel('Error')

plt.xticks(cluster_range)

plt.show()
kmeans = KMeans(n_clusters = 3, init = 'k-means++')

y_kmeans = kmeans.fit_predict(data)
y_kmeans
count(y_kmeans)
from scipy.cluster.hierarchy import dendrogram, linkage
plt.figure(figsize=(20,10))

Z = linkage(data, method='ward')

dendrogram(Z, leaf_rotation=90, p=10, truncate_mode='level', leaf_font_size=6, color_threshold=8)

plt.title('Dendogram')

plt.show()
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=3, affinity='euclidean',  linkage='ward')

ac.fit(data)
ac.labels_
y_ac=ac.fit_predict(data)
count(y_ac)
first0=[2225,1554]

second1=[1230,2173]

third2=[2365,2093]

clusters=['Kmeans','Agglm Cluster']

d=pd.DataFrame({'Clusters':clusters,'FirstC':first0,'SecondC':second1,'ThirdC':third2})

d
df=data.copy()
kmeans = KMeans(3, n_init=5, max_iter=100)

kmeans.fit(df)

df['label'] = kmeans.labels_

df.head()
df['label'].value_counts()
sns.pairplot(df,hue='label')

plt.show()
from sklearn.decomposition import PCA
pca=PCA()

pca.fit(data)
data_pca= pca.transform(data)

data_pca.shape
pca.components_
cumsum=np.cumsum(pca.explained_variance_ratio_)

cumsum
plt.figure(figsize=(10,6))



plt.plot(range(1,34), cumsum, color='k', lw=2)



plt.xlabel('Number of components')

plt.ylabel('Total explained variance')



plt.axvline(8, c='b')

plt.axhline(0.9, c='r')



plt.show()
pca = PCA(n_components=8)

pca.fit(data)

data_pca = pd.DataFrame(pca.transform(data))

data_pca.shape
sns.pairplot(data_pca, diag_kind='kde')

plt.show()
cluster_range = range(1,16)

cluster_errors = []



for num_clusters in cluster_range:

    clusters = KMeans(num_clusters, n_init=10, max_iter=100)

    clusters.fit(data_pca)

    

    cluster_errors.append(clusters.inertia_)

    

pd.DataFrame({'num_clusters':cluster_range, 'Error': cluster_errors})
plt.figure(figsize=(10,5))

plt.plot(cluster_range, cluster_errors, marker = "o" )

plt.title('Elbow Plot')

plt.xlabel('Number of Clusters')

plt.ylabel('Error')

plt.xticks(cluster_range)

plt.show()
pca_df = data_pca.copy()

kmeans = KMeans(3, n_init=10, max_iter=100)

kmeans.fit(pca_df)

pca_df['label'] = kmeans.labels_

pca_df['label'].value_counts()
plt.figure(figsize=(20,10))

link = linkage(data_pca, method='ward')

dendrogram(link, leaf_rotation=90, p=10, truncate_mode='level', leaf_font_size=6, color_threshold=8)

plt.title('Dendogram')

plt.show()
ac = AgglomerativeClustering(n_clusters=3, affinity='euclidean',  linkage='ward')

ac.fit(data_pca)
y_ac=ac.fit_predict(data_pca)
count(y_ac)
first0=[2226,2756]

second1=[1231,2379]

third2=[2363,685]

clusters=['Kmeans','Agglm Cluster']

d=pd.DataFrame({'Clusters':clusters,'FirstC':first0,'SecondC':second1,'ThirdC':third2})

d
df.head()
X=df.drop(columns='label')

y=df['label']
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=1)



print(Xtrain.shape)

print(Xtest.shape)

print(ytrain.shape)

print(ytest.shape)
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(Xtrain, ytrain)
print('Training score =', lr.score(Xtrain, ytrain))

print('Test score =', lr.score(Xtest, ytest))
ypred1=lr.predict(Xtest)
acc1=(metrics.accuracy_score(ytest,ypred1))

acc1
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(ytest, ypred1)

sns.heatmap(cm, annot=True, fmt='d')

plt.show()
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()

dt.fit(Xtrain, ytrain)



print('Training score =', dt.score(Xtrain, ytrain))

print('Test score =', dt.score(Xtest, ytest))
ypred2=dt.predict(Xtest)
acc2=(metrics.accuracy_score(ytest,ypred2))

acc2
cm = confusion_matrix(ytest, ypred2)

sns.heatmap(cm, annot=True, fmt='d')

plt.show()
from sklearn.neighbors import KNeighborsClassifier



score=[]

for k in range(1,100):

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(Xtrain, ytrain)

    ypred3=knn.predict(Xtest)

    accuracy=metrics.accuracy_score(ypred3,ytest)

    score.append(accuracy*100)

    print (k,': ',accuracy)
score.index(max(score))+1
round(max(score))
knn = KNeighborsClassifier(n_neighbors=9)

knn.fit(Xtrain, ytrain)



print('Training score =', knn.score(Xtrain, ytrain))

print('Test score =', knn.score(Xtest, ytest))
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

gnb.fit(Xtrain, ytrain)



print('Training score =', gnb.score(Xtrain, ytrain))

print('Test score =', gnb.score(Xtest, ytest))
Algorithm=['LogisticRegression','Decision Tree','KNN','Naive Bayes']

Train_Accuracy=[0.985,1.00,0.977,0.988]

Test_Accuracy=[0.975,0.939,0.963,0.988]
Before_PCA = pd.DataFrame({'Algorithm': Algorithm,'Train_Accuracy': Train_Accuracy,'Test_Accuracy':Test_Accuracy})

Before_PCA
df1=data_pca.copy()
kmeans = KMeans(3, n_init=5, max_iter=100)

kmeans.fit(df1)

df1['label'] = kmeans.labels_

df1.head()
X1=df1.drop(columns='label')

y1=df1['label']
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.3, random_state=1)



print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
lr_pca = LogisticRegression()

lr_pca.fit(X_train, y_train)

print('Training score =', lr_pca.score(X_train, y_train))

print('Test score =', lr_pca.score(X_test, y_test))
dt_pca = DecisionTreeClassifier()

dt_pca.fit(X_train, y_train)

print('Training score =', dt_pca.score(X_train, y_train))

print('Test score =', dt_pca.score(X_test, y_test))
score=[]

for k in range(1,100):

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train, y_train)

    ypred=knn.predict(X_test)

    accuracy=metrics.accuracy_score(ypred,y_test)

    score.append(accuracy*100)

    print (k,': ',accuracy)
score.index(max(score))+1
(max(score))
knn_pca = KNeighborsClassifier(n_neighbors=7)

knn_pca.fit(X_train, y_train)



print('Training score =', knn_pca.score(X_train, y_train))

print('Test score =', knn_pca.score(X_test, y_test))
gnb_pca = GaussianNB()

gnb_pca.fit(X_train, y_train)

print('Training score =', gnb_pca.score(X_train, y_train))

print('Test score =', gnb_pca.score(X_test, y_test))
Algorithm=['LogisticRegression','Decision Tree','KNN','Naive Bayes']

Train_Accuracy=[0.987,1.00,0.987,0.975]

Test_Accuracy=[0.979,0.995,0.980,0.967]
After_PCA = pd.DataFrame({'Algorithm': Algorithm,'Train_Accuracy': Train_Accuracy,'Test_Accuracy':Test_Accuracy})

After_PCA
Algorithm=['LR BPCA','DT BPCA','KNN BPCA','NB BPCA','LR APCA','DT APCA','KNN APCA','NB APCA']

Train_Accuracy=[0.985,1.00,0.977,0.988,0.987,1.00,0.987,0.975]

Test_Accuracy=[0.975,0.939,0.963,0.988,0.979,0.995,0.980,0.967]
Final = pd.DataFrame({'Algorithm': Algorithm,'Train_Accuracy': Train_Accuracy,'Test_Accuracy':Test_Accuracy})

Final
plt.subplots(figsize=(15,6))

sns.lineplot(x="Algorithm", y="Train_Accuracy",data=Final,palette='hot',label='Train Accuracy')

sns.lineplot(x="Algorithm", y="Test_Accuracy",data=Final,palette='hot',label='Test Accuracy')



plt.xticks(rotation=90)

plt.title('MLA Accuracy Comparison')

plt.legend()

plt.show()