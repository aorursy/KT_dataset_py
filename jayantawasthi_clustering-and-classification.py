# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
%matplotlib inline
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv("/kaggle/input/uci-turkiye-student-evaluation-data-set/turkiye-student-evaluation_generic.csv")
train.head()
sns.countplot(x="class",data=train)
sns.countplot(x="class",hue="nb.repeat",data=train)
sns.countplot(x="nb.repeat",hue="difficulty",data=train)
sns.countplot(x="difficulty",hue="nb.repeat",data=train)
sns.countplot(x="difficulty",hue="attendance",data=train)
sns.countplot(x="class",hue="instr",data=train)
sns.countplot(x="class",hue="difficulty",data=train)
for i in train.columns:
    train.hist(i)
    plt.show()
X_res=train.values
X_res
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
features=scaler.fit_transform(X_res)
features
from sklearn.cluster import KMeans
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
y_pred = kmeans.fit_predict(features)
kmeans.labels_
kmeans.inertia_
m=[]
for k in range(1,80):
    km=KMeans(n_clusters=k, random_state=42).fit(features)
    m.append(km.inertia_)

m
plt.figure(figsize=(20,10))
plt.plot(range(1, 80),m, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Inertia", fontsize=14)
plt.show()
l=[]
p=[]
for k in range(1,10):
    km=KMeans(n_clusters=k, random_state=42).fit(features)
    l.append(km.inertia_)
    p.append(km)
plt.figure(figsize=(20,10))
plt.plot(range(1, 10),l, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Inertia", fontsize=14)
plt.show()
from sklearn.metrics import silhouette_score
silhouette_scores =[]
for model in p[1:]:
    s=silhouette_score(features, model.labels_)
    silhouette_scores.append(s)
    
plt.figure(figsize=(20, 10))
plt.plot(range(1, 9),silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score", fontsize=14)

plt.show()

from sklearn.decomposition import PCA
pca = PCA(n_components = 2)

f=pca.fit_transform(features)
f
l=[]
p=[]
for k in range(1,10):
    km=KMeans(n_clusters=k, random_state=42).fit(features)
    l.append(km.inertia_)
    p.append(km)
plt.figure(figsize=(20,10))
plt.plot(range(1, 10),l, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Inertia", fontsize=14)
plt.show()
silhouette_scores =[]
for model in p[1:]:
    s=silhouette_score(f, model.labels_)
    silhouette_scores.append(s)
plt.figure(figsize=(20, 10))
plt.plot(range(1, 9),silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score", fontsize=14)
plt.show()  
kmeans = KMeans(n_clusters=4, random_state=42)
y_pred = kmeans.fit_predict(f)
t=kmeans.labels_
plt.figure(figsize=(8, 6))
plt.scatter(f[:,0], f[:,1], c=kmeans.labels_.astype(float))
plt.show()
from sklearn.model_selection import train_test_split

np.random.seed(1234)

x_train,x_test,y_train,y_test = train_test_split(f,t, train_size=0.80, random_state=42)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
rnd_clf = RandomForestClassifier(n_estimators=200, max_leaf_nodes=8, random_state=42)
cv_scores = cross_val_score(rnd_clf, x_train, y_train, cv=10)
cv_scores.mean()
rnd_clf.fit(x_train,y_train)
rnd_clf.score(x_test,y_test)
n_estimator=[10,100,400,500,600,700,800,2000]
max_leaf_nodes=[2,4,6,8,16]
for i in n_estimator:
    for k in max_leaf_nodes:
       rnd_clf = RandomForestClassifier(n_estimators=i, max_leaf_nodes=k, random_state=42)
       rnd_clf.fit(x_train, y_train)
       t=rnd_clf.score(x_test,y_test)
       print(i,k,t)
    print("-"*40)