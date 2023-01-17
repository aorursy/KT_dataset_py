# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/regression-kernel/column_2C_weka.csv")
data.tail()
data.loc[:,'class'] = [1 if each == "Abnormal" else 0 for each in data.loc[:,'class']]

y = data.loc[:,'class']

x_data = data.drop(["class"],axis=1)
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))
# train test split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state=1)
# knn model

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 25) # n_neighbors = k

knn.fit(x_train,y_train)

prediction = knn.predict(x_test)

print(" {} nn score: {} ".format(25,knn.score(x_test,y_test)))
# find k value

score_list = []

for each in range(1,150):

 knn2 = KNeighborsClassifier(n_neighbors = each)

 knn2.fit(x_train,y_train)

 score_list.append(knn2.score(x_test,y_test))
plt.plot(range(1,150),score_list)

plt.xlabel("k values")

plt.ylabel("accuracy")

plt.show()
from sklearn.svm import SVC

svm = SVC(random_state = 1)

svm.fit(x_train,y_train)

# %% test

print("print accuracy of svm algo: ",svm.score(x_test,y_test))
# %% Naive bayes

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train,y_train)

# %% test

print("print accuracy of naive bayes algo: ",nb.score(x_test,y_test))
#%%

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()

dt.fit(x_train,y_train)

print("score: ", dt.score(x_test,y_test))
#%% random forest

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 100,random_state = 1)

rf.fit(x_train,y_train)

print("random forest algo result: ",rf.score(x_test,y_test))
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 100,random_state = 1)

rf.fit(x_train,y_train)

print("random forest algo result: ",rf.score(x_test,y_test))

y_pred = rf.predict(x_test)

y_true = y_test

#%% confusion matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true,y_pred)

# %% cm visualization

import seaborn as sns

import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()
data2 = data.loc[:,['degree_spondylolisthesis','pelvic_radius']]

from sklearn.cluster import KMeans

wcss = []

for k in range(1,15):

 kmeans = KMeans(n_clusters=k)

 kmeans.fit(data2)

 wcss.append(kmeans.inertia_)

plt.plot(range(1,15),wcss)

plt.xlabel("number of k (cluster) value")

plt.ylabel("wcss")

plt.show()
kmeans2 = KMeans(n_clusters=4)

clusters = kmeans2.fit_predict(data)

labels = clusters

plt.scatter(data['pelvic_radius'],data['degree_spondylolisthesis'],c = labels) #c=color

plt.xlabel('pelvic_radius')

plt.ylabel('degree_spondylolisthesis')

plt.show()
# %% dendogram

from scipy.cluster.hierarchy import linkage, dendrogram

merg = linkage(data2,method="ward")

dendrogram(merg,leaf_rotation = 90)

plt.xlabel("data points")

plt.ylabel("euclidean distance")

plt.show()
# %% HC

from sklearn.cluster import AgglomerativeClustering

hiyerartical_cluster = AgglomerativeClustering(n_clusters = 3,affinity= "euclidean",linkage = "ward")

cluster = hiyerartical_cluster.fit_predict(data)

labels2 = cluster

plt.scatter(data['pelvic_radius'],data['degree_spondylolisthesis'],c = labels) #c=color

plt.xlabel('pelvic_radius')

plt.ylabel('degree_spondylolisthesis')

plt.show()