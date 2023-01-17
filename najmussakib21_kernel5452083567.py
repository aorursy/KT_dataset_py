# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



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
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.cluster import KMeans

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,confusion_matrix
data = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')

data.head(5)
bins = (2,6.5,8)

labels = ['bad','good']

data['quality'] = pd.cut(data['quality'],bins=bins,labels=labels)
data['quality']
le = LabelEncoder()

data['quality'] = le.fit_transform(data['quality'])
X = data.drop('quality',axis=1).values

y = data['quality'].values.reshape(-1,1)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)
print("X_train shape: ",X_train.shape)

print("X_test shape: ",X_test.shape)

print("y_train shape: ",y_train.shape)

print("y_test shape: ",y_test.shape)
lreg = LogisticRegression().fit(X_train,y_train)
lreg_pred = lreg.predict(X_test)
lreg_cm = confusion_matrix(lreg_pred,y_test)

ax = sns.heatmap(lreg_cm,annot=True)

ax.set(xlabel='predict', ylabel='true')

lreg_as = accuracy_score(lreg_pred,y_test)

print("logistic regression accuracy score: ",lreg_as)
KN = KNeighborsClassifier(n_neighbors=5)
KN.fit(X_train,y_train)
kn_pred = KN.predict(X_test)
kn_pred
kn_cm = confusion_matrix(kn_pred,y_test)

ax = sns.heatmap(kn_cm,annot=True)

ax.set(xlabel='predict', ylabel='true')

kn_as = accuracy_score(kn_pred,y_test)

print("KNearest neighbors accuracy score: ",kn_as)
kmeans = KMeans(n_clusters=2).fit(X_test)
kmeans_predict = kmeans.predict(X_test)
km_cm = confusion_matrix(kmeans_predict,y_test)

ax = sns.heatmap(km_cm,annot=True)

ax.set(xlabel='predict', ylabel='true')

km_as = accuracy_score(kmeans_predict,y_test)

print("KMeans clustering accuracy score: ",km_as)
plt.scatter(X_test[kmeans_predict == 0, 0], X_test[kmeans_predict == 0, 1], s = 100, c = 'red', label = 'Cluster 0')

plt.scatter(X_test[kmeans_predict == 1, 0], X_test[kmeans_predict == 1, 1], s = 100, c = 'blue', label = 'Cluster 1')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')



plt.legend()

plt.show()