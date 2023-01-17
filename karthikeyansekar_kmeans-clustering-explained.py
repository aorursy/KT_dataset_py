# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



sns.set()



%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



RANDOM_STATE = 0



files = []



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        files.append(os.path.join(dirname, filename))

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv(files[1])

df.head()
df.info()
X = df.iloc[:, 1:5].values

y = df.iloc[:, 5].values # Not required, just in case to validate our Unsupervised Learning skill
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = RANDOM_STATE, stratify = y)
from sklearn.cluster import KMeans
wcss = []



for i in range(1,15):

    kmeans = KMeans(n_clusters = i, init='k-means++', random_state = RANDOM_STATE)

    kmeans.fit(X_train)

    wcss.append(kmeans.inertia_)



plt.title('Elbow Method')

plt.plot(range(1,15), wcss)

plt.xlabel('No of Clusters')

plt.ylabel('WCSS')

plt.show()
kmeans = KMeans(n_clusters = 3, init='k-means++', random_state = RANDOM_STATE)

kmeans.fit(X_train)

y_pred = kmeans.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

y_test = labelencoder.fit_transform(y_test)

y_test
y_pred
accuracy_score(y_test,y_pred)
# Lets re-name the cluster numbers in prediction data

y_pred[y_pred == 0] = 3

y_pred[y_pred == 1] = 0

y_pred[y_pred == 3] = 1
accuracy_score(y_test,y_pred)