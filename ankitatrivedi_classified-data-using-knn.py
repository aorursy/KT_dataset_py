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

        file = os.path.join(dirname, filename)



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv(file)

data.head(2)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit(data.drop('TARGET CLASS',axis =1))

sc_features = sc.transform(data.drop('TARGET CLASS',axis = 1))
data_feat = pd.DataFrame(sc_features,columns = data.columns[0:-1])

data_feat.head(2)
from sklearn.model_selection import train_test_split

X = data_feat

y = data['TARGET CLASS']

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.3 , random_state = 101)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)
pred = knn.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(pred,y_test))

print(confusion_matrix(pred,y_test))
error_rate = []

for i in range(1,40):

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train , y_train)

    pred_i = knn.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize = (10,6))

plt.plot(range(1,40),error_rate,color='blue',linestyle = 'dashed',marker = 'o',markerfacecolor='red',markersize=10)

plt.xlabel('K Value')

plt.title('Error Rate vs K value')
knn = KNeighborsClassifier(n_neighbors=18)

knn.fit(X_train, y_train)

pred = knn.predict(X_test)

print(classification_report(pred,y_test))

print(confusion_matrix(pred,y_test))
knn = KNeighborsClassifier(n_neighbors=15)

knn.fit(X_train, y_train)

pred = knn.predict(X_test)

print(classification_report(pred,y_test))

print(confusion_matrix(pred,y_test))