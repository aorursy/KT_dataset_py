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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv('/kaggle/input/Classified Data',index_col=0)

df.head()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS', axis=1))
scaled_features = scaler.transform(df.drop('TARGET CLASS', axis=1))
scaled_features
df_scaled = pd.DataFrame(scaled_features, columns=df.columns[:-1])

df_scaled.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_features, df['TARGET CLASS'], test_size=0.30, random_state=42)
from sklearn.neighbors import KNeighborsClassifier
# K=1

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)

pred = knn.predict(X_test)
pred
y_test
pred != y_test
np.mean(pred != y_test)
pred == y_test
np.mean(pred == y_test)
from sklearn.metrics import classification_report, confusion_matrix
confusion_matrix(y_test, pred)
classification_report(y_test, pred)
error_rate = []

for i in range(1,40):

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train, y_train)

    pred = knn.predict(X_test)

    error_rate.append(np.mean(pred != y_test))
plt.figure(figsize=(10,6))

plt.plot(range(1,40), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)

plt.title("Error Rate vs K Value")

plt.xlabel("K Value")

plt.ylabel("Error Rate")
#With K=15



knn = KNeighborsClassifier(n_neighbors=15)

knn.fit(X_train, y_train)

pred = knn.predict(X_test)
confusion_matrix(y_test, pred)
classification_report(y_test, pred)