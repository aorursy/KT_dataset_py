# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/Classified Data")
data.head()
data.info()
data.describe()
data.columns

data_copy = data
data_copy = data_copy.drop("Unnamed: 0", axis=  1)
data_copy.head()
data.tail()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(data_copy.drop("TARGET CLASS", axis= 1))
scaled_data = scaler.transform(data_copy.drop("TARGET CLASS", axis = 1))
scaled_data_final= pd.DataFrame(scaled_data, columns = data_copy.columns[:-1])
X = scaled_data_final
y = data_copy["TARGET CLASS"]
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 123 )
knn = KNeighborsClassifier()
knn.fit(X, y)
prediction_knn = knn.predict(X_test)
from sklearn import metrics
metrics.accuracy_score(y_test, prediction_knn)
knn15 = KNeighborsClassifier(n_neighbors= 15)
knn15.fit(X_train, y_train)
prediction_knn15 = knn15.predict(X_test)
metrics.accuracy_score(y_test, prediction_knn15)
error = []

for i in range(1,50):

    knn = KNeighborsClassifier(n_neighbors= i)

    knn.fit(X_train, y_train)

    pred_i = knn.predict(X_test)

    error.append(np.mean(pred_i != y_test))
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=  (11,8))

plt.plot(range(1,50), error )