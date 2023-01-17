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

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

cc = pd.read_csv(os.path.join(dirname, filename))

cc.head()

cc = cc.apply(pd.to_numeric, errors='coerce')
cc.dropna(inplace=True)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(cc.drop('Class', axis=1))
scaled_features = scaler.transform(cc.drop('Class', axis=1))
cc_feat = pd.DataFrame(scaled_features,columns=cc.columns[:-1])

from sklearn.model_selection import train_test_split
X = cc_feat
y = cc['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))

# To identify optimal K value, run albo mechanism
# run predictions for multiple K values
"""
error_rate = []

for i in range(1,10):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test)) # calculate error when pred value != test value, take mean using numpy method mean

plt.figure(figsize=(10,6))
plt.plot(range(1,10),error_rate,color='blue',linestyle='dashed',marker='o',markerfacecolor='red',markersize=10)
plt.title('Error Rate vs Value')
plt.xlabel('K')
plt.ylabel('Error rate')
"""
    
