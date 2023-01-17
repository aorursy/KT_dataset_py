# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline  
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv("/kaggle/input/classified-data/Classified Data", index_col=0)
data.head()
data.columns
data.info()
data.isnull().sum().sum()
X=data.iloc[0:,0:10]
X.head()
Y=data["TARGET CLASS"]
X.shape, Y.shape
count_classes=pd.value_counts(data["TARGET CLASS"])
count_classes
data["TARGET CLASS"].value_counts()
count_classes.plot(kind="bar", rot=0)
plt.title("Transaction class distribution")
plt.xlabel("Target Class")
plt.ylabel("frequency")
plt.xticks(range(2),labels=["Yes","No"])
#check balance data
from sklearn.preprocessing import StandardScaler
stc=StandardScaler()
X_scaled=stc.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X_scaled,Y, random_state=15, test_size=0.25)
X_train.shape, X_test.shape, Y_train.shape, Y_test.shape
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,Y_train)
pred=knn.predict(X_test)
pred
knn.score(X_test, Y_test)
from sklearn.metrics import confusion_matrix, classification_report
cmt=confusion_matrix(Y_test, pred)
cmt
print(classification_report(Y_test,pred))
err_rate=[]
for i in range(1,40):
  knn=KNeighborsClassifier(n_neighbors=i)
  knn.fit(X_train,Y_train)
  pred_i=knn.predict(X_test)
  err_rate.append(np.mean(pred_i != Y_test))
err_rate
plt.figure(figsize=(10,6))
plt.plot(range(1,40),err_rate,color='blue', linestyle='dashed', marker='o',markerfacecolor='red', markersize=10)
plt.title("Error Rate")
plt.xlabel("K value")
plt.ylabel("error")
k_value=np.array(err_rate).argmin()+1
k_value
knn=KNeighborsClassifier(n_neighbors=k_value)
knn.fit(X_train,Y_train)
pred_k_value=knn.predict(X_test)
knn.score(X_test,Y_test)
cmt=confusion_matrix(Y_test, pred_k_value)
cmt
print(classification_report(Y_test,pred_k_value))
from collections import Counter
print("Y_test: {}".format(Counter(Y_test)))
print("Knn Prediction: {}".format(Counter(pred_k_value)))
