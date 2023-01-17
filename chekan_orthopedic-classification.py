# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/column_2C_weka.csv")
data.info()
data.head()
data.describe()
y=data["class"].values #class
x_data=data.drop(["class"],axis=1) #features
x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data)) #normalization
y=[1 if each=="Abnormal" else 0 for each in data["class"]]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42) #splitting train and test data
scores=[]
for each in range(1,18):
    knn_t=KNeighborsClassifier(n_neighbors=each)
    knn_t.fit(x_train,y_train)
    scores.append(knn_t.score(x_test,y_test))
plt.plot(range(1,18),scores)
plt.xlabel("K values")
plt.ylabel("Scores")
plt.show()
knn=KNeighborsClassifier(n_neighbors=15)
knn.fit(x_train,y_train)
print("test accuracy is ",knn.score(x_test,y_test))
