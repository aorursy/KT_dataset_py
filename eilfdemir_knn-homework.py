# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/wisc_bc_data.csv")
data.head(10)
data.tail(10)
data["diagnosis"].unique()
data.info()
B = data[data["diagnosis"] == "B"]
M = data[data["diagnosis"] == "M"]
plt.scatter(B.radius_worst,B.texture_worst,color = "purple",label = "B")
plt.scatter(M.radius_worst,M.texture_worst,color = "blue",label = "M")
plt.xlabel("radius_worst")
plt.ylabel("texture_worst")
plt.legend()
plt.show()
data["diagnosis"] = [1 if each == "B" else 0 for each in data["diagnosis"] ]
data.head()
y = data["diagnosis"].values
x_data = data.loc[:,data.columns != 'diagnosis']
x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data))
from sklearn.model_selection import train_test_split


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3, random_state=1)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3) 
knn.fit(x_train,y_train) 
knn.predict(x_test)  
print("{} nn score:{}".format(3,knn.score(x_test,y_test)))
score_list = []  
for each in range(1,25):
    knn2 = KNeighborsClassifier(n_neighbors = each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))
    
plt.plot(range(1,25),score_list)
plt.xlabel("k")
plt.ylabel("accuracy")
plt.show()