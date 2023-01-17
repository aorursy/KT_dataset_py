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
# import data
data = pd.read_csv("../input/column_2C_weka.csv")
data.head()
data.tail()
data["class"].unique()
data.info()
# create new two datas
A = data[data["class"] == "Abnormal"]
N = data[data["class"] == "Normal"]
# visualize
plt.scatter(A.pelvic_radius,A.sacral_slope,color = "red",label = "Abnormal")
plt.scatter(N.pelvic_radius,N.sacral_slope,color = "blue",label = "Normal")
plt.xlabel("pelvic_radius")
plt.ylabel("sacral_slope")
plt.legend()
plt.show()
# class column's type change to integer.
data["class"] = [1 if each == "Abnormal" else 0 for each in data["class"] ]
data.head()
y = data["class"].values
x_data = data.loc[:,data.columns != 'class']
# normalization
x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data))
# train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3, random_state=1)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)  #n_neighbors = k
knn.fit(x_train,y_train)  # the model is creating
knn.predict(x_test)  # prediction 
print("{} nn score:{}".format(3,knn.score(x_test,y_test)))
# find k value
score_list = []  # we store the values we find in this list.
for each in range(1,25):
    knn2 = KNeighborsClassifier(n_neighbors = each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))
    
# visualize
plt.plot(range(1,25),score_list)
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.show()