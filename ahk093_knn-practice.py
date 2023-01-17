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
df = pd.read_csv("../input/column_2C_weka.csv")
df.info() # we can see any missing data in this dataset which is good for us
df.head()
A = df[df["class"] == "Abnormal"]
N = df[df["class"] == "Normal"]
plt.scatter(A.pelvic_radius,A.sacral_slope,color = "red",label = "Abnormal")
plt.scatter(N.pelvic_radius,N.sacral_slope,color = "green",label = "Normal")
plt.xlabel("pelvic_radius")
plt.ylabel("sacral_slope")
plt.legend()


fig, axes = plt.subplots(1, 2, figsize=(13.3,4))
      
axes[0].scatter(A.pelvic_radius, A.sacral_slope,color = "red")
axes[0].set_xlabel("pelvic_radius")
axes[0].set_ylabel("sacral_slope")
axes[0].set_title("Abnormal")

axes[1].scatter(N.pelvic_radius, N.sacral_slope,color = "green")
axes[1].set_xlabel("pelvic_radius")
axes[1].set_ylabel("sacral_slope")
axes[1].set_title("Normal")


plt.show()


plt.scatter(A.pelvic_radius,A.lumbar_lordosis_angle,color = "red",label = "Abnormal", alpha = 0.4)
plt.scatter(N.pelvic_radius,N.lumbar_lordosis_angle,color = "green",label = "Normal", alpha = 0.4)
plt.xlabel("pelvic_radius")
plt.ylabel("lumbar_lordosis_angle")
plt.legend()
plt.show()
# Binary translation

df["class"] = [1 if each == "Abnormal" else 0 for each in df["class"]]
y = df["class"].values
x = df.drop(["class"], axis = 1)

#Normalization
#x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))
#train_test_split

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3, random_state = 1)
#knn model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
print("{}nn score: {}".format(3,knn.score(x_test,y_test)))
score_list = []
for each in range(1,20):
    knn2 =  KNeighborsClassifier(n_neighbors= each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))
    
plt.plot(range(1,20),score_list)
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.show()
knn = KNeighborsClassifier(n_neighbors = 17)
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
print("{}nn score: {}".format(17,knn.score(x_test,y_test)))
