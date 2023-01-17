# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split 

from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")
data.tail()
data.fillna(0)
data["class"] = [1 if(each == "Abnormal") else 0 for each in data["class"]]
y=data["class"].values

y
x_data=data.drop(["class"],axis=1)
#normalization

x=(x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))
# train test split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
knn=KNeighborsClassifier(n_neighbors=5)

knn.fit(x_train,y_train)

prediction=knn.predict(x_test)
print(knn.score(x_test,y_test))
# optimum k

score_list=[]

for each in range(1,20):

    knn2=knn=KNeighborsClassifier(n_neighbors=each)

    knn2.fit(x_train,y_train)

    score_list.append(knn2.score(x_test,y_test))

plt.plot(range(1,20),score_list)

plt.xlabel("k")

plt.ylabel("accuracy")

plt.show()
knn=KNeighborsClassifier(n_neighbors=18)

knn.fit(x_train,y_train)

prediction=knn.predict(x_test)
print(knn.score(x_test,y_test))