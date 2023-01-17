# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data1=pd.read_csv("/kaggle/input/biomechanical-features-of-orthopedic-patients/column_3C_weka.csv")

data2=pd.read_csv("/kaggle/input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")
data1.info()
data2.info()
data2.head()
data2["class"]=[1 if each == "Abnormal" else 0 for each in data2["class"]]



x_data=data2.drop(["class"],axis=1)



x= (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))
y=data2["class"].values
from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
from sklearn.neighbors import KNeighborsClassifier



knn=KNeighborsClassifier(n_neighbors=3)

knn.fit(x_train,y_train)

prediction=knn.predict(x_test)



print("{} nn score: {}".format(3,knn.score(x_test,y_test)))
score_list=[]

for each in range(1,15):

    knn2=KNeighborsClassifier(n_neighbors=each)

    knn2.fit(x_train,y_train)

    score_list.append(knn2.score(x_test,y_test))

    

plt.plot(range(1,15),score_list)

plt.xlabel("k")

plt.ylabel("accuracy")

plt.show()
from sklearn.neighbors import KNeighborsClassifier



knn=KNeighborsClassifier(n_neighbors=7)

knn.fit(x_train,y_train)

prediction=knn.predict(x_test)



print("{} nn score: {}".format(3,knn.score(x_test,y_test)))