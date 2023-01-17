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
data=pd.read_csv("../input/column_2C_weka.csv")
data.head()
data.info()
data=data.rename(columns={"class": "degree"})

data.head()
A=data[data.degree=="Abnormal"]

N=data[data.degree=="Normal"]
#scatter plot

plt.scatter(A.pelvic_radius,A.lumbar_lordosis_angle,color="red",label="anormal")

plt.scatter(N.pelvic_radius,N.lumbar_lordosis_angle,color="green",label="normal")

plt.xlabel("pelvic_radius")

plt.ylabel("lumbar_lordosis_angle")

plt.legend()

plt.show()
#binary sisteme çevir

data.degree=[1 if each=="Normal" else 0 for each in data.degree]

x_data=data.drop(["degree"],axis=1)

y=data.degree.values
#normalization

x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))
#train test split

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
#knn model

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=3)# n_neighbors=k

knn.fit(x_train,y_train)

prediction=knn.predict(x_test)

print(" {} nn score: {}".format(3,knn.score(x_test,y_test)))
#en iyi değer



score_list=[]

for each in range(1,50):

    knn= KNeighborsClassifier(n_neighbors=each) # n_neighbors=k

    knn.fit(x_train,y_train)

    score_list.append(knn.score(x_test,y_test))

    

plt.plot(range(1,50),score_list)

plt.xlabel("k değerleri")

plt.ylabel("başarı değerleri")

plt.show()