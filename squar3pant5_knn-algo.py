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
data=pd.read_csv("../input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")
#İnformation dataset

data.info()

#data.head()

#data.tail()

#data.columns
Abnormal = data[data["class"] == "Abnormal"]

Normal = data[data["class"] == "Normal"]
#normalization

data["class"]=[1 if each=="Normal" else 0 for each in data["class"]]

#value counts

data["class"].value_counts()
x=data.drop(["class"],axis=1)

y=data["class"].values
#x

#y
#train test split

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#create Knn model

from sklearn.neighbors import KNeighborsClassifier

knn= KNeighborsClassifier(n_neighbors=3)#n_neigbors=k

knn.fit(x_train,y_train)

knn.predict(x_test)
print("{} nn score: {}".format(3,knn.score(x_test,y_test)))
#find best k values

score_list=[]

for each in range(1,25):

    knn2=KNeighborsClassifier(n_neighbors=each)

    knn2.fit(x_train,y_train)

    knn2.score(x_test,y_test)

    score_list.append(knn2.score(x_test,y_test))



plt.plot(range(1,25),score_list)

plt.xlabel("k values",color="blue")

plt.ylabel("accuracy",color="blue")

plt.show()
#if accuracy>85 k values are best values