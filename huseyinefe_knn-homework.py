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
data.info()
data.head(10)
data["class"] = [1 if each == "Abnormal" else 0 for each in data["class"]]
data.head()
y=data["class"].values

x_data=data.drop(["class"],axis=1)
#normalization

x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(x_train,y_train)

prediction = knn.predict(x_test)
print("{}, nn accuracy {}".format(3,knn.score(x_test,y_test)))
score_list = []

for each in range(1,15):

    knn2=KNeighborsClassifier(n_neighbors=each)

    knn2.fit(x_train,y_train)

    score_list.append(knn2.score(x_test,y_test))



 #visualization   

plt.figure(figsize=(10,10))

plt.plot(range(1,15),score_list,color="black")

plt.legend()

plt.xlabel=('n_neigbors values')

plt.ylabel=('accuracy')

plt.show()

    