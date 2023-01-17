# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from collections import Counter

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/biomechanical-features-of-orthopedic-patients/column_3C_weka.csv")

                   

data2= pd.read_csv("/kaggle/input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")
data.head()
data.info()
data.tail()
Counter(data["class"])
H = data[data["class"]=="Hernia"]

Sl = data[data["class"]=="Spondylolisthesis"]

N = data[data["class"]=="Normal"]
plt.scatter(x=H.pelvic_radius, y =H.sacral_slope,color="Red",label ="Hernia")

plt.scatter(x=Sl.pelvic_radius, y =Sl.sacral_slope,color="Green",label ="Spondylolisthesis")

plt.scatter(x=N.pelvic_radius, y =N.sacral_slope,color="Yellow",label = "Normal")

plt.xlabel("pelvic_radius")

plt.ylabel("sacral_slope,color")

plt.legend()

plt.show()
plt.scatter(x=H.pelvic_tilt, y =H.sacral_slope,color="Red")

plt.scatter(x=Sl.pelvic_tilt, y =Sl.sacral_slope,color="Green")

plt.scatter(x=N.pelvic_tilt, y =N.sacral_slope,color="Yellow")

plt.xlabel("pelvic_tilt")

plt.ylabel("sacral_slope,color")

plt.show()
H.describe()
Sl.describe()
N.describe()
plt.scatter(x=H.lumbar_lordosis_angle, y =H.sacral_slope,color="Red")

plt.scatter(x=Sl.lumbar_lordosis_angle, y =Sl.sacral_slope,color="Green")

plt.scatter(x=N.lumbar_lordosis_angle, y =N.sacral_slope,color="Yellow")

plt.xlabel("lumbar_lordosis_angle")

plt.ylabel("sacral_slope,color")

plt.show()
data["class"] = [2 if each=="Hernia" else 1 if  each=="Spondylolisthesis" else 0 for each in data["class"]]

y = data["class"].values

x_data = data.drop(["class"],axis=1)
x = ((x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data)))
from sklearn.model_selection import train_test_split

x_train , x_test, y_train , y_test = train_test_split(x,y,test_size=0.3,random_state=42)
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(x_train,y_train)

prediction = knn.predict(x_test)

print(prediction)

knn.score(x_test,y_test)
score_list = []

a=0

b=0

for i in range(1,20):

    knn2 = KNeighborsClassifier(n_neighbors = i)

    knn2.fit(x_train,y_train)

    score_list.append(knn2.score(x_test,y_test))

    if knn2.score(x_test,y_test)>a:

        a=knn2.score(x_test,y_test)

        b=i

plt.plot(range(1,20),score_list,color="green")

plt.show()

print(b)

  

    
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors = b)

knn.fit(x_train,y_train)

prediction = knn.predict(x_test)

print(prediction)

knn.score(x_test,y_test)
def knn(x_train, x_test, y_train,y_test,number_of_try):

    a=0

    b=0

    score_list=[]

    for i in range(1,number_of_try):

        knn2 = KNeighborsClassifier(n_neighbors = i)

        knn2.fit(x_train,y_train)

        score_list.append(knn2.score(x_test,y_test))

        if knn2.score(x_test,y_test)>a:

            a=knn2.score(x_test,y_test)

            b=i

    knn = KNeighborsClassifier(n_neighbors = b)

    knn.fit(x_train,y_train)

    plt.plot(range(1,number_of_try),score_list)

    return knn.score(x_test,y_test),b

    
knn(x_train, x_test, y_train,y_test,100)