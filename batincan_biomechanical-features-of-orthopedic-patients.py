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
data = pd.read_csv("/kaggle/input/biomechanical-features-of-orthopedic-patients/column_3C_weka.csv")

data2 = pd.read_csv("/kaggle/input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")

data.info()

data.head()
H = data[data["class"]=="Hernia"]

H.info()

S = data[data["class"]=="Spondylolisthesis"]

S.info()
N=data[data["class"]=="Normal"]

N.info()
#Visuluation

plt.scatter(H.pelvic_tilt,H.pelvic_incidence,color="red",label="Hernia")

plt.scatter(S.pelvic_tilt,S.pelvic_incidence,color="green",label="Spondylolisthesis")

plt.xlabel("pelvic_tilt")

plt.ylabel("pelvic_incidence")

plt.legend()

plt.show()
plt.scatter(N.pelvic_tilt,N.pelvic_incidence,color="red",label="Normal")

plt.scatter(S.pelvic_tilt,S.pelvic_incidence,color="green",label="Spondylolisthesis")

plt.xlabel("pelvic_tilt")

plt.ylabel("pelvic_incidence")

plt.legend()

plt.show()
plt.scatter(H.pelvic_tilt,H.pelvic_incidence,color="red",label="Hernia")

plt.scatter(N.pelvic_tilt,N.pelvic_incidence,color="blue",label="Normal")

plt.xlabel("pelvic_tilt")

plt.ylabel("pelvic_incidence")

plt.legend()

plt.show()
plt.scatter(H.pelvic_tilt,H.pelvic_incidence,color="red",label="Hernia",alpha=0.4)

plt.scatter(N.pelvic_tilt,N.pelvic_incidence,color="blue",label="Normal",alpha=0.4)

plt.scatter(S.pelvic_tilt,S.pelvic_incidence,color="green",label="Spondylolisthesis",alpha=0.4)

plt.xlabel("pelvic_tilt")

plt.ylabel("pelvic_incidence")

plt.legend()

plt.show()
data["class"]=[1 if each=="Hernia" else 2 if each=="Spondylolisthesis" else 0 for each in data["class"] ]

y=data["class"].values

x_data=data.drop(["class"],axis=1)

y

#normalization

x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))
#train-test split

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
#knn model

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=3) # k=değeri

knn.fit(x_train,y_train)

prediction=knn.predict(x_test)

prediction
print("{} knn score: {}".format(3,knn.score(x_test,y_test)))
score_list=[]

for each in range(1,150):

    knn2=KNeighborsClassifier(n_neighbors=each)

    knn2.fit(x_train,y_train)

    score_list.append(knn2.score(x_test,y_test))

    

plt.plot(range(1,150),score_list)

plt.xlabel("k values")

plt.ylabel("score-accuracy")
max_accuracy_value= max(score_list) 

max_accuracy_k = score_list.index(max_accuracy_value)+1 

print("for max score ==> k={},score={}".format(max_accuracy_k,max_accuracy_value))