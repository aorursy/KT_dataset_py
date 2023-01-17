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
d1  = pd.read_csv("../input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")

d2  = pd.read_csv("../input/biomechanical-features-of-orthopedic-patients/column_3C_weka.csv")
df = pd.concat([d1,d2],axis =0)
df.head()
df.info()
df.drop(["pelvic_tilt","pelvic_tilt numeric"],axis=1 , inplace = True)
Normal = df[df["class"] == "Normal"]

Abnormal = df[df["class"] == "Abnormal"]
plt.scatter(Normal.pelvic_incidence,Normal.pelvic_radius,color = "b" , label = "normal")

plt.scatter(Abnormal.pelvic_incidence,Abnormal.pelvic_radius,color = "c" , label = "abnormal")

plt.xlabel("pelvic_incidence")

plt.ylabel("pelvic_radius")

plt.legend()

plt.show()
df["class"] = [1 if each == "Normal" else 0 for each in df["class"]] 

y = df["class"].values

x_data = df.drop(["class"],axis = 1)
#normalization

x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))
from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.3 , random_state = 1)

#knn model 

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 1) #n_neighbors = k

knn.fit(x_train , y_train)

prediction = knn.predict(x_test)
print("{} knn score : {} ".format(3,knn.score(x_test,y_test)))
# find k value

score_list = []

for each in range(1,15):

    knn2 = KNeighborsClassifier(n_neighbors = each)

    knn2.fit(x_train , y_train)

    score_list.append(knn2.score(x_test,y_test))

    

plt.plot(range(1,15),score_list)

plt.xlabel("k values")

plt.ylabel("accuracy")

plt.show()