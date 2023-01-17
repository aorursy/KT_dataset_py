# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")
data.info()
# "data.class can not be used, so change column name

data = data.rename({'class': 'cl'}, axis=1)
a = data[data.cl == "Abnormal"]

n = data[data.cl == "Normal"]
#scatter plot 

plt.scatter(a.pelvic_incidence,a.pelvic_radius,color="red",label="abnormal",alpha=0.5)

plt.scatter(n.pelvic_incidence,n.pelvic_radius,color="green",label="normal",alpha=0.5)

plt.xlabel("pelvic_incidence")

plt.ylabel("pelvic_radius")

plt.legend()

plt.show()
data.cl = [1 if each == "Abnormal" else 0 for each in data.cl]
y = data.cl.values

x2 = data.drop(["cl"],axis=1)
# normalization

x = (x2 - np.min(x2))/(np.max(x2)-np.min(x2))
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=24)
# knn model

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 7) # n_neighbors = k

knn.fit(x_train,y_train)

prediction = knn.predict(x_test)
print("if k is {} \nknn score is {}".format(7,knn.score(x_test,y_test)))
# find k value

score_list = []

for each in range(1,15):

    knn2 = KNeighborsClassifier(n_neighbors = each)

    knn2.fit(x_train,y_train)   

    score_list.append(knn2.score(x_test,y_test))

    

plt.plot(range(1,15),score_list)

plt.xlabel("k values")

plt.ylabel("accuracy")

plt.show()
def knnvalue(n_neighbors):

    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(n_neighbors) # n_neighbors = k

    knn.fit(x_train,y_train)

    prediction = knn.predict(x_test)

    print("if k is {} \nknn score is {}".format(n_neighbors,knn.score(x_test,y_test)))
knnvalue(5)