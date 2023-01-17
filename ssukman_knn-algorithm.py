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


import pandas as pd

data = pd.read_csv("../input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")
data.head()
data.info()
Abnormal = data[data["class"] == "Abnormal"]

Normal   = data[data["class"] == "Normal"]

plt.scatter(Abnormal.pelvic_incidence,Abnormal.pelvic_radius,color="red",label="Abnormal",alpha= 0.4)

plt.scatter(Normal.pelvic_incidence,Normal.pelvic_radius,color="green",label="Normal",alpha= 0.4)

plt.xlabel("pelvic_incidence")

plt.ylabel("pelvic radius")

plt.legend()

plt.show()
#reduce class values 



data["class"]= [1 if each == "Abnormal" else 0 for each in data["class"]]

data.info()
y = data["class"].values

x_data = data.drop(["class"],axis=1)

#Normalization data values

#x = (x_data- np.min(x_data))/(np.max(x_data)-np.min(x_data))

x_data.head()
#Train Test Split

from sklearn.model_selection import train_test_split

x_train,x_test,y_train, y_test = train_test_split(x_data,y,test_size = 0.3, random_state = 1)

#KNN MODEL

from sklearn.neighbors import KNeighborsClassifier



KNN = KNeighborsClassifier(n_neighbors= 18)

KNN.fit(x_train,y_train)

prediction = KNN.predict(x_test)

print(" {} nn score: {} ".format(18,KNN.score(x_test,y_test)))

# Find Best K value

scorelist = []

for each in range(1,25):

    Knn1 = KNeighborsClassifier(n_neighbors = each)

    Knn1.fit(x_train,y_train)

    scorelist.append(Knn1.score(x_test,y_test))





plt.plot(range(1,25),scorelist)

plt.xlabel("K Values")

plt.ylabel("Accuracy")

plt.figure(figsize=[13,8])

plt.show()

print("Best accuracy is {} with K = {}".format(np.max(scorelist),1+scorelist.index(np.max(scorelist))))