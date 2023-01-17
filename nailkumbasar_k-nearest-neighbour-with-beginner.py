# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# KNN Algoritması Tutorial

#* import dataset

#* datasetimi tanımı

#* dataset görselleştirme

#* knn algoritması ne demek açıkla

#* knn with sklearn

#* ödev

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

# %%

data = pd.read_csv("../input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")



target_name = 'class'

data[target_name]

data
data.rename(columns={"pelvic_tilt numeric": "pelvic_tilt", "class": "siniflandir"}, inplace = True)

data.columns
data.info()
N = data[data.siniflandir == "Normal"]

A = data[data.siniflandir == "Abnormal"]

# scatter plot

plt.scatter(N.pelvic_incidence,N.lumbar_lordosis_angle,color="red",label="Normal",alpha= 0.3)

plt.scatter(A.pelvic_incidence,A.lumbar_lordosis_angle,color="green",label="Anormal",alpha= 0.3)

plt.xlabel("pelvic_incidence")

plt.ylabel("pelvic_tilt")

plt.legend()

plt.show()

data.siniflandir = [1 if each == "Normal" else 0 for each in data.siniflandir]

y = data.siniflandir.values

x_data = data.drop(["siniflandir"],axis=1)
# normalization 

x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))

x
# train test split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state=1)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3) # n_neighbors = k

knn.fit(x_train,y_train)

prediction = knn.predict(x_test)

print(" {} nn score: {} ".format(3,knn.score(x_test,y_test)))
# %%

# find best k value

score_list = []

for each in range(1,15):

    knn2 = KNeighborsClassifier(n_neighbors = each)

    knn2.fit(x_train,y_train)

    score_list.append(knn2.score(x_test,y_test))

    

plt.plot(range(1,15),score_list)

plt.xlabel("k values")

plt.ylabel("accuracy")

plt.show()
