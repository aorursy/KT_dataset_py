# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# import warnings

import warnings

# ignore warnings

warnings.filterwarnings("ignore")

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# read csv (comma separated value) into data

data = pd.read_csv('../input/column_2C_weka.csv')

data.head()
#summary info about data

A = data[data["class"] == "Abnormal"]

N = data[data["class"] == "Normal"]

plt.scatter(A.pelvic_incidence, A.pelvic_radius, color="red", label = "Abnormal")

plt.scatter(N.pelvic_incidence, N.pelvic_radius, color="blue", label = "Normal")

plt.legend()

plt.xlabel("pelvic_incidence")

plt.ylabel("pelvic_radius")

plt.show()
# data variables and normalization

data["class"] = [1 if each == "Abnormal" else 0 for each in data["class"]]

y = data["class"].values

x_temp = data.drop(["class"], axis=1)

x = (x_temp- np.min(x_temp)) / (np.max(x_temp)-np.min(x_temp)).values
#%%

# train test split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state=1)
# %%

# knn model

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 13) # n_neighbors = k

knn.fit(x_train,y_train)

prediction = knn.predict(x_test)

print(" {} nn score: {} ".format(13,knn.score(x_test,y_test)))
#k value test

score_list = []

liste = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

for each in range(1,15):

    neigh_test = KNeighborsClassifier(n_neighbors=each)

    neigh_test.fit(x_train,y_train)

    score_list.append(neigh_test.score(x_test,y_test))

plt.plot(range(1,15), score_list)

plt.xlabel("range")

plt.ylabel("score")

plt.show()

   