# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
dataset = pd.read_csv("../input/column_2C_weka.csv")
dataset.head()

#summary info about data

A = dataset[dataset["class"] == "Abnormal"]

N = dataset[dataset["class"] == "Normal"]

#data visualization

plt.scatter(A.pelvic_incidence, A.pelvic_radius, color="red", label = "Abnormal")

plt.scatter(N.pelvic_incidence, N.pelvic_radius, color="blue", label = "Normal")

plt.legend()

plt.xlabel("pelvic_incidence")

plt.ylabel("pelvic_radius")

plt.show()
# data variables and normalization

dataset["class"] = [1 if each == "Abnormal" else 0 for each in dataset["class"]]

y = dataset["class"].values

x_temp = dataset.drop(["class"], axis=1)

x = (x_temp- np.min(x_temp)) / (np.max(x_temp)-np.min(x_temp)).values
# train test split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.3, random_state=42)
# find k value

# This for I want using that one from the interactive visualization tools plotly lib.

score_list = []

liste = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

for each in range(1,15):

    neigh_test = KNeighborsClassifier(n_neighbors=each)

    neigh_test.fit(x_train,y_train)

    score_list.append(neigh_test.score(x_test,y_test))



liste = np.linspace(1,15,15)

iplot([go.Scatter(x=liste, y=score_list)])

   

# KNN Model

neigh = KNeighborsClassifier(n_neighbors=3) #n_neighbors=k degeri

neigh.fit(x_train, y_train)

prediction = neigh.predict(x_test)
print("{} knn score accuracy: {} ".format(3,neigh.score(x_test,y_test)))