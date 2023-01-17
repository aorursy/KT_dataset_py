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
data=pd.read_csv("/kaggle/input/biomechanical-features-of-orthopedic-patients/column_3C_weka.csv")
data
from sklearn.neighbors import KNeighborsClassifier
data["class"]=[0 if each=="Normal" else 1 for each in data["class"]]
y=data["class"].values
x_data=data.drop(["class"], axis=1)

x=x_data.values

x=(x-np.min(x))/(np.max(x)-np.min(x))
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.2, random_state=42)
score_list=[]

for each in range(1,30):

    knn2=KNeighborsClassifier(n_neighbors=each)

    knn2.fit(x_train,y_train)

    score_list.append(100*knn2.score(x_test,y_test))

    print("k=", each, "Prediction Accuracy:", 100*knn2.score(x_test,y_test))



plt.plot([*range(1,30)], score_list)

plt.xlabel("k Value")

plt.ylabel("Accuracy %")

plt.show()
optimal_k_value=score_list.index(max(score_list))+1
knn=KNeighborsClassifier(n_neighbors=optimal_k_value)

knn.fit(x_train,y_train)

print("KNN Prediction Accuracy: {}%".format(100*knn.score(x_test,y_test)))