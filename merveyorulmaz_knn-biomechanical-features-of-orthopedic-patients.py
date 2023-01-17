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

data.head()
Abnormal = data[data["class"]=="Abnormal"]

Normal = data[data["class"]=="Normal"]
Abnormal.shape
Normal.shape
#Veri görselleştirme



plt.scatter(Abnormal.pelvic_incidence, Abnormal.pelvic_radius,color="red", label="AN")

plt.scatter(Normal.pelvic_incidence, Normal.pelvic_radius, color="green", label="N")

plt.legend()

plt.xlabel("Anormal")

plt.ylabel("Normal")

plt.show()
data["class"] = [1 if each=="Abnormal" else 0 for each in data["class"]]

data.head()
y = data["class"].values

y[205:220]
x_data = data.drop(["class"], axis=1)

x_data.head(3)
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))

x.head()
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
print("{} nn score: {}".format(3, knn.score(x_test,y_test)))
score_list=[]



for each in range(1,15):

    knn2 = KNeighborsClassifier(n_neighbors=each)

    knn2.fit(x_train,y_train)

    score_list.append(knn2.score(x_test,y_test))



plt.plot(range(1,15), score_list, color="green")

plt.xlabel("k values")

plt.ylabel("accurarcy")

plt.show()