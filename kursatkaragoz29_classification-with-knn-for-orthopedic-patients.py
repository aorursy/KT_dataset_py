

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv("/kaggle/input/biomechanical-features-of-orthopedic-patients/column_3C_weka.csv")

data.info()
data.head()
data['class'].value_counts()

# Class label1 = Spondylolisthesis (Ortopedi bir hastalık türü)

# Class label2 = Normal

# Class label3 = Hernia (Fıtık hastalığı)
spl = data[data['class'] == 'Spondylolisthesis']

normal = data[data['class'] == 'Normal']

hernia = data[data['class'] == 'Hernia']
# Scatter plot

plt.scatter(spl.lumbar_lordosis_angle, spl.sacral_slope, label="Spondylolisthesis",color="red",alpha=0.5)

plt.scatter(normal.lumbar_lordosis_angle, normal.sacral_slope, label="Normal",color="Green",alpha=0.5)

plt.scatter(hernia.lumbar_lordosis_angle, hernia.sacral_slope, label="Hernia",color="Blue",alpha=0.2)

plt.xlabel("lumbar_lordosis_angle")

plt.ylabel("sacral_slope")

plt.legend()

plt.show()
# Normal = 0   , Spondylolisthesis = 1, Hernia=2

data['class'] = [0 if each == "Normal" else 1 if each == "Spondylolisthesis" else 2 for each in data['class']]



y = data['class'].values

x_data = data.drop(['class'],axis=1)

x_data.head()
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))

x.head()
# train test split

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4,random_state=1)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3) # n_neighbors = k

knn.fit(x_train,y_train)

prediction = knn.predict(x_test)
prediction
print("{} nn score: {} ".format(3,knn.score(x_test,y_test))) #k = hyper parameter
score_list = []

for each in range(1,50):

    knn2 = KNeighborsClassifier(n_neighbors = each) # n_neighbors = k

    knn2.fit(x_train,y_train)

    knn2.score(x_test,y_test)

    score_list.append(knn2.score(x_test,y_test))

plt.plot(range(1,50),score_list)

plt.xlabel("K Values")

plt.ylabel("Accuracy( K Values)")

plt.show()
max_accuracy_value= max(score_list) # what is max score value ?

max_accuracy_k = score_list.index(max_accuracy_value)+1   # what is hyper parameter(k) value for max score ?

print("for max score ==> k={},score={}".format(max_accuracy_k,max_accuracy_value))