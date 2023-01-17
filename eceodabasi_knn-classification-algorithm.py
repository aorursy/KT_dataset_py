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
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np



data = pd.read_csv('../input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv')
data.info()
data.tail()

data
Normal = data[data['class'] == "Normal"]

Abnormal = data[data['class'] == "Abnormal"]



plt.scatter(Normal.pelvic_incidence,Normal.degree_spondylolisthesis, color= "m", label="Normal", alpha=0.4)

plt.scatter(Abnormal.pelvic_incidence,Abnormal.degree_spondylolisthesis, color= "c", label="Abnormal", alpha=0.4)

plt.xlabel("pelvic_incidence")

plt.ylabel("degree_spondylolisthesis")

plt.legend()

plt.show()
data['class'] = [1 if each =="Normal" else 0 for each in data['class'] ]

y = data['class'].values

x_data = data.drop(['class'],axis=1)
# normalization

x = ((x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data)))

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2 , random_state = 42)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(x_train,y_train)

prediction = knn.predict(x_test)

print("{} knn score: {}" .format(3, knn.score(x_test,y_test)))

# finding k value for the best prediction

score_list=[]



for i in range(1,10):

    knn2 = KNeighborsClassifier(n_neighbors = i)

    knn2.fit(x_train,y_train)

    score_list.append(knn2.score(x_test,y_test))



plt.plot(range(1,10),score_list)

plt.xlabel("k values")

plt.ylabel("accuracy")

plt.legend()

plt.show()



print("Best accuracy is {} with K = {}".format(np.max(score_list),1+score_list.index(np.max(score_list))))

best_k_accuracy = np.max(score_list)

best_k = 1 + score_list.index(np.max(score_list))


