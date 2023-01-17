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
data.head()
A = data[data["class"] == "Abnormal"]

N = data[data["class"] == "Normal"]
# scatter plot

plt.figure(figsize = (10,8))

plt.scatter(A.pelvic_incidence,A.sacral_slope,color="red",label = "Abnormal",alpha=0.3)

plt.scatter(N.pelvic_incidence,N.sacral_slope,color="green",label = "Normal",alpha=0.3)

plt.xlabel("pelvic_incidence")

plt.ylabel("sacral_slope")

plt.legend()

plt.show()
class_ = [1 if i=='Abnormal' else 0 for i in data.loc[:,'class']]
y = data["class"].values

x_data = data.drop(["class"],axis=1) 
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state = 1)

#knn model

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 5) # n_neigbotrs = choosen k number

knn.fit(x_train,y_train)

prediction = knn.predict(x_test)
print("{} nn score: {}".format(5,knn.score(x_test,y_test))) # %95 oranında
# find k value

train_accuracy = []

test_accuracy = []

for each in range(1,30):

    knn2 = KNeighborsClassifier(n_neighbors= each)

    knn2.fit(x_train,y_train)

    #train accuracy

    train_accuracy.append(knn2.score(x_train, y_train))

    # test accuracy

    test_accuracy.append(knn2.score(x_test, y_test))



plt.figure(figsize = (10,8))

plt.plot(range(1,30),train_accuracy,color = "blue", label = "Train Accuracy")

plt.plot(range(1,30), test_accuracy,color = "red", label = "Test Accuracy")

plt.xlabel("K Values (number of neighbors)")

plt.ylabel("Accuracy")

plt.legend()

plt.show()

print("Best accuracy is {} with K = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))