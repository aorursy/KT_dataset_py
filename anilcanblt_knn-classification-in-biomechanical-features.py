# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/column_2C_weka.csv")
data.head()
data.describe()
data.info()
#Division of normal and abnormal parts to use them in plotting seperately
Abnormal = data[data["class"] == "Abnormal"]
Normal = data[data["class"] == "Normal"]
#Size of each seperated subdata sets.
sns.countplot(x="class", data=data)
data.loc[:,'class'].value_counts()
#correlation map
#just to find which features I should use.
f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(data.corr(), annot=True, linewidths=.8, fmt= '.1f',ax=ax)
plt.show()
#here is the visualization part. 
f,ax = plt.subplots(figsize=(12, 8))
plt.scatter(Abnormal.pelvic_incidence, Abnormal.pelvic_radius, color="red", alpha=0.5, label="Abnormal")
plt.scatter(Normal.pelvic_incidence, Normal.pelvic_radius, color="green", alpha=0.8, label="Normal")
plt.xlabel("pelvic_incidence")
plt.ylabel("pelvic_radius")
plt.legend()
plt.show()
#To make things easy, I've converted "object" types into "integers". 0 and 1
data["class"] = [1 if i == "Abnormal" else 0 for i in data["class"]]
#Determination of our label and input data.
y = data['class'].values
x_data = data.drop(['class'], axis=1)
#Normalization of the data. Normally I would use it but as I've seen that it reduces my test accuracy, I won't use it. 
#But use should run it to see what I'm trying to explain. 
#x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))
#Simply dividing data into some percentages, like 30% for test and 70% for train
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y, test_size=0.3, random_state=1)
#Here is the KNN part. 
from sklearn.neighbors import KNeighborsClassifier
neighbors = 13
knn = KNeighborsClassifier(n_neighbors=neighbors)
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)
print("score for {} neighbors is: {}".format(neighbors, knn.score(x_test,y_test)))
#To find the best neighbour value for our KNN, we use for loop and simply try each neighbour values.
train = []
test = []
for i in range(1,15):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(x_train,y_train)
    test.append(knn.score(x_test,y_test))
    train.append(knn.score(x_train,y_train))
#After obtaining the result, we are just gonna plot them to see the result clearly.
f,ax = plt.subplots(figsize=(12, 8))
plt.plot(range(1,15),train, color="red", label="train")
plt.plot(range(1,15),test, color="green", label="test")
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.legend()
plt.show()