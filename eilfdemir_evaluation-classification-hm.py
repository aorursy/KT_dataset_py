# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/wisc_bc_data.csv')
data.head(10)
data.drop(["id"],axis=1,inplace=True)

data.head(10)
M = data[data.diagnosis=='M']
B = data[data.diagnosis=='B']
plt.scatter(M.smoothness_mean,M.compactness_mean,color='purple',label='Malignant',alpha=0.3)
plt.scatter(B.smoothness_mean,B.compactness_mean,color='blue',label='Benign',alpha=0.3)
plt.xlabel('Malignant')
plt.ylabel('Benign')
plt.legend()
plt.show()
data.diagnosis= [1 if each=="M" else 0 for each in data.diagnosis]
y=data.diagnosis.values
x_data = data.drop(["diagnosis"],axis=1)
x = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))  # Normalization
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3) 
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
print("{} nn score: {}".format(3,knn.score(x_test,y_test)))
score_list = []
for each in range(1,15):
    knn2 = KNeighborsClassifier(n_neighbors = each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))
    
plt.plot(range(1,15),score_list)
plt.xlabel("k")
plt.ylabel("accuracy")
plt.show()
from sklearn.svm import SVC
svm = SVC(random_state=1) 
svm.fit(x_train,y_train)

print("primy accuracy of SVM algorithm : ",svm.score(x_test,y_test))
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)

#test
print("Accuracy of Naive-Bayes Algorithm",nb.score(x_test,y_test))
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.15,random_state=42)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
# Accuracy
print("Accuracy of Decision Tree Algorithm",dt.score(x_test,y_test))
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100,random_state=1)
rf.fit(x_train,y_train)
print("Accuracy of Random Forest Algorithm",rf.score(x_test,y_test))
accuracy_list=[]
for i in range(1,11,1):
    rf = RandomForestClassifier(n_estimators=i,random_state=1) 
    rf.fit(x_train,y_train)
    accuracy_list.append(rf.score(x_test,y_test))
plt.plot(range(1,11),accuracy_list)
plt.xlabel("Number of estimators")
plt.ylabel("Accuracy")
plt.show()