# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Load dataset 
mushrooms=pd.read_csv("../input/mushrooms.csv")
#encode labels 
from sklearn import preprocessing
a=preprocessing.LabelEncoder()
for columns in mushrooms.columns:
    mushrooms[columns]=a.fit_transform(mushrooms[columns])
#split features and target
y=mushrooms['class']
x=mushrooms.drop(['class'],axis=1)
#split data for test and train
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=16)
#apply knn 
from sklearn.neighbors import KNeighborsClassifier as KNN
#choose the best k 
neighbors=np.arange(1,9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
for i,k in enumerate(neighbors):
    knn=KNN(n_neighbors=k)
    knn.fit(x_train,y_train)
    train_accuracy[i]=knn.score(x_train,y_train)
    test_accuracy[i]=knn.score(x_test,y_test)
#generate plot 
import matplotlib.pyplot as plt
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()
#k=2 is gives the best accuracy score. 
knn=KNN(n_neighbors=2)
knn.fit(x_train,y_train)
print(knn.score(x_test,y_test))
#initiate random forest model to extract importance features according to it
from sklearn.ensemble import RandomForestRegressor as RFR
rf=RFR()
rf.fit(x_train,y_train)

#Visualize Dataset features importance
importances=pd.Series(data=rf.feature_importances_,index=x_train.columns)
importances_sorted=importances.sort_values()
importances_sorted.plot(kind='barh',color='red')
plt.title('Features Importances')
plt.show()
