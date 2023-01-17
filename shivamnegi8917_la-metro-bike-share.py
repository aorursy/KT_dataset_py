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
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
train = pd.read_csv("../input/metro-bike-share-trip-data.csv")
print ('There are',len(train.columns),'columns:')
for x in train.columns:
    print(x+' ',end=',')
train.head()
train.tail()
print("Number of columns (features) in the given dataset is :",train.shape[1])
print("Number of rows (entries) in the given dataset is :",train.shape[0])
train.info()
train_na = (train.isnull().sum()*100)/len(train)
print("Percentage of Missing Data in each feature:")
train_na.sort_values(ascending=False)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
train = train.dropna()
fig, ax = plt.subplots()
ax.scatter(train['Passholder Type'], train['Duration'])
plt.ylabel('Duration (in seconds)', fontsize=13)
plt.xlabel('PassHolder Type', fontsize=13)
plt.show()
l = []
import math 
degrees_to_radians = math.pi/180.0
def distance_on_unit_sphere(lat1, long1, lat2, long2):
    phi1 = (90.0 - lat1)*degrees_to_radians
    phi2 = (90.0 - lat2)*degrees_to_radians
    
    theta1 = long1*degrees_to_radians
    theta2 = long2*degrees_to_radians
    
    a = ((math.sin(phi1)*math.sin(phi2)*math.cos(theta1 - theta2)) +(math.cos(phi1)*math.cos(phi2)))
    if a>1:
        a=0.999999
    dis = math.acos( a )
    return dis*6373
for i in range(97825):
    l.append(distance_on_unit_sphere(train['Starting Station Latitude'].iloc[i],
                                     train['Starting Station Longitude'].iloc[i],
                                     train['Ending Station Latitude'].iloc[i],
                                     train['Ending Station Longitude'].iloc[i]))
temp = pd.DataFrame(data=[train['Duration'],
                               train['Starting Station Latitude'],
                               train['Starting Station Longitude'],
                               train['Ending Station Latitude'],
                               train['Ending Station Longitude'],
                               train['Plan Duration']],
                               index=['Duration',
                                      'Starting Station Latitude',
                                      'Starting Station Longitude',
                                      'Ending Station Latitude',
                                      'Ending Station Longitude',
                                      'Plan Duration'])
distance = pd.DataFrame({'Distance':l})
new_train = temp.T
print("Shape of new train ",new_train.shape)
print ("Shape of distance ",distance.shape)
new_train = new_train.reset_index(drop=True)
new_train = pd.concat([distance,
                       new_train,
                       pd.get_dummies(data=train['Passholder Type']).reset_index(),
                       pd.get_dummies(data=train['Trip Route Category'],drop_first=True).reset_index()],
                       axis=1)
new_train = new_train.drop('index',axis=1)
new_train.info()
print("There are 3 different types of Passholder : ")
train['Passholder Type'].value_counts()
X1 = new_train.drop('Walk-up',axis=1)
y1 = new_train['Walk-up']
X_train,X_test,y_train,y_test = train_test_split(X1,y1,test_size=0.33)
lr = LogisticRegression()
lr.fit(X_train,y_train)
pred1 = lr.predict(X_test)
print(classification_report(y_test,pred1))
print(confusion_matrix(y_test,pred1))
X2 = new_train.drop('Monthly Pass',axis=1)
y2 = new_train['Monthly Pass']
X_train,X_test,y_train,y_test = train_test_split(X2,y2,test_size=0.33)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train,y_train)
pred2 = clf.predict(X_test)
# pred2 = clf2.predict(X_test)
print(classification_report(y_test,pred2))
print(confusion_matrix(y_test,pred2))
X3 = new_train.drop('Flex Pass',axis=1)
y3 = new_train['Flex Pass']
X_train,X_test,y_train,y_test = train_test_split(X3,y3,test_size=0.33)
from sklearn.tree import DecisionTreeClassifier
clf2 = DecisionTreeClassifier()
clf2.fit(X_train,y_train)
pred3 = clf2.predict(X_test)
print(classification_report(y_test,pred3))
print(confusion_matrix(y_test,pred3))