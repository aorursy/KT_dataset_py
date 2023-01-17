import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
%matplotlib inline
from sklearn.linear_model import LogisticRegression # to apply the Logistic regression
from sklearn.model_selection import train_test_split # to split the data into two parts
from sklearn import svm 
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV# for tuning parameter
from sklearn.ensemble import RandomForestClassifier # for random forest classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics 

import os
print(os.listdir("../input"))
data = pd.read_csv("../input/data.csv",header=0) 
data.info()
data.columns 
#get rid of some columnes 
if "id" in data.columns:
    data.drop("id",axis=1,inplace=True)
if "Unnamed: 32" in data.columns: 
    data.drop("Unnamed: 32",axis=1,inplace=True)
    
categorical = False 
#map diagnosis data into categorical integer values
if categorical == False:
    data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})
    categorical = True

data['diagnosis'].head()
#split train/test data
train, test = train_test_split(data, test_size = 0.3)
train.info()
x_train = train.iloc[:,1:]
y_train = train.diagnosis

x_test = test.iloc[:,1:]
y_test = test.diagnosis
model = svm.SVC()
model.fit(x_train,y_train)
prediction=model.predict(x_test)
metrics.accuracy_score(prediction,y_test)
features_one = ['texture_mean','perimeter_mean','smoothness_mean','compactness_mean','symmetry_mean']
x_train_1 = train[features_one]
x_test_1 = test[features_one]
model_1 = svm.SVC()
model_1.fit(x_train_1,y_train)
prediction=model_1.predict(x_test_1)
metrics.accuracy_score(prediction,y_test)
model_3=RandomForestClassifier(n_estimators=100)
model_3.fit(x_train_1,y_train)
prediction = model_3.predict(x_test_1)
metrics.accuracy_score(prediction,y_test)