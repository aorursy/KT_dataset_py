# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import Dense
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
DF = pd.read_csv("../input/weatherAUS.csv", index_col = 0)
#print (DF['Location'].unique());
DF.head()
DF.drop(["Evaporation","Sunshine"],axis = 1,inplace=True)
DF.replace({"No":0,"Yes":1},inplace = True)
print (DF['Cloud9am'].unique())
DF.fillna({"Cloud9am":0,"Cloud3pm":0,"RainToday":0,"Rainfall":0},inplace = True);
DF.dropna(inplace=True);
print(DF.info())
corrMatrix = DF.corr();
corrValues = corrMatrix['RainToday'].sort_values();
print (corrValues)
thres = 0.2
drops = corrValues[np.abs(corrValues<thres)].index.values;
drops
drops = np.concatenate((drops ,["RainTomorrow","Location","RainToday","Rainfall","RISK_MM"]),axis = 0)
x = DF.drop(drops,axis = 1).copy();
y = DF['RainTomorrow'].copy();
change = ["WindGustDir","WindDir9am","WindDir3pm"];
categoricalX = x[change];
oneEncode = pd.get_dummies(categoricalX,drop_first = True);
x.drop(change,axis = 1,inplace = True);
x = pd.concat([x,oneEncode],axis = 1);
x.shape

trainX,testX,trainY,testY = train_test_split(x,y,test_size=0.25);
trainX.shape
paramGrid = {"max_depth":[1,2,3,4,5,6,7,8,9,10,None]};
clf = DecisionTreeClassifier();
gridClf = GridSearchCV(estimator=clf, param_grid=paramGrid,cv=10,n_jobs = 4);
gridClf.fit(trainX,trainY);
yPredClf = gridClf.predict(testX);
print("Decission Tree Classifier Accuracy Score:{0:.3f}".format(accuracy_score(testY,yPredClf)))
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score

# Confusion matrix
print(confusion_matrix(testY,yPredClf))
print(precision_score(testY, yPredClf))
print(recall_score(testY,yPredClf))
print(f1_score(testY,yPredClf))
print(trainX.shape);
print(trainY.shape);
model = Sequential();
model.add(Dense(12,input_shape=(48,),activation='relu'));
model.add(Dense(8,activation= 'relu'));
model.add(Dense(1,activation='sigmoid'));
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy']);
model.fit(trainX, trainY, epochs = 20, batch_size = 1, verbose=1)
yPred = model.predict(testX)
score = model.evaluate(testX,testY,verbose=1)
print(score)
