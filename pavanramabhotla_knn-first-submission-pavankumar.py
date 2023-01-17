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
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
import matplotlib.pylab as plt
#Convert the categorical variables in dummy variables
data_dummies = pd.get_dummies(data)
data_dummies.head(5)
data_dummies.set_index('uid')
features = data_dummies.iloc[:,0:108]
X=features.values
Y=data_dummies1['target']
#Scaling of Data
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(X)
#Splitting into Test and Train
rescaledX_train, rescaledX_test, Y_train, Y_test=train_test_split(rescaledX,Y,test_size=0.2,random_state=0)
#Running the code to see which value of n produces greatest accuracy
training_accuracy = []
test_accuracy = []
neighbor_settings = range(1,5)
for n_neighbors in neighbor_settings:
#build the model
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(rescaledX_train,Y_train)
    training_accuracy.append(clf.score(rescaledX_train, Y_train))
    test_accuracy.append(clf.score(rescaledX_test,Y_test))

#Plotting the data
plt.plot(neighbor_settings,training_accuracy,label="Training Accuracy")
plt.plot(neighbor_settings,test_accuracy,label="Test Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()

#Final Model with N=3
clf_final = KNeighborsClassifier(n_neighbors=3)
clf_final.fit(rescaledX_train,Y_train)
clf_final.score(rescaledX_test,Y_test)
#Reading data from test file to make predictions
pred = pd.read_csv('D:\Open Data Science Course\Lecture 3\Competition\comp.csv')
pred.set_index('uid')
pred_dummies = pd.get_dummies(pred)
pred_dummies.head(5)
#Creating a features array from the new values of X
features_pred = pred_dummies.iloc[:,0:108]
X_new=features_pred.values
rescaledX_new = scaler.fit_transform(X_new)
#Making Prediction and generating a CSV
Y_new=clf_final.predict(rescaledX_new)
pred['target'] = Y_new
pred.to_csv('submission.csv')