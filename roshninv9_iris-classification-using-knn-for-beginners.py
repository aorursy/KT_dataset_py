#getting started...

from sklearn.datasets import load_iris
iris=load_iris()
x=iris.data

x
y=iris.target
y
feature_names = iris.feature_names

feature_names
target_names = iris.target_names
target_names
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test =train_test_split(x,y,test_size=0.4)
print(X_train.shape)
print(X_test.shape)
from sklearn.neighbors import KNeighborsClassifier
knn =KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,Y_train)
y_pred= knn.predict(X_test)
from sklearn import metrics
print(metrics.accuracy_score(Y_test,y_pred))
#import required packages
from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from math import sqrt
import matplotlib.pyplot as plt
%matplotlib inline
rmse_val = [] #to store rmse values for different k
for K in range(20):
    K = K+1
    model = neighbors.KNeighborsRegressor(n_neighbors = K)

    model.fit(X_train, Y_train)  #fit the model
    
    pred=model.predict(X_test) #make prediction on test set
    
    error = sqrt(mean_squared_error(Y_test,pred)) #calculate rmse
   
    rmse_val.append(error) #store rmse values
   
    print('RMSE value for k= ' , K , 'is:', error)
#plotting the rmse vals

import pandas as pd
curve=pd.DataFrame(rmse_val)#elbow curve
curve.plot()
knn =KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train,Y_train)
y_pred= knn.predict(X_test)

from sklearn import metrics
print(metrics.accuracy_score(Y_test,y_pred))
#see,our score has improved well.....
#thats it tada....
