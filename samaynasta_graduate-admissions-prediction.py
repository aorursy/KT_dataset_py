import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split



dataset = pd.read_csv("../input/graduate-admissions/Admission_Predict.csv")

dataset.set_index("Serial No.",inplace = True)

X=dataset.iloc[:,2:-1].values

Y=dataset.iloc[:,-1].values



X_train ,X_test , Y_train , Y_test = train_test_split(X,Y,test_size=0.25,random_state=0)



from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train=sc.fit_transform(X_train)

X_test=sc.fit_transform(X_test)

#print(X_train)



from sklearn.ensemble import RandomForestRegressor

regressor =  RandomForestRegressor(n_estimators=100, criterion='mse')

regressor.fit(X_train,Y_train)



Y_pred=regressor.predict(X_test)

#print(Y_pred)



from sklearn.metrics import mean_squared_error

MSE= mean_squared_error(Y_test,Y_pred,multioutput='uniform_average', squared=True)

import math

RMSE = math.sqrt(MSE)

print(RMSE)

#print(Y_pred.shape)

#print(Y_test.shape)

#print(Y_pred)

#print(Y_test)

#print(np.concatenate((Y_test.reshape(len(Y_test),1),Y_pred.reshape(len(Y_pred),1),1)))



#print(dataset.corr())






