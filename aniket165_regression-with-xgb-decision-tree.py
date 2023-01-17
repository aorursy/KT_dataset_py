import warnings
warnings.filterwarnings("ignore")
%matplotlib inline
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd
import numpy as np
import matplotlib as mpl
import seaborn as sns
import csv
import numpy as np
import operator
import random
import datetime
import statsmodels.api as sm
import statsmodels.formula.api as smf
import sklearn.discriminant_analysis
import sklearn.linear_model as skl_lm




from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, classification_report, precision_score
from sklearn import preprocessing
from sklearn import neighbors
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from datetime import timedelta
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import hamming_loss, accuracy_score 
from pandas import DataFrame
from datetime import datetime
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
PATH ='/kaggle/input/covid19-global-forecasting-week-4'
datatrain = pd.read_csv(f'{PATH}/train.csv')
datatest = pd.read_csv(f'{PATH}/test.csv')
date = pd.to_datetime(datatrain["Date"])
datet = pd.to_datetime(datatest["Date"])
print (date)

ldate = int(len(date))
ldatet = int(len(datet))
print("Length of training- date is", ldate)
print("Length of test- date is", ldatet)
m = []
d = []
for i in range(0,ldate):
    dx = (date[i].strftime("%d"))
    mx = (date[i].strftime("%m"))
    m.append(int(mx))
    d.append(int(dx))

mt = []
dt = []
for i in range(0,ldatet):
    dtx = (datet[i].strftime("%d"))
    mtx = (datet[i].strftime("%m"))
    mt.append(int(mtx))
    dt.append(int(dtx))
train = datatrain
test = datatest
train.insert(6,"Month",m,False)
train.insert(7,"Day",d,False)
test.insert(4,"Month",mt,False)
test.insert(5,"Day",dt,False)
train.head()
test.head()
print("Datatrain")
traindays = datatrain['Date'].nunique()
print("Number of Country_Region: ", datatrain['Country_Region'].nunique())
print("Number of Province_State: ", datatrain['Province_State'].nunique())
print("Number of Days: ", traindays)

notrain = datatrain['Id'].nunique()
print("Number of datapoints in train:", notrain)
lotrain = int(notrain/traindays)
print("L Trains:", lotrain)
print("Datatest")
testdays = datatest['Date'].nunique()
print("Number of Days: ", testdays)
notest = datatest['ForecastId'].nunique()
print("Number of datapoints in test:", notest)
lotest = int(notest/testdays)
print("L Test:", lotest)

zt = datet[0]
daycount = []
for i in range(0,lotrain):
    for j in range(1,traindays+1):
        daycount.append(j)
for i in range(traindays):
    if(zt == date[i]):
        zx = i
        print(zx)
        
daytest = []
for i in range(0,lotest):
    for j in range(1,testdays+1):
        jr = zx + j
        daytest.append(jr)
train.insert(8,"DayCount",daycount,False)
test.insert(6,"DayCount",daytest,False)
traincount = int(len(train["Date"]))

testcount = int(len(test["Date"]))
train.Province_State = train.Province_State.fillna(0)
empty = 0
for i in range(0,traincount):
    if(train.Province_State[i] == empty):
        train.Province_State[i] = train.Country_Region[i]
test.Province_State = test.Province_State.fillna(0)
empty = 0
for i in range(0,testcount):
    if(test.Province_State[i] == empty):
        test.Province_State[i] = test.Country_Region[i]
label = preprocessing.LabelEncoder()
train.Country_Region = label.fit_transform(train.Country_Region)
train.Province_State = label.fit_transform(train.Province_State)
test.Country_Region = label.fit_transform(test.Country_Region)
test.Province_State = label.fit_transform(test.Province_State)

X = np.c_[train["Province_State"], train["Country_Region"], train["DayCount"], train["Month"], train["Day"]]
Xt = np.c_[test["Province_State"], test["Country_Region"], test["DayCount"], test["Month"], test["Day"]]
display(X.shape)
display(Xt.shape)
Y1 = train["ConfirmedCases"]
Y2 = train["Fatalities"]
display(Y1.shape)
display(Y2.shape)
Y1.head(100)
#Y1.shape
Y2.head(100)
def rmsle(y_true, y_pred):
    return mean_squared_log_error(y_true, y_pred)**(1/2);
Regressor_Concases = XGBRegressor(n_estimators = 1000, gamma = 0, learning_rate = 0.05,  
                                  random_state = 42 , max_depth = 20) 

Regressor_fatalities = XGBRegressor(n_estimators = 1000 , gamma = 0, learning_rate = 0.05,  
                                    random_state = 42 , max_depth = 20)
import math
from sklearn.metrics import mean_squared_log_error

Regressor_Concases.fit(X,Y1.ravel())
A1 = Regressor_Concases.predict(X)
#B = mean_squared_error(A,Y1)
#print("Training - Mean Squared Error is: ",B) 

A1=abs(A1)
Y1=abs(Y1)




B1 = mean_squared_error(A1,Y1)
B2 = mean_squared_log_error(A1,Y1)
print("-------Confirmed Case---------s")
print("Training - Mean Squared Error is: ",B1)
print("Training - Mean Squared LOG Error is: ",B2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A1,Y1))
display(A1.shape)
#print("Training - ROOT Mean Squared Error is: ",math.sqrt(B1))
#print(rmsle(A1,Y1))
ypred = Regressor_Concases.predict(Xt)
ypred = pd.DataFrame({'ConfirmedCases' : ypred}) 
ypred = round(ypred)
ypred.head(20) 
ypred.shape
Regressor_fatalities.fit(X,Y2.ravel())
ypred2= Regressor_fatalities.predict(Xt)
A2 = Regressor_fatalities.predict(X)
A2 = np.round(A2)
A2=abs(A2)
Y2=abs(Y2)


C1 = mean_squared_error(A2,Y2)
C2 = mean_squared_log_error(A2,Y2)
print("--------Fatalities-------")
print("Training - Mean Squared Error is: ",C1)
print("Training - Mean Squared LOG Error is: ",C2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A2,Y2))
display(A2.shape)
ypred2 = pd.DataFrame({'Fatalities' : ypred2}) 
ypred2 = round(ypred2)
ypred2.head(20) 
final_pred = pd.DataFrame()
forecast = test["ForecastId"]
final_pred.insert(0,"ForecastId",forecast,False)
final_pred.insert(1,"ConfirmedCases",ypred,False)
final_pred.insert(2, "Fatalities",ypred2,False)
final_pred
print(final_pred)
Regressor_Concases = XGBRegressor(n_estimators = 1000, gamma = 0, learning_rate = 0.01,  
                                  random_state = 42 , max_depth = 20) 
Regressor_fatalities = XGBRegressor(n_estimators = 1000 , gamma = 0, learning_rate = 0.01,  
                                    random_state = 42 , max_depth = 20)



import math
from sklearn.metrics import mean_squared_log_error
Regressor_Concases.fit(X,Y1.ravel())
A1 = Regressor_Concases.predict(X)
#B = mean_squared_error(A,Y1)
#print("Training - Mean Squared Error is: ",B) 

A1=abs(A1)
Y1=abs(Y1)


B1 = mean_squared_error(A1,Y1)
B2 = mean_squared_log_error(A1,Y1)
print("-------Confirmed Case---------s")
print("Training - Mean Squared Error is: ",B1)
print("Training - Mean Squared LOG Error is: ",B2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A1,Y1))
display(A1.shape)
#print("Training - ROOT Mean Squared Error is: ",math.sqrt(B1))
#print(rmsle(A1,Y1))


ypred = Regressor_Concases.predict(Xt)
ypred = pd.DataFrame({'ConfirmedCases' : ypred}) 
ypred = round(ypred)
ypred.head(20) 

Regressor_fatalities.fit(X,Y2.ravel())
ypred2= Regressor_fatalities.predict(Xt)
A2 = Regressor_fatalities.predict(X)
A2 = np.round(A2)
A2=abs(A2)
Y2=abs(Y2)

C1 = mean_squared_error(A2,Y2)
C2 = mean_squared_log_error(A2,Y2)
print("--------Fatalities-------")
print("Training - Mean Squared Error is: ",C1)
print("Training - Mean Squared LOG Error is: ",C2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A2,Y2))
display(A2.shape)

ypred2 = pd.DataFrame({'Fatalities' : ypred2}) 
ypred2 = round(ypred2)
ypred2.head(20) 
Regressor_Concases = XGBRegressor(n_estimators = 1000, gamma = 0, learning_rate = 0.5,  
                                  random_state = 42 , max_depth = 20) 
Regressor_fatalities = XGBRegressor(n_estimators = 1000 , gamma = 0, learning_rate = 0.5,  
                                    random_state = 42 , max_depth = 20)



import math
from sklearn.metrics import mean_squared_log_error
Regressor_Concases.fit(X,Y1.ravel())
A1 = Regressor_Concases.predict(X)
#B = mean_squared_error(A,Y1)
#print("Training - Mean Squared Error is: ",B) 

A1=abs(A1)
Y1=abs(Y1)


B1 = mean_squared_error(A1,Y1)
B2 = mean_squared_log_error(A1,Y1)
print("-------Confirmed Case---------s")
print("Training - Mean Squared Error is: ",B1)
print("Training - Mean Squared LOG Error is: ",B2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A1,Y1))
display(A1.shape)
#print("Training - ROOT Mean Squared Error is: ",math.sqrt(B1))
#print(rmsle(A1,Y1))


ypred = Regressor_Concases.predict(Xt)
ypred = pd.DataFrame({'ConfirmedCases' : ypred}) 
ypred = round(ypred)
ypred.head(20) 

Regressor_fatalities.fit(X,Y2.ravel())
ypred2= Regressor_fatalities.predict(Xt)
A2 = Regressor_fatalities.predict(X)
A2 = np.round(A2)
A2=abs(A2)
Y2=abs(Y2)

C1 = mean_squared_error(A2,Y2)
C2 = mean_squared_log_error(A2,Y2)
print("--------Fatalities-------")
print("Training - Mean Squared Error is: ",C1)
print("Training - Mean Squared LOG Error is: ",C2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A2,Y2))
display(A2.shape)

ypred2 = pd.DataFrame({'Fatalities' : ypred2}) 
ypred2 = round(ypred2)
ypred2.head(20) 
Regressor_Concases = XGBRegressor(n_estimators = 1000, gamma = 0, learning_rate = 0.9999,  
                                  random_state = 42 , max_depth = 20) 
Regressor_fatalities = XGBRegressor(n_estimators = 1000 , gamma = 0, learning_rate = 0.9999,  
                                    random_state = 42 , max_depth = 20)



import math
from sklearn.metrics import mean_squared_log_error
Regressor_Concases.fit(X,Y1.ravel())
A1 = Regressor_Concases.predict(X)
#B = mean_squared_error(A,Y1)
#print("Training - Mean Squared Error is: ",B) 

A1=abs(A1)
Y1=abs(Y1)


B1 = mean_squared_error(A1,Y1)
B2 = mean_squared_log_error(A1,Y1)
print("-------Confirmed Case---------s")
print("Training - Mean Squared Error is: ",B1)
print("Training - Mean Squared LOG Error is: ",B2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A1,Y1))
display(A1.shape)
#print("Training - ROOT Mean Squared Error is: ",math.sqrt(B1))
#print(rmsle(A1,Y1))


ypred = Regressor_Concases.predict(Xt)
ypred = pd.DataFrame({'ConfirmedCases' : ypred}) 
ypred = round(ypred)
ypred.head(20) 

Regressor_fatalities.fit(X,Y2.ravel())
ypred2= Regressor_fatalities.predict(Xt)
A2 = Regressor_fatalities.predict(X)
A2 = np.round(A2)
A2=abs(A2)
Y2=abs(Y2)

C1 = mean_squared_error(A2,Y2)
C2 = mean_squared_log_error(A2,Y2)
print("--------Fatalities-------")
print("Training - Mean Squared Error is: ",C1)
print("Training - Mean Squared LOG Error is: ",C2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A2,Y2))
display(A2.shape)

ypred2 = pd.DataFrame({'Fatalities' : ypred2}) 
ypred2 = round(ypred2)
ypred2.head(20) 
Regressor_Concases = XGBRegressor(n_estimators = 1000, gamma = 0, learning_rate = 0.9999,  
                                  random_state = 42 , max_depth = 20,colsample_bytree=0.3) 
Regressor_fatalities = XGBRegressor(n_estimators = 1000 , gamma = 0, learning_rate = 0.9999,  
                                    random_state = 42 , max_depth = 20,colsample_bytree=0.3)



import math
from sklearn.metrics import mean_squared_log_error
Regressor_Concases.fit(X,Y1.ravel())
A1 = Regressor_Concases.predict(X)
#B = mean_squared_error(A,Y1)
#print("Training - Mean Squared Error is: ",B) 

A1=abs(A1)
Y1=abs(Y1)


B1 = mean_squared_error(A1,Y1)
B2 = mean_squared_log_error(A1,Y1)
print("-------Confirmed Case---------s")
print("Training - Mean Squared Error is: ",B1)
print("Training - Mean Squared LOG Error is: ",B2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A1,Y1))
display(A1.shape)
#print("Training - ROOT Mean Squared Error is: ",math.sqrt(B1))
#print(rmsle(A1,Y1))


ypred = Regressor_Concases.predict(Xt)
ypred = pd.DataFrame({'ConfirmedCases' : ypred}) 
ypred = round(ypred)
ypred.head(20) 

Regressor_fatalities.fit(X,Y2.ravel())
ypred2= Regressor_fatalities.predict(Xt)
A2 = Regressor_fatalities.predict(X)
A2 = np.round(A2)
A2=abs(A2)
Y2=abs(Y2)

C1 = mean_squared_error(A2,Y2)
C2 = mean_squared_log_error(A2,Y2)
print("--------Fatalities-------")
print("Training - Mean Squared Error is: ",C1)
print("Training - Mean Squared LOG Error is: ",C2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A2,Y2))
display(A2.shape)

ypred2 = pd.DataFrame({'Fatalities' : ypred2}) 
ypred2 = round(ypred2)
ypred2.head(20) 
Regressor_Concases = XGBRegressor(n_estimators = 1000, gamma = 0, learning_rate = 0.9999,  
                                  random_state = 42 , max_depth = 20,colsample_bytree=0.5) 
Regressor_fatalities = XGBRegressor(n_estimators = 1000 , gamma = 0, learning_rate = 0.9999,  
                                    random_state = 42 , max_depth = 20,colsample_bytree=0.5)



import math
from sklearn.metrics import mean_squared_log_error
Regressor_Concases.fit(X,Y1.ravel())
A1 = Regressor_Concases.predict(X)
#B = mean_squared_error(A,Y1)
#print("Training - Mean Squared Error is: ",B) 

A1=abs(A1)
Y1=abs(Y1)


B1 = mean_squared_error(A1,Y1)
B2 = mean_squared_log_error(A1,Y1)
print("-------Confirmed Case---------s")
print("Training - Mean Squared Error is: ",B1)
print("Training - Mean Squared LOG Error is: ",B2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A1,Y1))
display(A1.shape)
#print("Training - ROOT Mean Squared Error is: ",math.sqrt(B1))
#print(rmsle(A1,Y1))


ypred = Regressor_Concases.predict(Xt)
ypred = pd.DataFrame({'ConfirmedCases' : ypred}) 
ypred = round(ypred)
ypred.head(20) 

Regressor_fatalities.fit(X,Y2.ravel())
ypred2= Regressor_fatalities.predict(Xt)
A2 = Regressor_fatalities.predict(X)
A2 = np.round(A2)
A2=abs(A2)
Y2=abs(Y2)

C1 = mean_squared_error(A2,Y2)
C2 = mean_squared_log_error(A2,Y2)
print("--------Fatalities-------")
print("Training - Mean Squared Error is: ",C1)
print("Training - Mean Squared LOG Error is: ",C2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A2,Y2))
display(A2.shape)

ypred2 = pd.DataFrame({'Fatalities' : ypred2}) 
ypred2 = round(ypred2)
ypred2.head(20) 
Regressor_Concases = XGBRegressor(n_estimators = 1000, gamma = 0, learning_rate = 0.9999,  
                                  random_state = 42 , max_depth = 20,colsample_bytree=0.8) 
Regressor_fatalities = XGBRegressor(n_estimators = 1000 , gamma = 0, learning_rate = 0.9999,  
                                    random_state = 42 , max_depth = 20,colsample_bytree=0.8)



import math
from sklearn.metrics import mean_squared_log_error
Regressor_Concases.fit(X,Y1.ravel())
A1 = Regressor_Concases.predict(X)
#B = mean_squared_error(A,Y1)
#print("Training - Mean Squared Error is: ",B) 

A1=abs(A1)
Y1=abs(Y1)


B1 = mean_squared_error(A1,Y1)
B2 = mean_squared_log_error(A1,Y1)
print("-------Confirmed Case---------s")
print("Training - Mean Squared Error is: ",B1)
print("Training - Mean Squared LOG Error is: ",B2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A1,Y1))
display(A1.shape)
#print("Training - ROOT Mean Squared Error is: ",math.sqrt(B1))
#print(rmsle(A1,Y1))


ypred = Regressor_Concases.predict(Xt)
ypred = pd.DataFrame({'ConfirmedCases' : ypred}) 
ypred = round(ypred)
ypred.head(20) 

Regressor_fatalities.fit(X,Y2.ravel())
ypred2= Regressor_fatalities.predict(Xt)
A2 = Regressor_fatalities.predict(X)
A2 = np.round(A2)
A2=abs(A2)
Y2=abs(Y2)

C1 = mean_squared_error(A2,Y2)
C2 = mean_squared_log_error(A2,Y2)
print("--------Fatalities-------")
print("Training - Mean Squared Error is: ",C1)
print("Training - Mean Squared LOG Error is: ",C2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A2,Y2))
display(A2.shape)

ypred2 = pd.DataFrame({'Fatalities' : ypred2}) 
ypred2 = round(ypred2)
ypred2.head(20) 
Regressor_Concases = XGBRegressor(n_estimators = 1000, gamma = 0, learning_rate = 0.9999,  
                                  random_state = 42 , max_depth = 20,colsample_bytree=0) 
Regressor_fatalities = XGBRegressor(n_estimators = 1000 , gamma = 0, learning_rate = 0.9999,  
                                    random_state = 42 , max_depth = 20,colsample_bytree=0)



import math
from sklearn.metrics import mean_squared_log_error
Regressor_Concases.fit(X,Y1.ravel())
A1 = Regressor_Concases.predict(X)
#B = mean_squared_error(A,Y1)
#print("Training - Mean Squared Error is: ",B) 

A1=abs(A1)
Y1=abs(Y1)


B1 = mean_squared_error(A1,Y1)
B2 = mean_squared_log_error(A1,Y1)
print("-------Confirmed Case---------s")
print("Training - Mean Squared Error is: ",B1)
print("Training - Mean Squared LOG Error is: ",B2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A1,Y1))
display(A1.shape)
#print("Training - ROOT Mean Squared Error is: ",math.sqrt(B1))
#print(rmsle(A1,Y1))


ypred = Regressor_Concases.predict(Xt)
ypred = pd.DataFrame({'ConfirmedCases' : ypred}) 
ypred = round(ypred)
ypred.head(20) 

Regressor_fatalities.fit(X,Y2.ravel())
ypred2= Regressor_fatalities.predict(Xt)
A2 = Regressor_fatalities.predict(X)
A2 = np.round(A2)
A2=abs(A2)
Y2=abs(Y2)

C1 = mean_squared_error(A2,Y2)
C2 = mean_squared_log_error(A2,Y2)
print("--------Fatalities-------")
print("Training - Mean Squared Error is: ",C1)
print("Training - Mean Squared LOG Error is: ",C2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A2,Y2))
display(A2.shape)

ypred2 = pd.DataFrame({'Fatalities' : ypred2}) 
ypred2 = round(ypred2)
ypred2.head(20) 
Regressor_Concases = XGBRegressor(n_estimators = 1000, gamma = 0, learning_rate = 0.9999,  
                                  random_state = 42 , max_depth = 20)#,colsample_bytree=0) 
Regressor_fatalities = XGBRegressor(n_estimators = 1000 , gamma = 0, learning_rate = 0.9999,  
                                    random_state = 42 , max_depth = 20)#,colsample_bytree=0)



import math
from sklearn.metrics import mean_squared_log_error
Regressor_Concases.fit(X,Y1.ravel())
A1 = Regressor_Concases.predict(X)
#B = mean_squared_error(A,Y1)
#print("Training - Mean Squared Error is: ",B) 

A1=abs(A1)
Y1=abs(Y1)


B1 = mean_squared_error(A1,Y1)
B2 = mean_squared_log_error(A1,Y1)
print("-------Confirmed Case---------s")
print("Training - Mean Squared Error is: ",B1)
print("Training - Mean Squared LOG Error is: ",B2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A1,Y1))
display(A1.shape)
#print("Training - ROOT Mean Squared Error is: ",math.sqrt(B1))
#print(rmsle(A1,Y1))


ypred = Regressor_Concases.predict(Xt)
ypred = pd.DataFrame({'ConfirmedCases' : ypred}) 
ypred = round(ypred)
ypred.head(20) 

Regressor_fatalities.fit(X,Y2.ravel())
ypred2= Regressor_fatalities.predict(Xt)
A2 = Regressor_fatalities.predict(X)
A2 = np.round(A2)
A2=abs(A2)
Y2=abs(Y2)

C1 = mean_squared_error(A2,Y2)
C2 = mean_squared_log_error(A2,Y2)
print("--------Fatalities-------")
print("Training - Mean Squared Error is: ",C1)
print("Training - Mean Squared LOG Error is: ",C2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A2,Y2))
display(A2.shape)

ypred2 = pd.DataFrame({'Fatalities' : ypred2}) 
ypred2 = round(ypred2)
ypred2.head(20) 
Regressor_Concases = XGBRegressor(n_estimators = 1000, gamma = 0, learning_rate = 0.9999,  
                                  random_state = 42 , max_depth = 20, max_delta=0)#,colsample_bytree=0) 
Regressor_fatalities = XGBRegressor(n_estimators = 1000 , gamma = 0, learning_rate = 0.9999,  
                                    random_state = 42 , max_depth = 20,max_delta=0)#,colsample_bytree=0)



import math
from sklearn.metrics import mean_squared_log_error
Regressor_Concases.fit(X,Y1.ravel())
A1 = Regressor_Concases.predict(X)
#B = mean_squared_error(A,Y1)
#print("Training - Mean Squared Error is: ",B) 

A1=abs(A1)
Y1=abs(Y1)


B1 = mean_squared_error(A1,Y1)
B2 = mean_squared_log_error(A1,Y1)
print("-------Confirmed Case---------s")
print("Training - Mean Squared Error is: ",B1)
print("Training - Mean Squared LOG Error is: ",B2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A1,Y1))
display(A1.shape)
#print("Training - ROOT Mean Squared Error is: ",math.sqrt(B1))
#print(rmsle(A1,Y1))


ypred = Regressor_Concases.predict(Xt)
ypred = pd.DataFrame({'ConfirmedCases' : ypred}) 
ypred = round(ypred)
ypred.head(20) 

Regressor_fatalities.fit(X,Y2.ravel())
ypred2= Regressor_fatalities.predict(Xt)
A2 = Regressor_fatalities.predict(X)
A2 = np.round(A2)
A2=abs(A2)
Y2=abs(Y2)

C1 = mean_squared_error(A2,Y2)
C2 = mean_squared_log_error(A2,Y2)
print("--------Fatalities-------")
print("Training - Mean Squared Error is: ",C1)
print("Training - Mean Squared LOG Error is: ",C2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A2,Y2))
display(A2.shape)

ypred2 = pd.DataFrame({'Fatalities' : ypred2}) 
ypred2 = round(ypred2)
ypred2.head(20) 
Regressor_Concases = XGBRegressor(n_estimators = 1000, gamma = 0, learning_rate = 0.9999,  
                                  random_state = 42 , max_depth = 20, max_delta=0.5)#,colsample_bytree=0) 
Regressor_fatalities = XGBRegressor(n_estimators = 1000 , gamma = 0, learning_rate = 0.9999,  
                                    random_state = 42 , max_depth = 20,max_delta=0.5)#,colsample_bytree=0)



import math
from sklearn.metrics import mean_squared_log_error
Regressor_Concases.fit(X,Y1.ravel())
A1 = Regressor_Concases.predict(X)
#B = mean_squared_error(A,Y1)
#print("Training - Mean Squared Error is: ",B) 

A1=abs(A1)
Y1=abs(Y1)


B1 = mean_squared_error(A1,Y1)
B2 = mean_squared_log_error(A1,Y1)
print("-------Confirmed Case---------s")
print("Training - Mean Squared Error is: ",B1)
print("Training - Mean Squared LOG Error is: ",B2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A1,Y1))
display(A1.shape)
#print("Training - ROOT Mean Squared Error is: ",math.sqrt(B1))
#print(rmsle(A1,Y1))


ypred = Regressor_Concases.predict(Xt)
ypred = pd.DataFrame({'ConfirmedCases' : ypred}) 
ypred = round(ypred)
ypred.head(20) 

Regressor_fatalities.fit(X,Y2.ravel())
ypred2= Regressor_fatalities.predict(Xt)
A2 = Regressor_fatalities.predict(X)
A2 = np.round(A2)
A2=abs(A2)
Y2=abs(Y2)

C1 = mean_squared_error(A2,Y2)
C2 = mean_squared_log_error(A2,Y2)
print("--------Fatalities-------")
print("Training - Mean Squared Error is: ",C1)
print("Training - Mean Squared LOG Error is: ",C2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A2,Y2))
display(A2.shape)

ypred2 = pd.DataFrame({'Fatalities' : ypred2}) 
ypred2 = round(ypred2)
ypred2.head(20) 
Regressor_Concases = XGBRegressor(n_estimators = 1000, gamma = 0, learning_rate = 0.9999,  
                                  random_state = 42 , max_depth = 20, max_delta=-0.5)#,colsample_bytree=0) 
Regressor_fatalities = XGBRegressor(n_estimators = 1000 , gamma = 0, learning_rate = 0.9999,  
                                    random_state = 42 , max_depth = 20,max_delta=-0.5)#,colsample_bytree=0)



import math
from sklearn.metrics import mean_squared_log_error
Regressor_Concases.fit(X,Y1.ravel())
A1 = Regressor_Concases.predict(X)
#B = mean_squared_error(A,Y1)
#print("Training - Mean Squared Error is: ",B) 

A1=abs(A1)
Y1=abs(Y1)


B1 = mean_squared_error(A1,Y1)
B2 = mean_squared_log_error(A1,Y1)
print("-------Confirmed Case---------s")
print("Training - Mean Squared Error is: ",B1)
print("Training - Mean Squared LOG Error is: ",B2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A1,Y1))
display(A1.shape)
#print("Training - ROOT Mean Squared Error is: ",math.sqrt(B1))
#print(rmsle(A1,Y1))


ypred = Regressor_Concases.predict(Xt)
ypred = pd.DataFrame({'ConfirmedCases' : ypred}) 
ypred = round(ypred)
ypred.head(20) 

Regressor_fatalities.fit(X,Y2.ravel())
ypred2= Regressor_fatalities.predict(Xt)
A2 = Regressor_fatalities.predict(X)
A2 = np.round(A2)
A2=abs(A2)
Y2=abs(Y2)

C1 = mean_squared_error(A2,Y2)
C2 = mean_squared_log_error(A2,Y2)
print("--------Fatalities-------")
print("Training - Mean Squared Error is: ",C1)
print("Training - Mean Squared LOG Error is: ",C2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A2,Y2))
display(A2.shape)

ypred2 = pd.DataFrame({'Fatalities' : ypred2}) 
ypred2 = round(ypred2)
ypred2.head(20) 
Regressor_Concases = XGBRegressor(n_estimators = 1000, gamma = 0, learning_rate = 0.9999,  
                                  random_state = 42 , max_depth = 20, max_delta=-0.5, reg_lambda=0)#,colsample_bytree=0) 
Regressor_fatalities = XGBRegressor(n_estimators = 1000 , gamma = 0, learning_rate = 0.9999,  
                                    random_state = 42 , max_depth = 20,max_delta=-0.5,reg_lambda=0)#,colsample_bytree=0)



import math
from sklearn.metrics import mean_squared_log_error
Regressor_Concases.fit(X,Y1.ravel())
A1 = Regressor_Concases.predict(X)
#B = mean_squared_error(A,Y1)
#print("Training - Mean Squared Error is: ",B) 

A1=abs(A1)
Y1=abs(Y1)


B1 = mean_squared_error(A1,Y1)
B2 = mean_squared_log_error(A1,Y1)
print("-------Confirmed Case---------s")
print("Training - Mean Squared Error is: ",B1)
print("Training - Mean Squared LOG Error is: ",B2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A1,Y1))
display(A1.shape)
#print("Training - ROOT Mean Squared Error is: ",math.sqrt(B1))
#print(rmsle(A1,Y1))


ypred = Regressor_Concases.predict(Xt)
ypred = pd.DataFrame({'ConfirmedCases' : ypred}) 
ypred = round(ypred)
ypred.head(20) 

Regressor_fatalities.fit(X,Y2.ravel())
ypred2= Regressor_fatalities.predict(Xt)
A2 = Regressor_fatalities.predict(X)
A2 = np.round(A2)
A2=abs(A2)
Y2=abs(Y2)

C1 = mean_squared_error(A2,Y2)
C2 = mean_squared_log_error(A2,Y2)
print("--------Fatalities-------")
print("Training - Mean Squared Error is: ",C1)
print("Training - Mean Squared LOG Error is: ",C2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A2,Y2))
display(A2.shape)

ypred2 = pd.DataFrame({'Fatalities' : ypred2}) 
ypred2 = round(ypred2)
ypred2.head(20) 
Regressor_Concases = XGBRegressor(n_estimators = 1000, gamma = 0, learning_rate = 0.9999,  
                                  random_state = 42 , max_depth = 20, max_delta=-0.5, reg_lambda=1)#,colsample_bytree=0) 
Regressor_fatalities = XGBRegressor(n_estimators = 1000 , gamma = 0, learning_rate = 0.9999,  
                                    random_state = 42 , max_depth = 20,max_delta=-0.5,reg_lambda=1)#,colsample_bytree=0)



import math
from sklearn.metrics import mean_squared_log_error
Regressor_Concases.fit(X,Y1.ravel())
A1 = Regressor_Concases.predict(X)
#B = mean_squared_error(A,Y1)
#print("Training - Mean Squared Error is: ",B) 

A1=abs(A1)
Y1=abs(Y1)


B1 = mean_squared_error(A1,Y1)
B2 = mean_squared_log_error(A1,Y1)
print("-------Confirmed Case---------s")
print("Training - Mean Squared Error is: ",B1)
print("Training - Mean Squared LOG Error is: ",B2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A1,Y1))
display(A1.shape)
#print("Training - ROOT Mean Squared Error is: ",math.sqrt(B1))
#print(rmsle(A1,Y1))


ypred = Regressor_Concases.predict(Xt)
ypred = pd.DataFrame({'ConfirmedCases' : ypred}) 
ypred = round(ypred)
ypred.head(20) 

Regressor_fatalities.fit(X,Y2.ravel())
ypred2= Regressor_fatalities.predict(Xt)
A2 = Regressor_fatalities.predict(X)
A2 = np.round(A2)
A2=abs(A2)
Y2=abs(Y2)

C1 = mean_squared_error(A2,Y2)
C2 = mean_squared_log_error(A2,Y2)
print("--------Fatalities-------")
print("Training - Mean Squared Error is: ",C1)
print("Training - Mean Squared LOG Error is: ",C2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A2,Y2))
display(A2.shape)

ypred2 = pd.DataFrame({'Fatalities' : ypred2}) 
ypred2 = round(ypred2)
ypred2.head(20) 
Regressor_Concases = XGBRegressor(n_estimators = 1000, gamma = 0, learning_rate = 0.9999,  
                                  random_state = 42 , max_depth = 20, max_delta=-0.5, reg_lambda=1,reg_alpha=0) 
#,colsample_bytree=0) 
Regressor_fatalities = XGBRegressor(n_estimators = 1000 , gamma = 0, learning_rate = 0.9999,  
                                    random_state = 42 , max_depth = 20,max_delta=-0.5,reg_lambda=1,reg_alpha=0) 
#,colsample_bytree=0)



import math
from sklearn.metrics import mean_squared_log_error
Regressor_Concases.fit(X,Y1.ravel())
A1 = Regressor_Concases.predict(X)
#B = mean_squared_error(A,Y1)
#print("Training - Mean Squared Error is: ",B) 

A1=abs(A1)
Y1=abs(Y1)


B1 = mean_squared_error(A1,Y1)
B2 = mean_squared_log_error(A1,Y1)
print("-------Confirmed Case---------s")
print("Training - Mean Squared Error is: ",B1)
print("Training - Mean Squared LOG Error is: ",B2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A1,Y1))
display(A1.shape)
#print("Training - ROOT Mean Squared Error is: ",math.sqrt(B1))
#print(rmsle(A1,Y1))

ypred = Regressor_Concases.predict(Xt)
ypred = pd.DataFrame({'ConfirmedCases' : ypred}) 
ypred = round(ypred)
ypred.head(20) 

Regressor_fatalities.fit(X,Y2.ravel())
ypred2= Regressor_fatalities.predict(Xt)
A2 = Regressor_fatalities.predict(X)
A2 = np.round(A2)
A2=abs(A2)
Y2=abs(Y2)

C1 = mean_squared_error(A2,Y2)
C2 = mean_squared_log_error(A2,Y2)
print("--------Fatalities-------")
print("Training - Mean Squared Error is: ",C1)
print("Training - Mean Squared LOG Error is: ",C2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A2,Y2))
display(A2.shape)

ypred2 = pd.DataFrame({'Fatalities' : ypred2}) 
ypred2 = round(ypred2)
ypred2.head(20) 

Regressor_Concases = XGBRegressor(n_estimators = 1000, gamma = 0, learning_rate = 0.9999,  
                                  random_state = 42 , max_depth = 20, max_delta=-0.5, reg_lambda=0,reg_alpha=1) 
#,colsample_bytree=0) 
Regressor_fatalities = XGBRegressor(n_estimators = 1000 , gamma = 0, learning_rate = 0.9999,  
                                    random_state = 42 , max_depth = 20,max_delta=-0.5,reg_lambda=0,reg_alpha=1) 
#,colsample_bytree=0)



import math
from sklearn.metrics import mean_squared_log_error
Regressor_Concases.fit(X,Y1.ravel())
A1 = Regressor_Concases.predict(X)
#B = mean_squared_error(A,Y1)
#print("Training - Mean Squared Error is: ",B) 

A1=abs(A1)
Y1=abs(Y1)


B1 = mean_squared_error(A1,Y1)
B2 = mean_squared_log_error(A1,Y1)
print("-------Confirmed Case---------s")
print("Training - Mean Squared Error is: ",B1)
print("Training - Mean Squared LOG Error is: ",B2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A1,Y1))
display(A1.shape)
#print("Training - ROOT Mean Squared Error is: ",math.sqrt(B1))
#print(rmsle(A1,Y1))

ypred = Regressor_Concases.predict(Xt)
ypred = pd.DataFrame({'ConfirmedCases' : ypred}) 
ypred = round(ypred)
ypred.head(20) 

Regressor_fatalities.fit(X,Y2.ravel())
ypred2= Regressor_fatalities.predict(Xt)
A2 = Regressor_fatalities.predict(X)
A2 = np.round(A2)
A2=abs(A2)
Y2=abs(Y2)

C1 = mean_squared_error(A2,Y2)
C2 = mean_squared_log_error(A2,Y2)
print("--------Fatalities-------")
print("Training - Mean Squared Error is: ",C1)
print("Training - Mean Squared LOG Error is: ",C2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A2,Y2))
display(A2.shape)

ypred2 = pd.DataFrame({'Fatalities' : ypred2}) 
ypred2 = round(ypred2)
ypred2.head(20) 

Regressor_Concases = XGBRegressor(n_estimators = 1000, gamma = 0, learning_rate = 0.9999,  
                                  random_state = 42 , max_depth = 20, max_delta=-0.5, reg_lambda=1,reg_alpha=0) 
#,colsample_bytree=0) 
Regressor_fatalities = XGBRegressor(n_estimators = 1000 , gamma = 0, learning_rate = 0.9999,  
                                    random_state = 42 , max_depth = 20,max_delta=-0.5,reg_lambda=1,reg_alpha=0) 
#,colsample_bytree=0)



import math
from sklearn.metrics import mean_squared_log_error
Regressor_Concases.fit(X,Y1.ravel())
A1 = Regressor_Concases.predict(X)
#B = mean_squared_error(A,Y1)
#print("Training - Mean Squared Error is: ",B) 

A1=abs(A1)
Y1=abs(Y1)


B1 = mean_squared_error(A1,Y1)
B2 = mean_squared_log_error(A1,Y1)
print("-------Confirmed Case---------s")
print("Training - Mean Squared Error is: ",B1)
print("Training - Mean Squared LOG Error is: ",B2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A1,Y1))
display(A1.shape)
#print("Training - ROOT Mean Squared Error is: ",math.sqrt(B1))
#print(rmsle(A1,Y1))

ypred = Regressor_Concases.predict(Xt)
ypred = pd.DataFrame({'ConfirmedCases' : ypred}) 
ypred = round(ypred)
ypred.head(20) 

Regressor_fatalities.fit(X,Y2.ravel())
ypred2= Regressor_fatalities.predict(Xt)
A2 = Regressor_fatalities.predict(X)
A2 = np.round(A2)
A2=abs(A2)
Y2=abs(Y2)

C1 = mean_squared_error(A2,Y2)
C2 = mean_squared_log_error(A2,Y2)
print("--------Fatalities-------")
print("Training - Mean Squared Error is: ",C1)
print("Training - Mean Squared LOG Error is: ",C2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A2,Y2))
display(A2.shape)

ypred2 = pd.DataFrame({'Fatalities' : ypred2}) 
ypred2 = round(ypred2)
ypred2.head(20) 

Regressor_Concases = XGBRegressor(n_estimators = 1000, gamma = 0, learning_rate = 0.9999,  
                                  random_state = 42 , max_depth = 20, max_delta=-0.5, reg_lambda=1,reg_alpha=0)
                                 #scale_pos_weight=10) 
#,colsample_bytree=0) 
Regressor_fatalities = XGBRegressor(n_estimators = 1000 , gamma = 0, learning_rate = 0.9999,  
                                    random_state = 42 , max_depth = 20,max_delta=-0.5,reg_lambda=1,reg_alpha=0)
                                   #scale_pos_weight=10) 
#,colsample_bytree=0)



import math
from sklearn.metrics import mean_squared_log_error
Regressor_Concases.fit(X,Y1.ravel())
A1 = Regressor_Concases.predict(X)
#B = mean_squared_error(A,Y1)
#print("Training - Mean Squared Error is: ",B) 

A1=abs(A1)
Y1=abs(Y1)


B1 = mean_squared_error(A1,Y1)
B2 = mean_squared_log_error(A1,Y1)
print("-------Confirmed Case---------s")
print("Training - Mean Squared Error is: ",B1)
print("Training - Mean Squared LOG Error is: ",B2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A1,Y1))
display(A1.shape)
#print("Training - ROOT Mean Squared Error is: ",math.sqrt(B1))
#print(rmsle(A1,Y1))

ypred = Regressor_Concases.predict(Xt)
ypred = pd.DataFrame({'ConfirmedCases' : ypred}) 
ypred = round(ypred)
ypred.head(20) 

Regressor_fatalities.fit(X,Y2.ravel())
ypred2= Regressor_fatalities.predict(Xt)
A2 = Regressor_fatalities.predict(X)
A2 = np.round(A2)
A2=abs(A2)
Y2=abs(Y2)

C1 = mean_squared_error(A2,Y2)
C2 = mean_squared_log_error(A2,Y2)
print("--------Fatalities-------")
print("Training - Mean Squared Error is: ",C1)
print("Training - Mean Squared LOG Error is: ",C2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A2,Y2))
display(A2.shape)

ypred2 = pd.DataFrame({'Fatalities' : ypred2}) 
ypred2 = round(ypred2)
ypred2.head(20) 

Regressor_Concases = XGBRegressor(n_estimators = 10, gamma = 0, learning_rate = 0.9999,  
                                  random_state = 42 , max_depth = 20, max_delta=-0.5, reg_lambda=1,reg_alpha=0)
                                 #scale_pos_weight=10) 
#,colsample_bytree=0) 
Regressor_fatalities = XGBRegressor(n_estimators = 10 , gamma = 0, learning_rate = 0.9999,  
                                    random_state = 42 , max_depth = 20,max_delta=-0.5,reg_lambda=1,reg_alpha=0)
                                   #scale_pos_weight=10) 
#,colsample_bytree=0)



import math
from sklearn.metrics import mean_squared_log_error
Regressor_Concases.fit(X,Y1.ravel())
A1 = Regressor_Concases.predict(X)
#B = mean_squared_error(A,Y1)
#print("Training - Mean Squared Error is: ",B) 

A1=abs(A1)
Y1=abs(Y1)


B1 = mean_squared_error(A1,Y1)
B2 = mean_squared_log_error(A1,Y1)
print("-------Confirmed Case---------s")
print("Training - Mean Squared Error is: ",B1)
print("Training - Mean Squared LOG Error is: ",B2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A1,Y1))
display(A1.shape)
#print("Training - ROOT Mean Squared Error is: ",math.sqrt(B1))
#print(rmsle(A1,Y1))

ypred = Regressor_Concases.predict(Xt)
ypred = pd.DataFrame({'ConfirmedCases' : ypred}) 
ypred = round(ypred)
ypred.head(20) 

Regressor_fatalities.fit(X,Y2.ravel())
ypred2= Regressor_fatalities.predict(Xt)
A2 = Regressor_fatalities.predict(X)
A2 = np.round(A2)
A2=abs(A2)
Y2=abs(Y2)

C1 = mean_squared_error(A2,Y2)
C2 = mean_squared_log_error(A2,Y2)
print("--------Fatalities-------")
print("Training - Mean Squared Error is: ",C1)
print("Training - Mean Squared LOG Error is: ",C2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A2,Y2))
display(A2.shape)

ypred2 = pd.DataFrame({'Fatalities' : ypred2}) 
ypred2 = round(ypred2)
ypred2.head(20) 

Regressor_Concases = XGBRegressor(n_estimators = 2000, gamma = 0, learning_rate = 0.9999,  
                                  random_state = 42 , max_depth = 20, max_delta=-0.5, reg_lambda=1,reg_alpha=0)
                                 #scale_pos_weight=10) 
#,colsample_bytree=0) 
Regressor_fatalities = XGBRegressor(n_estimators = 2000 , gamma = 0, learning_rate = 0.9999,  
                                    random_state = 42 , max_depth = 20,max_delta=-0.5,reg_lambda=1,reg_alpha=0)
                                   #scale_pos_weight=10) 
#,colsample_bytree=0)



import math
from sklearn.metrics import mean_squared_log_error
Regressor_Concases.fit(X,Y1.ravel())
A1 = Regressor_Concases.predict(X)
#B = mean_squared_error(A,Y1)
#print("Training - Mean Squared Error is: ",B) 

A1=abs(A1)
Y1=abs(Y1)


B1 = mean_squared_error(A1,Y1)
B2 = mean_squared_log_error(A1,Y1)
print("-------Confirmed Case---------s")
print("Training - Mean Squared Error is: ",B1)
print("Training - Mean Squared LOG Error is: ",B2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A1,Y1))
display(A1.shape)
#print("Training - ROOT Mean Squared Error is: ",math.sqrt(B1))
#print(rmsle(A1,Y1))

ypred = Regressor_Concases.predict(Xt)
ypred = pd.DataFrame({'ConfirmedCases' : ypred}) 
ypred = round(ypred)
ypred.head(20) 

Regressor_fatalities.fit(X,Y2.ravel())
ypred2= Regressor_fatalities.predict(Xt)
A2 = Regressor_fatalities.predict(X)
A2 = np.round(A2)
A2=abs(A2)
Y2=abs(Y2)

C1 = mean_squared_error(A2,Y2)
C2 = mean_squared_log_error(A2,Y2)
print("--------Fatalities-------")
print("Training - Mean Squared Error is: ",C1)
print("Training - Mean Squared LOG Error is: ",C2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A2,Y2))
display(A2.shape)

ypred2 = pd.DataFrame({'Fatalities' : ypred2}) 
ypred2 = round(ypred2)
ypred2.head(20) 

Regressor_Concases = XGBRegressor(n_estimators = 4000, gamma = 0, learning_rate = 0.9999,  
                                  random_state = 42 , max_depth = 20, max_delta=-0.5, reg_lambda=1,reg_alpha=0)
                                 #scale_pos_weight=10) 
#,colsample_bytree=0) 
Regressor_fatalities = XGBRegressor(n_estimators = 4000 , gamma = 0, learning_rate = 0.9999,  
                                    random_state = 42 , max_depth = 20,max_delta=-0.5,reg_lambda=1,reg_alpha=0)
                                   #scale_pos_weight=10) 
#,colsample_bytree=0)



import math
from sklearn.metrics import mean_squared_log_error
Regressor_Concases.fit(X,Y1.ravel())
A1 = Regressor_Concases.predict(X)
#B = mean_squared_error(A,Y1)
#print("Training - Mean Squared Error is: ",B) 

A1=abs(A1)
Y1=abs(Y1)


B1 = mean_squared_error(A1,Y1)
B2 = mean_squared_log_error(A1,Y1)
print("-------Confirmed Case---------s")
print("Training - Mean Squared Error is: ",B1)
print("Training - Mean Squared LOG Error is: ",B2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A1,Y1))
display(A1.shape)
#print("Training - ROOT Mean Squared Error is: ",math.sqrt(B1))
#print(rmsle(A1,Y1))

ypred = Regressor_Concases.predict(Xt)
ypred = pd.DataFrame({'ConfirmedCases' : ypred}) 
ypred = round(ypred)
ypred.head(20) 

Regressor_fatalities.fit(X,Y2.ravel())
ypred2= Regressor_fatalities.predict(Xt)
A2 = Regressor_fatalities.predict(X)
A2 = np.round(A2)
A2=abs(A2)
Y2=abs(Y2)

C1 = mean_squared_error(A2,Y2)
C2 = mean_squared_log_error(A2,Y2)
print("--------Fatalities-------")
print("Training - Mean Squared Error is: ",C1)
print("Training - Mean Squared LOG Error is: ",C2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A2,Y2))
display(A2.shape)

ypred2 = pd.DataFrame({'Fatalities' : ypred2}) 
ypred2 = round(ypred2)
ypred2.head(20) 

Regressor_Concases = XGBRegressor(n_estimators = 2000, gamma = 0, learning_rate = 0.9999,  
                                  random_state = 42 , max_depth = 20, max_delta=-0.5, reg_lambda=1,reg_alpha=0)
                                 #scale_pos_weight=10) 
#,colsample_bytree=0) 
Regressor_fatalities = XGBRegressor(n_estimators = 2000 , gamma = 0, learning_rate = 0.9999,  
                                    random_state = 42 , max_depth = 20,max_delta=-0.5,reg_lambda=1,reg_alpha=0)
                                   #scale_pos_weight=10) 
#,colsample_bytree=0)



import math
from sklearn.metrics import mean_squared_log_error
Regressor_Concases.fit(X,Y1.ravel())
A1 = Regressor_Concases.predict(X)
#B = mean_squared_error(A,Y1)
#print("Training - Mean Squared Error is: ",B) 

A1=abs(A1)
Y1=abs(Y1)


B1 = mean_squared_error(A1,Y1)
B2 = mean_squared_log_error(A1,Y1)
print("-------Confirmed Case---------s")
print("Training - Mean Squared Error is: ",B1)
print("Training - Mean Squared LOG Error is: ",B2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A1,Y1))
display(A1.shape)
#print("Training - ROOT Mean Squared Error is: ",math.sqrt(B1))
#print(rmsle(A1,Y1))

ypred = Regressor_Concases.predict(Xt)
ypred = pd.DataFrame({'ConfirmedCases' : ypred}) 
ypred = round(ypred)
ypred.head(20) 

Regressor_fatalities.fit(X,Y2.ravel())
ypred2= Regressor_fatalities.predict(Xt)
A2 = Regressor_fatalities.predict(X)
A2 = np.round(A2)
A2=abs(A2)
Y2=abs(Y2)

C1 = mean_squared_error(A2,Y2)
C2 = mean_squared_log_error(A2,Y2)
print("--------Fatalities-------")
print("Training - Mean Squared Error is: ",C1)
print("Training - Mean Squared LOG Error is: ",C2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A2,Y2))
display(A2.shape)

ypred2 = pd.DataFrame({'Fatalities' : ypred2}) 
ypred2 = round(ypred2)
ypred2.head(20) 

Regressor_Concases = XGBRegressor(n_estimators = 2000, gamma = 0, learning_rate = 0.9999,  
                                  random_state = 42 , max_depth = 20, max_delta=-0.5, reg_lambda=1,reg_alpha=0,
                                 subsample=0)
                                 #scale_pos_weight=10) 
#,colsample_bytree=0) 
Regressor_fatalities = XGBRegressor(n_estimators = 2000 , gamma = 0, learning_rate = 0.9999,  
                                    random_state = 42 , max_depth = 20,max_delta=-0.5,reg_lambda=1,reg_alpha=0,
                                   subsample=0)
                                   #scale_pos_weight=10) 
#,colsample_bytree=0)



import math
from sklearn.metrics import mean_squared_log_error
Regressor_Concases.fit(X,Y1.ravel())
A1 = Regressor_Concases.predict(X)
#B = mean_squared_error(A,Y1)
#print("Training - Mean Squared Error is: ",B) 

A1=abs(A1)
Y1=abs(Y1)


B1 = mean_squared_error(A1,Y1)
B2 = mean_squared_log_error(A1,Y1)
print("-------Confirmed Case---------s")
print("Training - Mean Squared Error is: ",B1)
print("Training - Mean Squared LOG Error is: ",B2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A1,Y1))
display(A1.shape)
#print("Training - ROOT Mean Squared Error is: ",math.sqrt(B1))
#print(rmsle(A1,Y1))

ypred = Regressor_Concases.predict(Xt)
ypred = pd.DataFrame({'ConfirmedCases' : ypred}) 
ypred = round(ypred)
ypred.head(20) 

Regressor_fatalities.fit(X,Y2.ravel())
ypred2= Regressor_fatalities.predict(Xt)
A2 = Regressor_fatalities.predict(X)
A2 = np.round(A2)
A2=abs(A2)
Y2=abs(Y2)

C1 = mean_squared_error(A2,Y2)
C2 = mean_squared_log_error(A2,Y2)
print("--------Fatalities-------")
print("Training - Mean Squared Error is: ",C1)
print("Training - Mean Squared LOG Error is: ",C2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A2,Y2))
display(A2.shape)

ypred2 = pd.DataFrame({'Fatalities' : ypred2}) 
ypred2 = round(ypred2)
ypred2.head(20) 

Regressor_Concases = XGBRegressor(n_estimators = 2000, gamma = 0, learning_rate = 0.9999,  
                                  random_state = 42 , max_depth = 20, max_delta=-0.5, reg_lambda=1,reg_alpha=0,
                                 subsample=0.25)
                                 #scale_pos_weight=10) 
#,colsample_bytree=0) 
Regressor_fatalities = XGBRegressor(n_estimators = 2000 , gamma = 0, learning_rate = 0.9999,  
                                    random_state = 42 , max_depth = 20,max_delta=-0.5,reg_lambda=1,reg_alpha=0,
                                   subsample=0.25)
                                   #scale_pos_weight=10) 
#,colsample_bytree=0)



import math
from sklearn.metrics import mean_squared_log_error
Regressor_Concases.fit(X,Y1.ravel())
A1 = Regressor_Concases.predict(X)
#B = mean_squared_error(A,Y1)
#print("Training - Mean Squared Error is: ",B) 

A1=abs(A1)
Y1=abs(Y1)


B1 = mean_squared_error(A1,Y1)
B2 = mean_squared_log_error(A1,Y1)
print("-------Confirmed Case---------s")
print("Training - Mean Squared Error is: ",B1)
print("Training - Mean Squared LOG Error is: ",B2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A1,Y1))
display(A1.shape)
#print("Training - ROOT Mean Squared Error is: ",math.sqrt(B1))
#print(rmsle(A1,Y1))

ypred = Regressor_Concases.predict(Xt)
ypred = pd.DataFrame({'ConfirmedCases' : ypred}) 
ypred = round(ypred)
ypred.head(20) 

Regressor_fatalities.fit(X,Y2.ravel())
ypred2= Regressor_fatalities.predict(Xt)
A2 = Regressor_fatalities.predict(X)
A2 = np.round(A2)
A2=abs(A2)
Y2=abs(Y2)

C1 = mean_squared_error(A2,Y2)
C2 = mean_squared_log_error(A2,Y2)
print("--------Fatalities-------")
print("Training - Mean Squared Error is: ",C1)
print("Training - Mean Squared LOG Error is: ",C2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A2,Y2))
display(A2.shape)

ypred2 = pd.DataFrame({'Fatalities' : ypred2}) 
ypred2 = round(ypred2)
ypred2.head(20) 

Regressor_Concases = XGBRegressor(n_estimators = 2000, gamma = 0, learning_rate = 0.9999,  
                                  random_state = 42 , max_depth = 20, max_delta=-0.5, reg_lambda=1,reg_alpha=0,
                                 subsample=0.5)
                                 #scale_pos_weight=10) 
#,colsample_bytree=0) 
Regressor_fatalities = XGBRegressor(n_estimators = 2000 , gamma = 0, learning_rate = 0.9999,  
                                    random_state = 42 , max_depth = 20,max_delta=-0.5,reg_lambda=1,reg_alpha=0,
                                   subsample=0.5)
                                   #scale_pos_weight=10) 
#,colsample_bytree=0)



import math
from sklearn.metrics import mean_squared_log_error
Regressor_Concases.fit(X,Y1.ravel())
A1 = Regressor_Concases.predict(X)
#B = mean_squared_error(A,Y1)
#print("Training - Mean Squared Error is: ",B) 

A1=abs(A1)
Y1=abs(Y1)


B1 = mean_squared_error(A1,Y1)
B2 = mean_squared_log_error(A1,Y1)
print("-------Confirmed Case---------s")
print("Training - Mean Squared Error is: ",B1)
print("Training - Mean Squared LOG Error is: ",B2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A1,Y1))
display(A1.shape)
#print("Training - ROOT Mean Squared Error is: ",math.sqrt(B1))
#print(rmsle(A1,Y1))

ypred = Regressor_Concases.predict(Xt)
ypred = pd.DataFrame({'ConfirmedCases' : ypred}) 
ypred = round(ypred)
ypred.head(20) 

Regressor_fatalities.fit(X,Y2.ravel())
ypred2= Regressor_fatalities.predict(Xt)
A2 = Regressor_fatalities.predict(X)
A2 = np.round(A2)
A2=abs(A2)
Y2=abs(Y2)

C1 = mean_squared_error(A2,Y2)
C2 = mean_squared_log_error(A2,Y2)
print("--------Fatalities-------")
print("Training - Mean Squared Error is: ",C1)
print("Training - Mean Squared LOG Error is: ",C2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A2,Y2))
display(A2.shape)

ypred2 = pd.DataFrame({'Fatalities' : ypred2}) 
ypred2 = round(ypred2)
ypred2.head(20) 

Regressor_Concases = XGBRegressor(n_estimators = 2000, gamma = 0, learning_rate = 0.9999,  
                                  random_state = 42 , max_depth = 20, max_delta=-0.5, reg_lambda=1,reg_alpha=0,
                                 subsample=0.99)
                                 #scale_pos_weight=10) 
#,colsample_bytree=0) 
Regressor_fatalities = XGBRegressor(n_estimators = 2000 , gamma = 0, learning_rate = 0.9999,  
                                    random_state = 42 , max_depth = 20,max_delta=-0.5,reg_lambda=1,reg_alpha=0,
                                   subsample=0.99)
                                   #scale_pos_weight=10) 
#,colsample_bytree=0)



import math
from sklearn.metrics import mean_squared_log_error
Regressor_Concases.fit(X,Y1.ravel())
A1 = Regressor_Concases.predict(X)
#B = mean_squared_error(A,Y1)
#print("Training - Mean Squared Error is: ",B) 

A1=abs(A1)
Y1=abs(Y1)


B1 = mean_squared_error(A1,Y1)
B2 = mean_squared_log_error(A1,Y1)
print("-------Confirmed Case---------s")
print("Training - Mean Squared Error is: ",B1)
print("Training - Mean Squared LOG Error is: ",B2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A1,Y1))
display(A1.shape)
#print("Training - ROOT Mean Squared Error is: ",math.sqrt(B1))
#print(rmsle(A1,Y1))

ypred = Regressor_Concases.predict(Xt)
ypred = pd.DataFrame({'ConfirmedCases' : ypred}) 
ypred = round(ypred)
ypred.head(20) 

Regressor_fatalities.fit(X,Y2.ravel())
ypred2= Regressor_fatalities.predict(Xt)
A2 = Regressor_fatalities.predict(X)
A2 = np.round(A2)
A2=abs(A2)
Y2=abs(Y2)

C1 = mean_squared_error(A2,Y2)
C2 = mean_squared_log_error(A2,Y2)
print("--------Fatalities-------")
print("Training - Mean Squared Error is: ",C1)
print("Training - Mean Squared LOG Error is: ",C2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A2,Y2))
display(A2.shape)

ypred2 = pd.DataFrame({'Fatalities' : ypred2}) 
ypred2 = round(ypred2)
ypred2.head(20) 

Regressor_Concases = XGBRegressor(n_estimators = 2000, gamma = 0, learning_rate = 0.9999,  
                                  random_state = 42 , max_depth = 20, max_delta=-0.5, reg_lambda=1,reg_alpha=0,
                                 subsample=0.5)
                                 #scale_pos_weight=10) 
#,colsample_bytree=0) 
Regressor_fatalities = XGBRegressor(n_estimators = 2000 , gamma = 0, learning_rate = 0.9999,  
                                    random_state = 42 , max_depth = 20,max_delta=-0.5,reg_lambda=1,reg_alpha=0,
                                   subsample=0.5)
                                   #scale_pos_weight=10) 
#,colsample_bytree=0)



import math
from sklearn.metrics import mean_squared_log_error
Regressor_Concases.fit(X,Y1.ravel())
A1 = Regressor_Concases.predict(X)
#B = mean_squared_error(A,Y1)
#print("Training - Mean Squared Error is: ",B) 

A1=abs(A1)
Y1=abs(Y1)


B1 = mean_squared_error(A1,Y1)
B2 = mean_squared_log_error(A1,Y1)
print("-------Confirmed Case---------s")
print("Training - Mean Squared Error is: ",B1)
print("Training - Mean Squared LOG Error is: ",B2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A1,Y1))
display(A1.shape)
#print("Training - ROOT Mean Squared Error is: ",math.sqrt(B1))
#print(rmsle(A1,Y1))

ypred = Regressor_Concases.predict(Xt)
ypred = pd.DataFrame({'ConfirmedCases' : ypred}) 
ypred = round(ypred)
ypred.head(20) 

Regressor_fatalities.fit(X,Y2.ravel())
ypred2= Regressor_fatalities.predict(Xt)
A2 = Regressor_fatalities.predict(X)
A2 = np.round(A2)
A2=abs(A2)
Y2=abs(Y2)

C1 = mean_squared_error(A2,Y2)
C2 = mean_squared_log_error(A2,Y2)
print("--------Fatalities-------")
print("Training - Mean Squared Error is: ",C1)
print("Training - Mean Squared LOG Error is: ",C2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A2,Y2))
display(A2.shape)

ypred2 = pd.DataFrame({'Fatalities' : ypred2}) 
ypred2 = round(ypred2)
ypred2.head(20) 

Regressor_Concases = XGBRegressor(n_estimators = 2000, gamma = 0, learning_rate = 0.9999,  
                                  random_state = 42 , max_depth = 20, max_delta=-0.5, reg_lambda=1,reg_alpha=0,
                                 subsample=0.5,min_child_weight=1)
                                 #scale_pos_weight=10) 
#,colsample_bytree=0) 
Regressor_fatalities = XGBRegressor(n_estimators = 2000 , gamma = 0, learning_rate = 0.9999,  
                                    random_state = 42 , max_depth = 20,max_delta=-0.5,reg_lambda=1,reg_alpha=0,
                                   subsample=0.5,min_child_weight=1)
                                   #scale_pos_weight=10) 
#,colsample_bytree=0)



import math
from sklearn.metrics import mean_squared_log_error
Regressor_Concases.fit(X,Y1.ravel())
A1 = Regressor_Concases.predict(X)
#B = mean_squared_error(A,Y1)
#print("Training - Mean Squared Error is: ",B) 

A1=abs(A1)
Y1=abs(Y1)


B1 = mean_squared_error(A1,Y1)
B2 = mean_squared_log_error(A1,Y1)
print("-------Confirmed Case---------s")
print("Training - Mean Squared Error is: ",B1)
print("Training - Mean Squared LOG Error is: ",B2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A1,Y1))
display(A1.shape)
#print("Training - ROOT Mean Squared Error is: ",math.sqrt(B1))
#print(rmsle(A1,Y1))

ypred = Regressor_Concases.predict(Xt)
ypred = pd.DataFrame({'ConfirmedCases' : ypred}) 
ypred = round(ypred)
ypred.head(20) 

Regressor_fatalities.fit(X,Y2.ravel())
ypred2= Regressor_fatalities.predict(Xt)
A2 = Regressor_fatalities.predict(X)
A2 = np.round(A2)
A2=abs(A2)
Y2=abs(Y2)

C1 = mean_squared_error(A2,Y2)
C2 = mean_squared_log_error(A2,Y2)
print("--------Fatalities-------")
print("Training - Mean Squared Error is: ",C1)
print("Training - Mean Squared LOG Error is: ",C2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A2,Y2))
display(A2.shape)

ypred2 = pd.DataFrame({'Fatalities' : ypred2}) 
ypred2 = round(ypred2)
ypred2.head(20) 

Regressor_Concases = XGBRegressor(n_estimators = 2000, gamma = 0, learning_rate = 0.9999,  
                                  random_state = 42 , max_depth = 20, max_delta=-0.5, reg_lambda=1,reg_alpha=0,
                                 subsample=0.5,min_child_weight=50)
                                 #scale_pos_weight=10) 
#,colsample_bytree=0) 
Regressor_fatalities = XGBRegressor(n_estimators = 2000 , gamma = 0, learning_rate = 0.9999,  
                                    random_state = 42 , max_depth = 20,max_delta=-0.5,reg_lambda=1,reg_alpha=0,
                                   subsample=0.5,min_child_weight=50)
                                   #scale_pos_weight=10) 
#,colsample_bytree=0)



import math
from sklearn.metrics import mean_squared_log_error
Regressor_Concases.fit(X,Y1.ravel())
A1 = Regressor_Concases.predict(X)
#B = mean_squared_error(A,Y1)
#print("Training - Mean Squared Error is: ",B) 

A1=abs(A1)
Y1=abs(Y1)


B1 = mean_squared_error(A1,Y1)
B2 = mean_squared_log_error(A1,Y1)
print("-------Confirmed Case---------s")
print("Training - Mean Squared Error is: ",B1)
print("Training - Mean Squared LOG Error is: ",B2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A1,Y1))
display(A1.shape)
#print("Training - ROOT Mean Squared Error is: ",math.sqrt(B1))
#print(rmsle(A1,Y1))

ypred = Regressor_Concases.predict(Xt)
ypred = pd.DataFrame({'ConfirmedCases' : ypred}) 
ypred = round(ypred)
ypred.head(20) 

Regressor_fatalities.fit(X,Y2.ravel())
ypred2= Regressor_fatalities.predict(Xt)
A2 = Regressor_fatalities.predict(X)
A2 = np.round(A2)
A2=abs(A2)
Y2=abs(Y2)

C1 = mean_squared_error(A2,Y2)
C2 = mean_squared_log_error(A2,Y2)
print("--------Fatalities-------")
print("Training - Mean Squared Error is: ",C1)
print("Training - Mean Squared LOG Error is: ",C2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A2,Y2))
display(A2.shape)

ypred2 = pd.DataFrame({'Fatalities' : ypred2}) 
ypred2 = round(ypred2)
ypred2.head(20) 

Regressor_Concases = XGBRegressor(n_estimators = 2000, gamma = 0, learning_rate = 0.9999,  
                                  random_state = 42 , max_depth = 20, max_delta=-0.5, reg_lambda=1,reg_alpha=0,
                                 subsample=0.5,min_child_weight=1)
                                 #scale_pos_weight=10) 
#,colsample_bytree=0) 
Regressor_fatalities = XGBRegressor(n_estimators = 2000 , gamma = 0, learning_rate = 0.9999,  
                                    random_state = 42 , max_depth = 20,max_delta=-0.5,reg_lambda=1,reg_alpha=0,
                                   subsample=0.5,min_child_weight=1)
                                   #scale_pos_weight=10) 
#,colsample_bytree=0)



import math
from sklearn.metrics import mean_squared_log_error
Regressor_Concases.fit(X,Y1.ravel())
A1 = Regressor_Concases.predict(X)
#B = mean_squared_error(A,Y1)
#print("Training - Mean Squared Error is: ",B) 

A1=abs(A1)
Y1=abs(Y1)


B1 = mean_squared_error(A1,Y1)
B2 = mean_squared_log_error(A1,Y1)
print("-------Confirmed Case---------s")
print("Training - Mean Squared Error is: ",B1)
print("Training - Mean Squared LOG Error is: ",B2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A1,Y1))
display(A1.shape)
#print("Training - ROOT Mean Squared Error is: ",math.sqrt(B1))
#print(rmsle(A1,Y1))

ypred = Regressor_Concases.predict(Xt)
ypred = pd.DataFrame({'ConfirmedCases' : ypred}) 
ypred = round(ypred)
ypred.head(20) 

Regressor_fatalities.fit(X,Y2.ravel())
ypred2= Regressor_fatalities.predict(Xt)
A2 = Regressor_fatalities.predict(X)
A2 = np.round(A2)
A2=abs(A2)
Y2=abs(Y2)

C1 = mean_squared_error(A2,Y2)
C2 = mean_squared_log_error(A2,Y2)
print("--------Fatalities-------")
print("Training - Mean Squared Error is: ",C1)
print("Training - Mean Squared LOG Error is: ",C2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A2,Y2))
display(A2.shape)

ypred2 = pd.DataFrame({'Fatalities' : ypred2}) 
ypred2 = round(ypred2)
ypred2.head(20) 

Regressor_Concases = XGBRegressor(n_estimators = 2000, gamma = 0, learning_rate = 0.9999,  
                                  random_state = 42 , max_depth = 1, max_delta=-0.5, reg_lambda=1,reg_alpha=0,
                                 subsample=0.5,min_child_weight=1)
                                 #scale_pos_weight=10) 
#,colsample_bytree=0) 
Regressor_fatalities = XGBRegressor(n_estimators = 2000 , gamma = 0, learning_rate = 0.9999,  
                                    random_state = 42 , max_depth = 1,max_delta=-0.5,reg_lambda=1,reg_alpha=0,
                                   subsample=0.5,min_child_weight=1)
                                   #scale_pos_weight=10) 
#,colsample_bytree=0)



import math
from sklearn.metrics import mean_squared_log_error
Regressor_Concases.fit(X,Y1.ravel())
A1 = Regressor_Concases.predict(X)
#B = mean_squared_error(A,Y1)
#print("Training - Mean Squared Error is: ",B) 

A1=abs(A1)
Y1=abs(Y1)


B1 = mean_squared_error(A1,Y1)
B2 = mean_squared_log_error(A1,Y1)
print("-------Confirmed Case---------s")
print("Training - Mean Squared Error is: ",B1)
print("Training - Mean Squared LOG Error is: ",B2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A1,Y1))
display(A1.shape)
#print("Training - ROOT Mean Squared Error is: ",math.sqrt(B1))
#print(rmsle(A1,Y1))

ypred = Regressor_Concases.predict(Xt)
ypred = pd.DataFrame({'ConfirmedCases' : ypred}) 
ypred = round(ypred)
ypred.head(20) 

Regressor_fatalities.fit(X,Y2.ravel())
ypred2= Regressor_fatalities.predict(Xt)
A2 = Regressor_fatalities.predict(X)
A2 = np.round(A2)
A2=abs(A2)
Y2=abs(Y2)

C1 = mean_squared_error(A2,Y2)
C2 = mean_squared_log_error(A2,Y2)
print("--------Fatalities-------")
print("Training - Mean Squared Error is: ",C1)
print("Training - Mean Squared LOG Error is: ",C2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A2,Y2))
display(A2.shape)

ypred2 = pd.DataFrame({'Fatalities' : ypred2}) 
ypred2 = round(ypred2)
ypred2.head(20) 

Regressor_Concases = XGBRegressor(n_estimators = 2000, gamma = 0, learning_rate = 0.9999,  
                                  random_state = 42 , max_depth = 50, max_delta=-0.5, reg_lambda=1,reg_alpha=0,
                                 subsample=0.5,min_child_weight=1)
                                 #scale_pos_weight=10) 
#,colsample_bytree=0) 
Regressor_fatalities = XGBRegressor(n_estimators = 2000 , gamma = 0, learning_rate = 0.9999,  
                                    random_state = 42 , max_depth = 50,max_delta=-0.5,reg_lambda=1,reg_alpha=0,
                                   subsample=0.5,min_child_weight=1)
                                   #scale_pos_weight=10) 
#,colsample_bytree=0)



import math
from sklearn.metrics import mean_squared_log_error
Regressor_Concases.fit(X,Y1.ravel())
A1 = Regressor_Concases.predict(X)
#B = mean_squared_error(A,Y1)
#print("Training - Mean Squared Error is: ",B) 

A1=abs(A1)
Y1=abs(Y1)


B1 = mean_squared_error(A1,Y1)
B2 = mean_squared_log_error(A1,Y1)
print("-------Confirmed Case---------s")
print("Training - Mean Squared Error is: ",B1)
print("Training - Mean Squared LOG Error is: ",B2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A1,Y1))
display(A1.shape)
#print("Training - ROOT Mean Squared Error is: ",math.sqrt(B1))
#print(rmsle(A1,Y1))

ypred = Regressor_Concases.predict(Xt)
ypred = pd.DataFrame({'ConfirmedCases' : ypred}) 
ypred = round(ypred)
ypred.head(20) 

Regressor_fatalities.fit(X,Y2.ravel())
ypred2= Regressor_fatalities.predict(Xt)
A2 = Regressor_fatalities.predict(X)
A2 = np.round(A2)
A2=abs(A2)
Y2=abs(Y2)

C1 = mean_squared_error(A2,Y2)
C2 = mean_squared_log_error(A2,Y2)
print("--------Fatalities-------")
print("Training - Mean Squared Error is: ",C1)
print("Training - Mean Squared LOG Error is: ",C2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A2,Y2))
display(A2.shape)

ypred2 = pd.DataFrame({'Fatalities' : ypred2}) 
ypred2 = round(ypred2)
ypred2.head(20) 

Regressor_Concases = XGBRegressor(n_estimators = 2000, gamma = 0, learning_rate = 0.9999,  
                                  random_state = 42 , max_depth = 20, max_delta=-0.5, reg_lambda=1,reg_alpha=0,
                                 subsample=0.5,min_child_weight=1)
                                 #scale_pos_weight=10) 
#,colsample_bytree=0) 
Regressor_fatalities = XGBRegressor(n_estimators = 2000 , gamma = 0, learning_rate = 0.9999,  
                                    random_state = 42 , max_depth = 20,max_delta=-0.5,reg_lambda=1,reg_alpha=0,
                                   subsample=0.5,min_child_weight=1)
                                   #scale_pos_weight=10) 
#,colsample_bytree=0)



import math
from sklearn.metrics import mean_squared_log_error
Regressor_Concases.fit(X,Y1.ravel())
A1 = Regressor_Concases.predict(X)
#B = mean_squared_error(A,Y1)
#print("Training - Mean Squared Error is: ",B) 

A1=abs(A1)
Y1=abs(Y1)


B1 = mean_squared_error(A1,Y1)
B2 = mean_squared_log_error(A1,Y1)
print("-------Confirmed Case---------s")
print("Training - Mean Squared Error is: ",B1)
print("Training - Mean Squared LOG Error is: ",B2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A1,Y1))
display(A1.shape)
#print("Training - ROOT Mean Squared Error is: ",math.sqrt(B1))
#print(rmsle(A1,Y1))

ypred = Regressor_Concases.predict(Xt)
ypred = pd.DataFrame({'ConfirmedCases' : ypred}) 
ypred = round(ypred)
ypred.head(20) 

Regressor_fatalities.fit(X,Y2.ravel())
ypred2= Regressor_fatalities.predict(Xt)
A2 = Regressor_fatalities.predict(X)
A2 = np.round(A2)
A2=abs(A2)
Y2=abs(Y2)

C1 = mean_squared_error(A2,Y2)
C2 = mean_squared_log_error(A2,Y2)
print("--------Fatalities-------")
print("Training - Mean Squared Error is: ",C1)
print("Training - Mean Squared LOG Error is: ",C2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A2,Y2))
display(A2.shape)

ypred2 = pd.DataFrame({'Fatalities' : ypred2}) 
ypred2 = round(ypred2)
ypred2.head(20) 

Regressor_Concases = XGBRegressor(n_estimators = 2000, gamma = 0, learning_rate = 0.9999,  
                                  random_state = 42 , max_depth = 20, max_delta=-0.5, reg_lambda=1,reg_alpha=0,
                                 subsample=0.5,min_child_weight=1,seed=0)
                                 #scale_pos_weight=10) 
#,colsample_bytree=0) 
Regressor_fatalities = XGBRegressor(n_estimators = 2000 , gamma = 0, learning_rate = 0.9999,  
                                    random_state = 42 , max_depth = 20,max_delta=-0.5,reg_lambda=1,reg_alpha=0,
                                   subsample=0.5,min_child_weight=1,seed=0)
                                   #scale_pos_weight=10) 
#,colsample_bytree=0)



import math
from sklearn.metrics import mean_squared_log_error
Regressor_Concases.fit(X,Y1.ravel())
A1 = Regressor_Concases.predict(X)
#B = mean_squared_error(A,Y1)
#print("Training - Mean Squared Error is: ",B) 

A1=abs(A1)
Y1=abs(Y1)


B1 = mean_squared_error(A1,Y1)
B2 = mean_squared_log_error(A1,Y1)
print("-------Confirmed Case---------s")
print("Training - Mean Squared Error is: ",B1)
print("Training - Mean Squared LOG Error is: ",B2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A1,Y1))
display(A1.shape)
#print("Training - ROOT Mean Squared Error is: ",math.sqrt(B1))
#print(rmsle(A1,Y1))

ypred = Regressor_Concases.predict(Xt)
ypred = pd.DataFrame({'ConfirmedCases' : ypred}) 
ypred = round(ypred)
ypred.head(20) 

Regressor_fatalities.fit(X,Y2.ravel())
ypred2= Regressor_fatalities.predict(Xt)
A2 = Regressor_fatalities.predict(X)
A2 = np.round(A2)
A2=abs(A2)
Y2=abs(Y2)

C1 = mean_squared_error(A2,Y2)
C2 = mean_squared_log_error(A2,Y2)
print("--------Fatalities-------")
print("Training - Mean Squared Error is: ",C1)
print("Training - Mean Squared LOG Error is: ",C2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A2,Y2))
display(A2.shape)

ypred2 = pd.DataFrame({'Fatalities' : ypred2}) 
ypred2 = round(ypred2)
ypred2.head(20) 

Regressor_Concases = XGBRegressor(n_estimators = 2000, gamma = 0, learning_rate = 0.9999,  
                                  random_state = 42 , max_depth = 20, max_delta=-0.5, reg_lambda=1,reg_alpha=0,
                                 subsample=0.5,min_child_weight=1,seed=10)
                                 #scale_pos_weight=10) 
#,colsample_bytree=0) 
Regressor_fatalities = XGBRegressor(n_estimators = 2000 , gamma = 0, learning_rate = 0.9999,  
                                    random_state = 42 , max_depth = 20,max_delta=-0.5,reg_lambda=1,reg_alpha=0,
                                   subsample=0.5,min_child_weight=1,seed=10)
                                   #scale_pos_weight=10) 
#,colsample_bytree=0)



import math
from sklearn.metrics import mean_squared_log_error
Regressor_Concases.fit(X,Y1.ravel())
A1 = Regressor_Concases.predict(X)
#B = mean_squared_error(A,Y1)
#print("Training - Mean Squared Error is: ",B) 

A1=abs(A1)
Y1=abs(Y1)


B1 = mean_squared_error(A1,Y1)
B2 = mean_squared_log_error(A1,Y1)
print("-------Confirmed Case---------s")
print("Training - Mean Squared Error is: ",B1)
print("Training - Mean Squared LOG Error is: ",B2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A1,Y1))
display(A1.shape)
#print("Training - ROOT Mean Squared Error is: ",math.sqrt(B1))
#print(rmsle(A1,Y1))

ypred = Regressor_Concases.predict(Xt)
ypred = pd.DataFrame({'ConfirmedCases' : ypred}) 
ypred = round(ypred)
ypred.head(20) 

Regressor_fatalities.fit(X,Y2.ravel())
ypred2= Regressor_fatalities.predict(Xt)
A2 = Regressor_fatalities.predict(X)
A2 = np.round(A2)
A2=abs(A2)
Y2=abs(Y2)

C1 = mean_squared_error(A2,Y2)
C2 = mean_squared_log_error(A2,Y2)
print("--------Fatalities-------")
print("Training - Mean Squared Error is: ",C1)
print("Training - Mean Squared LOG Error is: ",C2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A2,Y2))
display(A2.shape)

ypred2 = pd.DataFrame({'Fatalities' : ypred2}) 
ypred2 = round(ypred2)
ypred2.head(20) 

Regressor_Concases = XGBRegressor(n_estimators = 2000, gamma = 0, learning_rate = 0.9999,  
                                  random_state = 42 , max_depth = 20, max_delta=-0.5, reg_lambda=1,reg_alpha=0,
                                 subsample=0.5,min_child_weight=1,seed=100)
                                 #scale_pos_weight=10) 
#,colsample_bytree=0) 
Regressor_fatalities = XGBRegressor(n_estimators = 2000 , gamma = 0, learning_rate = 0.9999,  
                                    random_state = 42 , max_depth = 20,max_delta=-0.5,reg_lambda=1,reg_alpha=0,
                                   subsample=0.5,min_child_weight=1,seed=100)
                                   #scale_pos_weight=10) 
#,colsample_bytree=0)



import math
from sklearn.metrics import mean_squared_log_error
Regressor_Concases.fit(X,Y1.ravel())
A1 = Regressor_Concases.predict(X)
#B = mean_squared_error(A,Y1)
#print("Training - Mean Squared Error is: ",B) 

A1=abs(A1)
Y1=abs(Y1)


B1 = mean_squared_error(A1,Y1)
B2 = mean_squared_log_error(A1,Y1)
print("-------Confirmed Case---------s")
print("Training - Mean Squared Error is: ",B1)
print("Training - Mean Squared LOG Error is: ",B2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A1,Y1))
display(A1.shape)
#print("Training - ROOT Mean Squared Error is: ",math.sqrt(B1))
#print(rmsle(A1,Y1))

ypred = Regressor_Concases.predict(Xt)
ypred = pd.DataFrame({'ConfirmedCases' : ypred}) 
ypred = round(ypred)
ypred.head(20) 

Regressor_fatalities.fit(X,Y2.ravel())
ypred2= Regressor_fatalities.predict(Xt)
A2 = Regressor_fatalities.predict(X)
A2 = np.round(A2)
A2=abs(A2)
Y2=abs(Y2)

C1 = mean_squared_error(A2,Y2)
C2 = mean_squared_log_error(A2,Y2)
print("--------Fatalities-------")
print("Training - Mean Squared Error is: ",C1)
print("Training - Mean Squared LOG Error is: ",C2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A2,Y2))
display(A2.shape)

ypred2 = pd.DataFrame({'Fatalities' : ypred2}) 
ypred2 = round(ypred2)
ypred2.head(20) 

Regressor_Concases = XGBRegressor(n_estimators = 2000, gamma = 0, learning_rate = 0.9999,  
                                  random_state = 42 , max_depth = 20, max_delta=-0.5, reg_lambda=1,reg_alpha=0,
                                 subsample=0.5,min_child_weight=1,seed=10)
                                 #scale_pos_weight=10) 
#,colsample_bytree=0) 
Regressor_fatalities = XGBRegressor(n_estimators = 2000 , gamma = 0, learning_rate = 0.9999,  
                                    random_state = 42 , max_depth = 20,max_delta=-0.5,reg_lambda=1,reg_alpha=0,
                                   subsample=0.5,min_child_weight=1,seed=10)
                                   #scale_pos_weight=10) 
#,colsample_bytree=0)



import math
from sklearn.metrics import mean_squared_log_error
Regressor_Concases.fit(X,Y1.ravel())
A1 = Regressor_Concases.predict(X)
#B = mean_squared_error(A,Y1)
#print("Training - Mean Squared Error is: ",B) 

A1=abs(A1)
Y1=abs(Y1)


B1 = mean_squared_error(A1,Y1)
B2 = mean_squared_log_error(A1,Y1)
print("-------Confirmed Case---------s")
print("Training - Mean Squared Error is: ",B1)
print("Training - Mean Squared LOG Error is: ",B2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A1,Y1))
display(A1.shape)
#print("Training - ROOT Mean Squared Error is: ",math.sqrt(B1))
#print(rmsle(A1,Y1))

ypred = Regressor_Concases.predict(Xt)
ypred = pd.DataFrame({'ConfirmedCases' : ypred}) 
ypred = round(ypred)
ypred.head(20) 

Regressor_fatalities.fit(X,Y2.ravel())
ypred2= Regressor_fatalities.predict(Xt)
A2 = Regressor_fatalities.predict(X)
A2 = np.round(A2)
A2=abs(A2)
Y2=abs(Y2)

C1 = mean_squared_error(A2,Y2)
C2 = mean_squared_log_error(A2,Y2)
print("--------Fatalities-------")
print("Training - Mean Squared Error is: ",C1)
print("Training - Mean Squared LOG Error is: ",C2)
print("Training - ROOT Mean Squared LOG Error is: ",rmsle(A2,Y2))
display(A2.shape)

ypred2 = pd.DataFrame({'Fatalities' : ypred2}) 
ypred2 = round(ypred2)
ypred2.head(20) 

#final_pred.to_csv('submission.csv', index=False)