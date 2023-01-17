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

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

import csv

import numpy as np

import operator

import random

import datetime

import math





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

from sklearn.metrics import mean_squared_log_error

from math import sqrt



%matplotlib inline

import matplotlib.pyplot as plt



import statsmodels.api as sm

import statsmodels.formula.api as smf
PATH ='/kaggle/input/covid19-global-forecasting-week-4'

trainingdata = pd.read_csv(f'{PATH}/train.csv')

testdata = pd.read_csv(f'{PATH}/test.csv')
trainingdata.info()

testdata.info()
train = trainingdata

test = testdata
train.head() #displaying the dataset
train.shape

trainingdata["ConfirmedCases"]
trainingdata["Fatalities"]
test.shape
test.head()
%matplotlib inline

import matplotlib.pyplot as plt

trainingdata.hist(bins = 25, figsize = (20,15))

plt.show()
trainingdata.boxplot(column=['Fatalities', 'ConfirmedCases', 'Id'])
%matplotlib inline

import matplotlib.pyplot as plt

trainingdata.plot(kind='scatter', x='ConfirmedCases', y='Fatalities')

plt.legend()
corr_matrix = trainingdata.corr()

corr_matrix ["ConfirmedCases"].sort_values(ascending = False)
corr_matrix = trainingdata.corr()

corr_matrix ["Fatalities"].sort_values(ascending = False)
corr_matrix = trainingdata.corr()

corr_matrix ["Id"].sort_values(ascending = False)
test.head()
trainingdate = pd.to_datetime(trainingdata["Date"])

testdate = pd.to_datetime(testdata["Date"]) #datet

print (trainingdate)
ltrainingdate = int(len(trainingdate))

ltestdate = int(len(testdate))

print("The length of the date in training data is", ltrainingdate)

print("The length of the date in test data is", ltestdate)
m = []

d = []

for i in range(0,ltrainingdate):

    dx = (trainingdate[i].strftime("%d")) #converts date to a string

    mx = (trainingdate[i].strftime("%m"))

    m.append(int(mx))

    d.append(int(dx))



mt = []

dt = []

for i in range(0,ltestdate):

    dtx = (testdate[i].strftime("%d"))

    mtx = (testdate[i].strftime("%m"))

    mt.append(int(mtx))

    dt.append(int(dtx))

m = []

d = []

for i in range(0,ltrainingdate):

    dx = (trainingdate[i].strftime("%d")) #converts date to a string

    mx = (trainingdate[i].strftime("%m"))

    m.append(int(mx))

    d.append(int(dx))



mt = []

dt = []

for i in range(0,ltestdate):

    dtx = (testdate[i].strftime("%d"))

    mtx = (testdate[i].strftime("%m"))

    mt.append(int(mtx))

    dt.append(int(dtx))

train.insert(6,"Month",m,False) #added date in the dataset

train.insert(7,"Day",d,False)

test.insert(4,"Month",mt,False)

test.insert(5,"Day",dt,False)
train.head() #displaying the dataset
print("Training Data")

traindays = trainingdata['Date'].nunique() #To make no of unique dates

print("Number of Country_Region: ", trainingdata['Country_Region'].nunique()) #To make no of unique countries

print("Number of Province_State: ", trainingdata['Province_State'].nunique()) #To make no of unique states

print("Number of Days: ", traindays)



notrain = trainingdata['Id'].nunique()

print("Number of datapoints in train:", notrain)

lotrain = int(notrain/traindays) #no of data points in train / traindays (27000/87)

print("L Trains:", lotrain)

print("Test Data")

testdays = testdata['Date'].nunique()

print("Number of Days: ", testdays)

notest = testdata['ForecastId'].nunique()

print("Number of datapoints in test:", notest)

lotest = int(notest/testdays)

print("L Test:", lotest)



zt = testdate[0]

daycount = []

for i in range(0,lotrain):

    for j in range(1,traindays+1):

        daycount.append(j)



for i in range(traindays): #87

    if(zt == trainingdate[i]):

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



train.Province_State = train.Province_State.fillna(0) #Used to replace the null values #Making it 0 here

empty = 0

for i in range(0,traincount):

    if(train.Province_State[i] == empty):

        train.Province_State[i] = train.Country_Region[i]  
test.Province_State = test.Province_State.fillna(0)

empty = 0

for i in range(0,testcount):

    if(test.Province_State[i] == empty):

        test.Province_State[i] = test.Country_Region[i]
train.head()
label = preprocessing.LabelEncoder() #categorizing based on countries and province_state

train.Country_Region = label.fit_transform(train.Country_Region)

train.Province_State = label.fit_transform(train.Province_State)
test.Country_Region = label.fit_transform(test.Country_Region)

test.Province_State = label.fit_transform(test.Province_State)

X = np.c_[train["Province_State"], train["Country_Region"], train["DayCount"], train["Month"], train["Day"]]

Xt = np.c_[test["Province_State"], test["Country_Region"], test["DayCount"], test["Month"], test["Day"]]
X.shape
Xt.shape
Y1 = train["ConfirmedCases"]

Y2 = train["Fatalities"]
Y1.shape
Y2.shape
regr = XGBRegressor(n_estimators = 1500, gamma = 0, learning_rate = 0.8, random_state = 42, max_depth = 50, subsample = 1, reg_lambda = 0, reg_alpha = 0.5) # 0.0055

regr1 = XGBRegressor(n_estimators = 1500, gamma = 0, learning_rate = 0.8, random_state = 42, max_depth = 50, subsample = 1, reg_lambda = 0, reg_alpha = 0.5)
def rmsle(y_true, y_pred):

    return mean_squared_log_error(y_true, y_pred)**(1/2);
regr.fit(X,Y1.ravel()) #ravel returns a 1D array

yscore = regr.predict(X)

a = abs(yscore)

b=abs(Y1)

ascore = (rmsle(a,b))

print("RMSLE of Confirmed Cases is ",ascore) # learning rate = 0.02 , 0.129999711
ypred = regr.predict(Xt)

ypred = pd.DataFrame({'ConfirmedCases' : ypred}) 

ypred = round(ypred)

ypred.head(20) #273 281 299 349 
regr1.fit(X,Y2.ravel())

ypred2 = regr1.predict(Xt)

yptest = regr1.predict(X)

yptest = np.round(yptest)

c = abs(yptest)

d = abs(Y2)

ascore = rmsle(c,d)

print("RMSLE of Fatalities is", ascore)

ypred2 = pd.DataFrame({'Fatalities' : ypred2}) 

ypred2 = round(ypred2)

ypred2.head(20) # 6 6 7 7 11
ypred2.head(15)
ypc = pd.DataFrame()

forecast = test["ForecastId"]
ypc.insert(0,"ForecastId",forecast,False)

ypc.insert(1,"ConfirmedCases",ypred,False)

ypc.insert(2, "Fatalities",ypred2,False)
ypc
print(ypc)
ypc.to_csv('submission.csv', index=False)