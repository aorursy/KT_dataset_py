!pip install regressors
import numpy as np 

import pandas as pd 

import datetime as dt

from regressors import stats

from sklearn import linear_model as lm

import statsmodels.formula.api as sm

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression 

from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from mlxtend.feature_selection import SequentialFeatureSelector as sfs

from sklearn.model_selection import KFold, cross_val_score,cross_val_predict, LeaveOneOut



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/train.csv")

data = data.dropna()
data['Weather'] = data['Weather'].map({'Clear': 1, 'Cloudy': 2,'Light Rain': 3,'Heavy Rain': 4})

data['Season'] = data['Season'].map({'Spring': 1, 'Summer': 2,'Fall': 3,'Winter': 4})

#data.head()
#inputDF = data[["Time","IsHoliday","IsWorkingDay","Weather","Temperature","WindSpeed","Season","AdoptedTemperature", "Humidity", "Demand"]]

inputDF = data[["Humidity","AdoptedTemperature","Time","IsWorkingDay","IsHoliday"]]

inputDF['Time'] = pd.to_datetime(inputDF['Time'], format = '%H:%M:%S')

#inputDF['date-time'] = inputDF['Date'] + " " + inputDF['Time']

inputDF['Time-hour'] = inputDF['Time'].dt.hour

inputDF = inputDF.drop(columns=["Time"])



#inputDF.head()
outputDF = data[["Demand"]]

model = LinearRegression()

kf = KFold(12, shuffle=True, random_state=42).get_n_splits(inputDF)

rmse = np.sqrt(-cross_val_score(model, inputDF, outputDF, scoring="neg_mean_squared_error", cv = kf))

print(rmse.mean())
result = model.fit(inputDF, outputDF)
test = pd.read_csv("../input/test.csv")

test.head()
test['Weather'] = test['Weather'].map({'Clear': 1, 'Cloudy': 2,'Light Rain': 3,'Heavy Rain': 4})

test['Season'] = test['Season'].map({'Spring': 1, 'Summer': 2,'Fall': 3,'Winter': 4})

data.head()
inputDF1 = test[["Humidity","AdoptedTemperature","Time","IsHoliday","IsWorkingDay"]]

inputDF1['Time'] = pd.to_datetime(inputDF1['Time'], format = '%H:%M:%S')

inputDF1['Time-hour'] = inputDF1['Time'].dt.hour

inputDF1 = inputDF1.drop(columns=["Time"])



inputDF1.head()
newinput = inputDF1[["Humidity","AdoptedTemperature","Time-hour","IsHoliday","IsWorkingDay"]]

y_pred = model.predict(newinput)

print(type(y_pred))
submissionDF = pd.DataFrame({"Id": test["Id"],"Demand":y_pred[:,0].astype(int)})

#submissionDF.head()

submissionDF.to_csv('Submissionv4.csv',index=False)
#data preproccessing

d = pd.read_csv("../input/train.csv")

d = d.dropna()



#Model Fit

inputDF1 = d[["Humidity","AdoptedTemperature","Time"]]

inputDF1['Time'] = pd.to_datetime(inputDF1['Time'], format = '%H:%M:%S')

inputDF1['Time-hour'] = inputDF1['Time'].dt.hour

inputDF1 = inputDF1.drop(columns=["Time"])



inputDF1.head()

outputDf = d[['Demand']]

logisticRegr = LogisticRegression()

logisticRegr.fit(inputDF1, outputDf)





rmse = np.sqrt(-cross_val_score(model, inputDF, outputDF, scoring="neg_mean_squared_error", cv = kf))

print(rmse.mean())

#print(logisticRegr.intercept_)

#print(logisticRegr.coef_)



#outcomeDF = d[["Demand"]]

# model = lm.LinearRegression()

# results = model.fit(inputDF,outcomeDF)



# #Regression coefficients 

# print(model.intercept_, model.coef_)