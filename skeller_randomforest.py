#Load required Libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(context="notebook", style="darkgrid", palette="deep", font="sans-serif", font_scale=1, color_codes=True)

#Load Data and remove hyphen from Date column after convert the column to int

data= pd.read_csv("../input/covid19-global-forecasting-week-1/train.csv")

data["Date"] = data["Date"].apply(lambda x: x.replace("-",""))

data["Date"]  = data["Date"].astype(int)

#print first five rows

data.head()
#drop Province column and all not available entries

data = data.drop(['Province/State'],axis=1)

data = data.dropna()

data.isnull().sum()

test = pd.read_csv("../input/covid19-global-forecasting-week-1/test.csv")   

test["Date"] = test["Date"].apply(lambda x: x.replace("-",""))

test["Date"]  = test["Date"].astype(int)



test["Lat"]  = test["Lat"].fillna(12.5211)

test["Long"]  = test["Long"].fillna(69.9683)

test.isnull().sum()

#Asign columns for training and testing



x =data[['Lat', 'Long', 'Date']]

y1 = data[['ConfirmedCases']]

y2 = data[['Fatalities']]

x_test = test[['Lat', 'Long', 'Date']]

#y_test = test[['ConfirmedCases']]
#We are going to use Random Forest classifier for the forecast

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=200)
##

model.fit(x,y1)

pred1 = model.predict(x_test)

pred1 = pd.DataFrame(pred1)

pred1.columns = ["ConfirmedCases_prediction"]
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

                      max_depth=None, max_features='auto', max_leaf_nodes=None, 

                      n_estimators=150, random_state=None, n_jobs=1, verbose=0)
pred1.head()
##

model.fit(x,y2)

pred2 = model.predict(x_test)

pred2 = pd.DataFrame(pred2)

pred2.columns = ["Death_prediction"]
pred2.head()
Sub = pd.read_csv("../input/covid19-global-forecasting-week-1/submission.csv")

Sub.columns

sub_new = Sub[["ForecastId"]]
OP = pd.concat([pred1,pred2,sub_new],axis=1)

OP.head()

OP.columns = ['ConfirmedCases', 'Fatalities', 'ForecastId']

OP = OP[['ForecastId','ConfirmedCases', 'Fatalities']]

OP["ConfirmedCases"] = OP["ConfirmedCases"].astype(int)

OP["Fatalities"] = OP["Fatalities"].astype(int)

OP.head()
complete_test= pd.merge(test, OP, how="left", on="ForecastId")

complete_test.to_csv('complete_test.csv',index=False)
OP.to_csv("submission.csv",index=False)