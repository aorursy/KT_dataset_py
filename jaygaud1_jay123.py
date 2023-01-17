import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(context="notebook", style="darkgrid", palette="deep", font="sans-serif", font_scale=1, color_codes=True)
data= pd.read_csv("../input/covid19-global-forecasting-week-1/train.csv")
data["Date"].max() ## 2020-03-20

data = data[data["Date"]<="2020-03-20"]
data["Date"].max() ## 2020-03-11
data["Date"] = data["Date"].apply(lambda x: x.replace("-",""))

data["Date"]  = data["Date"].astype(int)
data = data.drop(['Province/State'],axis=1)

data = data.dropna()

data.isnull().sum()
data.head()
test = pd.read_csv("../input/covid19-global-forecasting-week-1/test.csv")   

test["Date"] = test["Date"].apply(lambda x: x.replace("-",""))

test["Date"]  = test["Date"].astype(int)



test["Lat"]  = test["Lat"].fillna(31.8257)

test["Long"]  = test["Long"].fillna(117.2264)

test.isnull().sum()

#31.8257,117.2264

test["Date"].min() ## 2020-03-12
x = data[['Lat', 'Long', 'Date']]

y1 = data[['ConfirmedCases']]

y2 = data[['Fatalities']]

x_test = test[['Lat', 'Long', 'Date']]
from sklearn.tree import DecisionTreeClassifier

Tree_model = DecisionTreeClassifier(criterion='entropy')
##

Tree_model.fit(x,y1)

pred1 = Tree_model.predict(x_test)

pred1 = pd.DataFrame(pred1)

pred1.columns = ["ConfirmedCases_prediction"]
pred1.head()
##

Tree_model.fit(x,y2)

pred2 = Tree_model.predict(x_test)

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


#### By using DecisionTreeRegressor

from sklearn.tree import DecisionTreeRegressor  

regressor = DecisionTreeRegressor(random_state = 0) 
regressor.fit(x,y1)

pred_r1 = regressor.predict(x_test)

pred_r1 = pd.DataFrame(pred_r1)

pred_r1.columns = ["ConfirmedCases_prediction"]

regressor.fit(x,y2)

pred_r2 = regressor.predict(x_test)

pred_r2 = pd.DataFrame(pred_r2)

pred_r2.columns = ["Death_prediction"]
OP_dr = pd.concat([sub_new,pred_r1,pred_r2],axis=1)

OP_dr.head()

OP_dr.columns = [ 'ForecastId','ConfirmedCases', 'Fatalities']

OP_dr["ConfirmedCases"] = OP_dr["ConfirmedCases"].astype(int)

OP_dr["Fatalities"] = OP_dr["Fatalities"].astype(int)

OP_dr.head()

OP_dr.to_csv("submission.csv",index=False)