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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import OrdinalEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, SGDRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error

from sklearn.tree import DecisionTreeRegressor

from sklearn.svm import SVR

#from sklearn.
train_df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/train.csv")

test_df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/test.csv")

world_pop_df = pd.read_csv("/kaggle/input/population-by-country-2020/population_by_country_2020.csv")
test_df.info()
train_df.Province_State.unique()
test_df.head()
world_pop_df.info()
columns = ['country','population','yearly_change','net_change','density','land_area','migrants_net','fert_rate','med_age','urban_pop','world_share']

world_pop_df.columns = columns

world_pop_df.world_share = world_pop_df.world_share.str.strip("%").astype(float)/100

print(world_pop_df.head())
set(train_df.Country_Region) - set(world_pop_df.country)

set(test_df.Country_Region) - set(world_pop_df.country)
set(world_pop_df.country) - set(train_df.Country_Region)
train_df.Country_Region.replace(to_replace="Korea, South", value = "South Korea", inplace = True)

train_df.Country_Region.replace(to_replace="US", value = "United States", inplace = True)

train_df.Country_Region.replace(to_replace="Congo (Brazzaville)", value = "Congo", inplace = True)

train_df.Country_Region.replace(to_replace="Congo (Kinshasa)", value = "Congo", inplace = True)

train_df.Country_Region.replace(to_replace="Saint Kitts and Nevis", value = "Saint Kitts & Nevis", inplace = True)

train_df.Country_Region.replace(to_replace="Saint Vincent and the Grenadines", value = "St. Vincent & Grenadines", inplace = True)

train_df.Country_Region.replace(to_replace="Cote d'Ivoire", value = "Côte d'Ivoire", inplace = True)

train_df.Country_Region.replace(to_replace="Czechia", value = "Czech Republic (Czechia)", inplace = True)

train_df.Country_Region.replace(to_replace="Taiwan*", value = "Taiwan", inplace = True)

train_df[train_df.Country_Region == "Congo"]



test_df.Country_Region.replace(to_replace="Korea, South", value = "South Korea", inplace = True)

test_df.Country_Region.replace(to_replace="US", value = "United States", inplace = True)

test_df.Country_Region.replace(to_replace="Congo (Brazzaville)", value = "Congo", inplace = True)

test_df.Country_Region.replace(to_replace="Congo (Kinshasa)", value = "Congo", inplace = True)

test_df.Country_Region.replace(to_replace="Saint Kitts and Nevis", value = "Saint Kitts & Nevis", inplace = True)

test_df.Country_Region.replace(to_replace="Saint Vincent and the Grenadines", value = "St. Vincent & Grenadines", inplace = True)

test_df.Country_Region.replace(to_replace="Cote d'Ivoire", value = "Côte d'Ivoire", inplace = True)

test_df.Country_Region.replace(to_replace="Czechia", value = "Czech Republic (Czechia)", inplace = True)

test_df.Country_Region.replace(to_replace="Taiwan*", value = "Taiwan", inplace = True)

test_df[test_df.Country_Region == "Congo"]
set(train_df.Country_Region) - set(world_pop_df.country)
train_df = train_df.merge(world_pop_df, how="left", left_on = train_df.Country_Region, right_on = world_pop_df.country)
train_df.Province_State.fillna(train_df.Country_Region, inplace = True)
train_df['Date'] = pd.to_datetime(train_df['Date'], format = "%Y-%m-%d")
data = pd.read_csv("/kaggle/input/covid-dataset/Dataset_covid19 (2).csv")
data.fillna(0,inplace=True)

data.drop(["Column1","day","week","dayofweek","year","month","med_age"], axis=1, inplace=True)

data.info()
test_df.Province_State.fillna(test_df.Country_Region,inplace=True)
data1 = data.groupby(["Province_State"]).sum()
data1.drop(["ConfirmedCases","Fatalities"], axis=1, inplace=True)
testdf = pd.merge(test_df, data1, how = "left", on="Province_State")

testdf_final = testdf.drop(["Id"],axis=1)
testdf_final.school = 1

testdf_final.office = 1

testdf_final.Restaurants = 1
def create_date(df):

    df['day'] = df.Date.dt.day

    df['week'] = df.Date.dt.week

    df['dayofweek'] = df.Date.dt.dayofweek

    df['month'] = df.Date.dt.month

    df['year'] = df.Date.dt.year
data['Date'] = pd.to_datetime(data['Date'], format = "%d-%m-%Y")

testdf_final['Date'] = pd.to_datetime(testdf_final['Date'], format = "%Y-%m-%d")
create_date(data)

create_date(testdf_final)
X = pd.get_dummies(data, columns = ["Province_State","Country_Region"], prefix = ["Province","Country"])

Xtest = pd.get_dummies(testdf_final, columns = ["Province_State","Country_Region"], prefix = ["Province","Country"])
X.drop("Date",axis=1,inplace=True)

Xtest.drop("Date",axis=1,inplace=True)
y_cc = X.loc[:,"ConfirmedCases"]

X.drop("ConfirmedCases", axis=1,inplace=True)



y_f = X.loc[:,"Fatalities"]

X.drop("Fatalities", axis=1, inplace=True)
X_train_cc, X_test_cc, y_train_cc, y_test_cc = train_test_split(X,y_cc, test_size = 0.2, random_state = 123)
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X,y_f, test_size = 0.2, random_state = 123)
def modelML(model, X_train, X_test,y_train, y_test, cc=1):

    if cc ==1:

        predict_cc = []

        model_cc = model.fit(X_train, y_train)

        predict_cc = model.predict(X_test)

        rmse_cc = np.sqrt(mean_squared_error(y_test, predict_cc))

        print("RMSE_CC = ", rmse_cc)

        return model_cc

    else:

        predict_f = []

        model_f = model.fit(X_train, y_train)

        predict_f = model_f.predict(X_test)

        rmse_f = np.sqrt(mean_squared_error(y_test, predict_f))

        print("RMSE_F = ", rmse_f)

        return model_f

        
# Linear Regression

#LR = LinearRegression()



#modelML(LR, X_train_cc, X_test_cc, y_train_cc,y_test_cc, cc=1)

#modelML(LR, X_train_f, X_test_f, y_train_f,y_test_f, cc=0)
# Decision Tree Regressor



#DTR = DecisionTreeRegressor()



#modelML(DTR, X_train_cc, X_test_cc, y_train_cc,y_test_cc, cc=1)

#modelML(DTR, X_train_f, X_test_f, y_train_f,y_test_f, cc=0)

# Random Forest regression



RFR = RandomForestRegressor()



#modelML(RFR, X_train_cc, X_test_cc, y_train_cc,y_test_cc, cc=1)

#modelML(RFR, X_train_f, X_test_f, y_train_f,y_test_f, cc=0)
# SVR 

#svr = SVR()

#modelML(svr, X_train_cc, X_test_cc, y_train_cc,y_test_cc, cc=1)

#modelML(svr, X_train_f, X_test_f, y_train_f,y_test_f, cc=0)
# SGDREGRESSOR



sgdr = SGDRegressor()

modelML(sgdr, X_train_cc, X_test_cc, y_train_cc,y_test_cc, cc=1)

modelML(sgdr, X_train_f, X_test_f, y_train_f,y_test_f, cc=0)
# Building model for final output



sgdr.fit(X,y_cc)

predict_cc_test = sgdr.predict(Xtest)
sgdr.fit(X,y_f)

predict_f_test = sgdr.predict(Xtest)
submission = pd.DataFrame({"ForecastId":Xtest.ForecastId,"ConfirmedCases": predict_cc_test,"Fatalities": predict_f_test})
submission.to_csv("submission.csv", index = False)