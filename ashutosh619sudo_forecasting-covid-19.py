# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/train.csv")
train_df.head()
india = train_df[train_df["Country_Region"]=="India"]
india.head()
india.tail()
plt.figure(figsize=(10,6))

plt.plot(india["ConfirmedCases"])

plt.xlabel("Time")

plt.ylabel("Number of Confirmed Cases")
plt.figure(figsize=(10,6))

plt.plot(india["Fatalities"])

plt.xlabel("Time")

plt.ylabel("Number of Fatalities")
china_states = train_df[train_df["Country_Region"]=="China"]["Province_State"].unique()
train_df["Province_State"].unique()
def province(state,country):

    if state == "nan":

        return country

    return state
train_df = train_df.fillna("nan")
train_df["Province_State"] = train_df.apply(lambda x: province(x["Province_State"],x["Country_Region"]),axis=1)
train_df
china_states
italy = train_df[train_df["Country_Region"]=="Italy"]
plt.figure(figsize=(10,6))

plt.plot(italy["ConfirmedCases"])

plt.xlabel("Time")

plt.ylabel("Number of Confirmed Cases")
plt.figure(figsize=(10,6))

plt.plot(italy["Fatalities"])

plt.xlabel("Time")

plt.ylabel("Number of Fatalities Cases")
from datetime import datetime
train_df.info()
train_df["Date"] = pd.to_datetime(train_df["Date"])
train_df["month"] = train_df["Date"].dt.month
train_df.head()
train_df['day'] = train_df['Date'].dt.day
train_df.tail()
train_df.drop('Date',axis=1,inplace=True)
train_df.head()
from sklearn import preprocessing
def labelencoder(data):

    le = preprocessing.LabelEncoder()

    new_data = le.fit_transform(data)

    return new_data
train_df["Country_Region"] = labelencoder(train_df["Country_Region"].values)

train_df["Province_State"] = labelencoder(train_df["Province_State"].values)
train_df.head()
test_df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/test.csv")
test_df["Date"] = pd.to_datetime(test_df["Date"])
test_df['day'] = test_df['Date'].dt.day
test_df["month"] = test_df["Date"].dt.month
test_df.drop('Date',axis=1,inplace=True)

test_df.head()
test_df.fillna("nan",inplace=True)

test_df["Province_State"] = test_df.apply(lambda x: province(x["Province_State"],x["Country_Region"]),axis=1)
test_df.head()
test_df["Country_Region"] = labelencoder(test_df["Country_Region"].values)

test_df["Province_State"] = labelencoder(test_df["Province_State"].values)
test_df.head()
countries = train_df["Country_Region"].unique()
from sklearn.preprocessing import PolynomialFeatures

#from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import MinMaxScaler
poly_reg_cc = PolynomialFeatures(degree = 4)

poly_reg_ft = PolynomialFeatures(degree = 4)



reg_cc = LinearRegression()

reg_ft = LinearRegression()



df_out = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})



for country in countries:

    states = train_df[train_df["Country_Region"]==country]["Province_State"].unique()

    for state in states:

        train_df_filt = train_df[(train_df["Country_Region"]==country)&(train_df["Province_State"]==state)]

        y_train_cc = train_df_filt["ConfirmedCases"].values

        y_train_ft = train_df_filt["Fatalities"].values

        

        

        X_train = train_df_filt[["month","day"]]

    

        

        test_df_filt = test_df[(test_df["Country_Region"]==country)&(test_df["Province_State"]==state)]

        X_test = test_df_filt.drop('ForecastId',axis=1)

        X_test = X_test[["month","day"]]

        test_Id = test_df_filt["ForecastId"].values

        

        scaler = MinMaxScaler()

        X_train = scaler.fit_transform(X_train)

        X_test = scaler.transform(X_test)

        

        

        scaler_cc = MinMaxScaler()

        scaler_ft = MinMaxScaler()

        

        y_train_cc=y_train_cc.reshape(-1,1)

        y_train_ft=y_train_ft.reshape(-1,1)

        

        y_train_cc=scaler_cc.fit_transform(y_train_cc)

        y_train_ft=scaler_ft.fit_transform(y_train_ft)

        

        y_train_cc = y_train_cc.flatten()

        y_train_ft = y_train_ft.flatten()

        

        X_train_poly = poly_reg_cc.fit_transform(X_train)

        reg_cc.fit(X_train_poly,y_train_cc)

        X_test_poly = poly_reg_cc.fit_transform(X_test)

        test_cc = reg_cc.predict(X_test_poly)

        

        test_cc = test_cc.reshape(-1,1)

        test_cc = scaler_cc.inverse_transform(test_cc)

        test_cc = test_cc.flatten()

        

        X_train_poly = poly_reg_ft.fit_transform(X_train)

        reg_ft.fit(X_train_poly,y_train_ft)

        X_test_poly = poly_reg_ft.fit_transform(X_test)

        test_ft = reg_ft.predict(X_test_poly)

        

        test_ft = test_ft.reshape(-1,1)

        

        test_ft = scaler_ft.inverse_transform(test_ft)

        test_ft = test_ft.flatten()

        

        df = pd.DataFrame({'ForecastId': test_Id, 'ConfirmedCases': test_cc, 'Fatalities': test_ft})

        

        df_out = pd.concat([df_out, df], axis=0)
df_out[:20]
df_out.head()
df_out["Fatalities"] = df_out["Fatalities"].apply(int)

df_out["ConfirmedCases"] = df_out["ConfirmedCases"].apply(int)

df_out[:10]
df_out["ForecastId"] = df_out["ForecastId"].astype('int32')

df_out["Fatalities"] = df_out["Fatalities"].astype('int32')

df_out["ConfirmedCases"] = df_out["ConfirmedCases"].astype('int32')

df_out.info()

df_out.to_csv("submission.csv",index=False)

sub = pd.read_csv("submission.csv")

sub[:20]
train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/train.csv")

test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/test.csv")



train["Province_State"] = train["Province_State"].fillna('')

test["Province_State"] = test["Province_State"].fillna('')



train["Month"], train["Day"] = 0, 0

for i in range(len(train)):

    train["Month"][i] = (train["Date"][i]).split("-")[1]

    train["Day"][i] = (train["Date"][i]).split("-")[2]

    

test["Month"], test["Day"] = 0, 0

for i in range(len(test)):

    test["Month"][i] = (test["Date"][i]).split("-")[1]

    test["Day"][i] = (test["Date"][i]).split("-")[2]
for i in range(len(train)):

    if train["Province_State"][i] != '':

        train["Country_Region"][i] = str(train["Province_State"][i]) + " (" + str(train["Country_Region"][i]) + ")"

       

for i in range(len(test)):

    if test["Province_State"][i] != '':

        test["Country_Region"][i] = str(test["Province_State"][i]) + " (" + str(test["Country_Region"][i]) + ")"

        

train.drop(columns = "Province_State", inplace=True)

test.drop(columns = "Province_State", inplace=True)



train.rename(columns = {"Country_Region" : "Country/State"}, inplace=True)

test.rename(columns = {"Country_Region" : "Country/State"}, inplace=True)

train.tail()
i = 0

for value in train["Country/State"].unique():

    if i < len(train):

        j = 1

        while(train["Country/State"][i] == value):

            train["Day"][i] = j

            j += 1; i += 1

            if i == len(train):

                break



i = 0

for value in test["Country/State"].unique():

    if i < len(test):

        j = 72

        while(test["Country/State"][i] == value):

            test["Day"][i] = j

            j += 1; i += 1

            if i == len(test):

                break
train = train.drop(columns = ["Date"])

test = test.drop(columns = ["Date"])
countries = train["Country/State"].unique()
from sklearn.preprocessing import PolynomialFeatures

poly_reg_cc = PolynomialFeatures(degree = 4)

poly_reg_ft = PolynomialFeatures(degree = 4)



from sklearn.linear_model import LinearRegression

reg_cc = LinearRegression()

reg_ft = LinearRegression()



from sklearn.preprocessing import StandardScaler



sub = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})

for value in countries:

    train_temp = train.loc[train["Country/State"] == value]

    test_temp = test.loc[test["Country/State"] == value]

    train_temp_cc = train_temp["ConfirmedCases"].loc[train["Country/State"] == value].to_frame()

    train_temp_ft = train_temp["Fatalities"].loc[train["Country/State"] == value].to_frame()

    

    train_temp_X = train_temp.iloc[:, 4:6]

    test_temp_X = test_temp.iloc[:, 2:4]

    sc1 = StandardScaler()

    train_temp_X = sc1.fit_transform(train_temp_X)

    test_temp_X = sc1.transform(test_temp_X)

    

    sc_cc = StandardScaler()

    sc_ft = StandardScaler()

    train_temp_cc = sc_cc.fit_transform(train_temp_cc)

    train_temp_ft = sc_ft.fit_transform(train_temp_ft)

    

    X_poly = poly_reg_cc.fit_transform(train_temp_X)

    reg_cc.fit(X_poly, train_temp_cc)

    test_cc = sc_cc.inverse_transform(reg_cc.predict(poly_reg_cc.fit_transform(test_temp_X)))

    

    X_poly = poly_reg_ft.fit_transform(train_temp_X)

    reg_ft.fit(X_poly, train_temp_ft)

    test_ft = sc_ft.inverse_transform(reg_ft.predict(poly_reg_ft.fit_transform(test_temp_X)))

    

    a = int(train["Day"].loc[train["Country/State"] == "India"].max())

    b = int(a - test_temp["Day"].min())

    

    test_cc[0:b+1] = sc_cc.inverse_transform(train_temp_cc)[(a-b-1):(a)]

    test_ft[0:b+1] = sc_ft.inverse_transform(train_temp_ft)[(a-b-1):(a)]

    

    test_cc = test_cc.flatten()

    test_ft = test_ft.flatten()

    sub_temp = pd.DataFrame({'ForecastId': test_temp["ForecastId"].loc[test["Country/State"] == value],

                             'ConfirmedCases': test_cc, 'Fatalities': test_ft})

    sub = pd.concat([sub, sub_temp], axis = 0)
sub["ForecastId"] = sub["ForecastId"].astype('int32')
sub[:20]
sub.to_csv("submission.csv", index = False)