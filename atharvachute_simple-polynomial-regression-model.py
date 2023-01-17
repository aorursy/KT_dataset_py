import numpy as np

import pandas as pd

import warnings

warnings.filterwarnings("ignore")



train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/train.csv")

test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/test.csv")
train.info()

test.info()
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

        train["Country_Region"][i] = train["Province_State"][i] + " (" + str(train["Country_Region"][i]) + ")"

       

for i in range(len(test)):

    if test["Province_State"][i] != '':

        test["Country_Region"][i] = test["Province_State"][i] + " (" + str(test["Country_Region"][i]) + ")"

        

train.drop(columns = "Province_State", inplace=True)

test.drop(columns = "Province_State", inplace=True)



train.rename(columns = {"Country_Region" : "Country/State"}, inplace=True)

test.rename(columns = {"Country_Region" : "Country/State"}, inplace=True)

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

                

train.rename(columns = {"Day" : "No. of days"}, inplace = True)

test.rename(columns = {"Day" : "No. of days"}, inplace = True)
train["Country/State"].loc[train["Country/State"] == "Taiwan*"] = "Taiwan"

test["Country/State"].loc[test["Country/State"] == "Taiwan*"] = "Taiwan"



train = train.drop(columns = ["Date"])

test = test.drop(columns = ["Date"])



countriesorstates = train["Country/State"].unique()

print(countriesorstates)
import matplotlib.pyplot as plt

import random

random_picks = random.choices(countriesorstates, k = 9)

for value in random_picks:

    train_temp = train.loc[train["Country/State"] == value]

    test_temp = test.loc[test["Country/State"] == value]

    train_temp_cc = train_temp["ConfirmedCases"].loc[train["Country/State"] == value]

    train_temp_ft = train_temp["Fatalities"].loc[train["Country/State"] == value]

    

    train_temp_X = train_temp.iloc[:, 5]

    test_temp_X = test_temp.iloc[:, 3]

    

    x = train_temp_X.to_numpy(dtype = float)

    y_1 = train_temp_cc.to_numpy(dtype = float)

    y_2= train_temp_ft.to_numpy(dtype = float)

    x = x.reshape(-1,)

    

    plt.plot(x, y_1, color ='red', label = value)

    plt.xlabel("No. of days")

    plt.ylabel('Confirmed cases')

    plt.legend()

    plt.title("Place: " + value)

    plt.show()

    

    plt.plot(x, y_2, color ='black', label = value)

    plt.xlabel("No. of days")

    plt.ylabel('Fatalities')

    plt.legend()

    plt.title("Place: " + value)

    plt.show()
from sklearn.preprocessing import PolynomialFeatures

poly_reg_cc = PolynomialFeatures(degree = 4)

poly_reg_ft = PolynomialFeatures(degree = 4)



from sklearn.linear_model import LinearRegression

reg_cc = LinearRegression()

reg_ft = LinearRegression()



from sklearn.preprocessing import StandardScaler



sub = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})

for value in countriesorstates:

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

    

    a = int(train["No. of days"].loc[train["Country/State"] == "India"].max())

    b = int(a - test_temp["No. of days"].min())

    

    test_cc[0:b+1] = sc_cc.inverse_transform(train_temp_cc)[(a-b-1):(a)]

    test_ft[0:b+1] = sc_ft.inverse_transform(train_temp_ft)[(a-b-1):(a)]

    

    test_cc = test_cc.flatten()

    test_ft = test_ft.flatten()

    sub_temp = pd.DataFrame({'ForecastId': test_temp["ForecastId"].loc[test["Country/State"] == value],

                             'ConfirmedCases': test_cc, 'Fatalities': test_ft})

    sub = pd.concat([sub, sub_temp], axis = 0)
sub.ForecastId = sub.ForecastId.astype('int')

for i in range(len(sub)):

    sub["ConfirmedCases"][i] = int(round(sub["ConfirmedCases"][i]))

    sub["Fatalities"][i] = int(round(sub["Fatalities"][i]))



sub.to_csv("submission.csv", index = False)