import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import os

from scipy.optimize import curve_fit

from sklearn.linear_model import LinearRegression

import time

os.listdir()
train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/train.csv")

test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/test.csv")

submission = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/submission.csv")

submit = test.copy()
submit["ConfirmedCases"] = 0

submit["Fatalities"] = 0

submit.head()
train.head()
train.ConfirmedCases.isna().sum(), train.Fatalities.isna().sum()
train.describe()
test.head() # dates to predict
submission.head() # sample submission
train.Country_Region.unique()
plt.plot(train.loc[train.Country_Region == "South Africa"].ConfirmedCases)
train.Date = pd.to_datetime(train.Date)

test.Date = pd.to_datetime(test.Date)

submit.Date = pd.to_datetime(submit.Date)
train.Date
(train.Date[1] - train.Date[0]).days # now we can do this
train["Day"] = train.Date.map(lambda date: (date - train.Date[0]).days)

test["Day"] = test.Date.map(lambda date: (date - train.Date[0]).days)

submit["Day"] = submit.Date.map(lambda date: (date - train.Date[0]).days)
def logit_func(x, L, k, x0):

    y = L/(1 + np.exp(-k*(x - x0)))

    return y
SA_xdata = train.loc[train.Country_Region == 'South Africa'].Day

SA_ydata = train.loc[train.Country_Region == 'South Africa'].ConfirmedCases

#SA_xdata
popt, pcov = curve_fit(logit_func, SA_xdata, SA_ydata, bounds = (0, np.inf))
popt, pcov
p_guess = popt # use this for later guesses 

p_guess
plt.plot(SA_xdata, SA_ydata, 'o')

plt.plot(SA_xdata, logit_func(SA_xdata, *popt), "--")

plt.plot(test.loc[test.Country_Region == "South Africa"].Day, logit_func(test.loc[test.Country_Region == "South Africa"].Day , *popt)) # predict and then check out 5 values)
time0 = time.time()

Country = "Italy"

xdata = train.loc[train.Country_Region == Country].Day#.values[..., np.newaxis]

ydata = train.loc[train.Country_Region == Country].ConfirmedCases#.values[..., np.newaxis]



popt, pcov = curve_fit(logit_func, xdata, ydata, bounds = (0, np.inf), maxfev=1e6)      # fit     p0=p_guess, 

predict = logit_func(submit.loc[submit.Country_Region == Country].Day , *popt) # predict

time1 = time.time()

print(f"elapsed_time = {time1-time0}")

p_guess = popt
p_guess
plt.plot(xdata, ydata, 'o')

plt.plot(xdata, logit_func(xdata, *popt), "--")

plt.plot(test.loc[test.Country_Region == Country].Day, logit_func(test.loc[test.Country_Region == Country].Day , *popt)) # predict and then check out 5 values)
for Country in test.Country_Region.unique():

    if train.loc[train["Country_Region"] == Country].Province_State.isna().sum() > 0:

        # grab data

        xdata = train.loc[train.Country_Region == Country].Day#.values[..., np.newaxis]

        ydata = train.loc[train.Country_Region == Country].ConfirmedCases#.values[..., np.newaxis]

        

        # fit model and predict values

        try:

            popt, pcov = curve_fit(logit_func, xdata, ydata, bounds = (0, np.inf), maxfev=1e6)      # fit     

            predict = logit_func(submit.loc[submit.Country_Region == Country].Day , *popt) # predict

           

            

        

        except: # for now lets just do linear regression for this exception

            xdata = xdata.values[..., np.newaxis]

            ydata = ydata.values[..., np.newaxis] # something silly happening here

            print(f"Linear regression for {Country}")

            reg = LinearRegression().fit(xdata, ydata) # fit

            predict = reg.predict(submit.loc[submit.Country_Region == Country].Day.values[..., np.newaxis])

            

        submit.loc[submit.Country_Region == Country, "ConfirmedCases"] = predict # set cases       



    else: # provinces exist

        for Province in test.loc[test.Country_Region == Country, "Province_State"].unique():

            xdata = train.loc[(train.Country_Region == Country) & (train.Province_State == Province)].Day

            ydata = train.loc[(train.Country_Region == Country) & (train.Province_State == Province)].ConfirmedCases



            # fit model and predict values

            try:

                popt, pcov = curve_fit(logit_func, xdata, ydata, bounds = (0, np.inf), maxfev=1e6)      # fit     

                predict = logit_func(submit.loc[(submit.Country_Region == Country) & (submit.Province_State == Province)].Day , *popt) # predict







            except: # for now lets just do linear regression for this exception

                xdata = xdata.values[..., np.newaxis]

                ydata = ydata.values[..., np.newaxis] # something silly happening here

                print(f"Linear regression for {Country} {Province}")

                reg = LinearRegression().fit(xdata, ydata) # fit

                predict = reg.predict(submit.loc[(submit.Country_Region == Country) & (submit.Province_State == Province)].Day.values[..., np.newaxis])



            submit.loc[(submit.Country_Region == Country) & (submit.Province_State == Province), "ConfirmedCases"] = predict # set cases              
submit.head()
for Country in test.Country_Region.unique():

    if train.loc[train["Country_Region"] == Country].Province_State.isna().sum() > 0:

        # grab data

        xdata = train.loc[train.Country_Region == Country].Day

        ydata = train.loc[train.Country_Region == Country].Fatalities

        

                # fit model and predict values

        try:

            popt, pcov = curve_fit(logit_func, xdata, ydata, bounds = (0, np.inf), maxfev=1e6)      # fit     

            predict = logit_func(submit.loc[submit.Country_Region == Country].Day , *popt) # predict

           

            

        

        except: # for now lets just do linear regression for this exception

            xdata = xdata.values[..., np.newaxis]

            ydata = ydata.values[..., np.newaxis] # something silly happening here

            print(f"Linear regression for {Country}")

            reg = LinearRegression().fit(xdata, ydata) # fit

            predict = reg.predict(submit.loc[submit.Country_Region == Country].Day.values[..., np.newaxis])

        

        submit.loc[submit.Country_Region == Country, "Fatalities"] = predict # set cases       



    else:

        for Province in test.loc[test.Country_Region == Country, "Province_State"].unique():

            xdata = train.loc[(train.Country_Region == Country) & (train.Province_State == Province)].Day

            ydata = train.loc[(train.Country_Region == Country) & (train.Province_State == Province)].Fatalities



            # fit model and predict values

            try:

                popt, pcov = curve_fit(logit_func, xdata, ydata, bounds = (0, np.inf),maxfev=1e6)      # fit     

                predict = logit_func(submit.loc[(submit.Country_Region == Country) & (submit.Province_State == Province)].Day , *popt) # predict







            except: # for now lets just do linear regression for this exception

                xdata = xdata.values[..., np.newaxis]

                ydata = ydata.values[..., np.newaxis] # something silly happening here

                print(f"Linear regression for {Country} {Province}")

                reg = LinearRegression().fit(xdata, ydata) # fit

                predict = reg.predict(submit.loc[(submit.Country_Region == Country) & (submit.Province_State == Province)].Day.values[..., np.newaxis])





            submit.loc[(submit.Country_Region == Country) & (submit.Province_State == Province), "Fatalities"] = predict # set cases              
submit.ConfirmedCases = submit.ConfirmedCases.astype('int')

submit.Fatalities = submit.Fatalities.astype('int')
def comp_func(Country, cases=True):

    if cases is True:

        plt.plot(train.loc[train.Country_Region == Country].Day, train.loc[train.Country_Region == Country].ConfirmedCases, "o")

        plt.plot(submit.loc[submit.Country_Region == Country].Day, submit.loc[submit.Country_Region == Country].ConfirmedCases, "--")

    else:

        plt.plot(train.loc[train.Country_Region == Country].Day, train.loc[train.Country_Region == Country].Fatalities , 'o')

        plt.plot(submit.loc[submit.Country_Region == Country].Day, submit.loc[submit.Country_Region == Country].Fatalities, '--')
comp_func("Italy")
comp_func("South Africa")
comp_func("Afghanistan")
submit.head()
submit.describe()
submission = submit.loc[:, ["ForecastId", "ConfirmedCases", "Fatalities"]]

submission.to_csv('./submission.csv', index = False)
submission.head()