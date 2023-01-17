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
train_df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/train.csv")

test_df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/test.csv")
train_df.head(5)
# check if any col is null

train_df.apply(lambda col: col.isnull().value_counts(), axis=0)
test_df.apply(lambda col: col.isna().value_counts(), axis=0)
# fill na

train_df["Province_State"] = train_df["Province_State"].fillna("")

test_df["Province_State"] = test_df["Province_State"].fillna("")
train_df["Date"] = pd.to_datetime(train_df["Date"])

test_df["Date"] = pd.to_datetime(test_df["Date"])
train_df["ConfirmedCases"] = np.log(train_df["ConfirmedCases"] + 1)

train_df["Fatalities"] = np.log(train_df["Fatalities"] + 1)
!pip install fbprophet
from fbprophet import Prophet
'''

country = "US"

state = "New York"

cap = 15

region_df = train_df[(train_df["Country_Region"] == country) & (train_df["Province_State"] == state)]

region_df = region_df[["Date","ConfirmedCases"]].rename(columns={"Date":"ds","ConfirmedCases":"y"})

region_df['cap'] = cap

test_region_df = test_df[(test_df["Country_Region"]==country) & (test_df["Province_State"]==state)][["ForecastId", "Date"]]

test_region_df.rename(columns={"Date":"ds"}, inplace=True)

test_region_df['cap'] = cap

'''
'''

case_model = Prophet(growth='logistic')

case_model.fit(region_df[-30:])

'''
'''

forecast = case_model.predict(test_region_df)

fig1 = case_model.plot(forecast)

'''
submission_df = []

cap = 15

for (country, state), region_df in train_df.groupby(["Country_Region","Province_State"]):

    # confirm cases model

    print("Fit confirmed case model for {}, {}".format(state, country))

    case_df = region_df[["Date","ConfirmedCases"]].rename(columns={"Date":"ds","ConfirmedCases":"y"})

    case_df['cap'] = cap

    case_model = Prophet(growth='logistic')

    case_model.fit(case_df)

    

    # fatalities model

    print("Fit fatality model for {}, {}".format(state, country))

    fatality_df = region_df[["Date","Fatalities"]].rename(columns={"Date":"ds","Fatalities":"y"})

    fatality_df['cap'] = cap

    fatality_model = Prophet(growth='logistic')

    fatality_model.fit(fatality_df)

    

    test_region_df = test_df[(test_df["Country_Region"]==country) & (test_df["Province_State"]==state)][["ForecastId", "Date"]]

    test_region_df.rename(columns={"Date":"ds"}, inplace=True)

    test_region_df['cap'] = cap

    

    print("Predict confirmed cases for {}, {}".format(state, country))

    cases_predict = case_model.predict(test_region_df[["ds","cap"]])[["ds", "yhat"]]

    test_region_df = test_region_df.merge(cases_predict, how="left", on="ds").rename(columns={"yhat": "ConfirmedCases"})

    

    print("Predict fatalities for {}, {}".format(state, country))

    fatality_predict = fatality_model.predict(test_region_df[["ds","cap"]])[["ds", "yhat"]]

    test_region_df = test_region_df.merge(fatality_predict, how="left", on="ds").rename(columns={"yhat": "Fatalities"})

    

    test_region_df.drop(columns=["ds"], inplace=True)

    

    test_region_df["ConfirmedCases"] = np.exp(test_region_df["ConfirmedCases"]) - 1

    test_region_df["Fatalities"] = np.exp(test_region_df["Fatalities"]) - 1

    

    submission_df.append(test_region_df)

submission_df = pd.concat(submission_df)

submission_df.drop(columns=["cap"], inplace=True)

submission_df
submission_df.to_csv("submission.csv", index=False)