from datetime import date, datetime, timedelta

import numpy as np

import pandas as pd



confirmed = pd.read_csv("../input/jhucovid19/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv")

deaths   = pd.read_csv("../input/jhucovid19/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv")
launch_date = date(2020, 3, 19)



public_leaderboard_start_date = launch_date - timedelta(7)

close_date = launch_date + timedelta(7)

final_evaluation_start_date = launch_date + timedelta(8)

final_evaluation_end_date = launch_date + timedelta(36)



final_evaluation_end_date
confirmed.columns = list(confirmed.columns[:4]) + [datetime.strptime(d, "%m/%d/%y").date().strftime("%Y-%m-%d") for d in confirmed.columns[4:]]

deaths.columns    = list(deaths.columns[:4])    + [datetime.strptime(d, "%m/%d/%y").date().strftime("%Y-%m-%d") for d in deaths.columns[4:]]
# Filter out cities, which seem to have worse data

confirmed = confirmed[((confirmed["Province/State"].isna()==True) | (confirmed["Province/State"].str.contains(",")==False))]

deaths    = deaths[((deaths["Province/State"].isna()==True) | (deaths["Province/State"].str.contains(",")==False))]
confirmed
deaths
dates_on_after_launch = [col for col in confirmed.columns[4:] if col>=launch_date.strftime("%Y-%m-%d")]

print("Removing %d columns: %s" % (len(dates_on_after_launch), str(dates_on_after_launch)))



cols_to_keep = [col for col in confirmed.columns if col not in dates_on_after_launch]



confirmed = confirmed[cols_to_keep]

deaths = deaths[cols_to_keep]
for i in range(36):

    this_date = (launch_date + timedelta(i)).strftime("%Y-%m-%d")

    confirmed.insert(len(confirmed.columns), this_date, np.NaN)

    deaths.insert(len(deaths.columns), this_date, np.NaN)
confirmed_melted = confirmed.melt(confirmed.columns[:4], confirmed.columns[4:], "Date", "ConfirmedCases")

#confirmed_melted.insert(5, "Type", "Confirmed")

deaths_melted = deaths.melt(deaths.columns[:4], deaths.columns[4:], "Date", "Fatalities")

#deaths_melted.insert(5, "Type", "Deaths")



confirmed_melted.sort_values(by=["Country/Region", "Province/State", "Date"], inplace=True)

deaths_melted.sort_values(by=["Country/Region", "Province/State", "Date"], inplace=True)



assert confirmed_melted.shape==deaths_melted.shape

assert list(confirmed_melted["Province/State"])==list(deaths_melted["Province/State"])

assert list(confirmed_melted["Country/Region"])==list(deaths_melted["Country/Region"])

assert list(confirmed_melted["Date"])==list(deaths_melted["Date"])



cases = confirmed_melted.merge(deaths_melted, on=["Province/State", "Country/Region", "Date", "Lat", "Long"], how="inner")



cases.sort_values(by=["Country/Region", "Province/State", "Date"], inplace=True)

cases.insert(0, "Id", range(1, cases.shape[0]+1))

cases
forecast = cases[cases["Date"]>=public_leaderboard_start_date.strftime("%Y-%m-%d")]

forecast.drop(columns="Id", inplace=True)

forecast.insert(0, "ForecastId", range(1, forecast.shape[0]+1))

forecast.insert(8, "Usage", "Ignored")

forecast.loc[forecast["Date"]<launch_date.strftime("%Y-%m-%d"),"Usage"]="Public"

forecast.loc[forecast["Date"]>=final_evaluation_start_date.strftime("%Y-%m-%d"),"Usage"]="Private"

forecast
train = cases[cases["Date"]<launch_date.strftime("%Y-%m-%d")]

train.to_csv("train.csv", index=False)

train
test = forecast[forecast.columns[:-3]]

test.to_csv("test.csv", index=False)

test
solution = forecast[["ForecastId", "ConfirmedCases", "Fatalities", "Usage"]]

solution["ConfirmedCases"].fillna(1, inplace=True)

solution["Fatalities"].fillna(1, inplace=True)

solution.to_csv("solution.csv", index=False)

solution
submission = forecast[["ForecastId", "ConfirmedCases", "Fatalities"]]

submission["ConfirmedCases"] = 1

submission["Fatalities"] = 1

submission.to_csv("submission.csv", index=False)



submission
## California competition data
ca_cases = cases[(cases["Country/Region"]=="US") & (cases["Province/State"]=="California")]

ca_cases["Id"] = range(1, ca_cases.shape[0]+1)

ca_train = ca_cases[ca_cases["Date"]<launch_date.strftime("%Y-%m-%d")]

ca_train.to_csv("ca_train.csv", index=False)

ca_train
ca_forecast = forecast[(forecast["Country/Region"]=="US") & (forecast["Province/State"]=="California")]

ca_forecast["ForecastId"] = range(1, ca_forecast.shape[0]+1)

ca_forecast
ca_test = ca_forecast[ca_forecast.columns[:-3]]

ca_test.to_csv("ca_test.csv", index=False)

ca_test
ca_solution = ca_forecast[["ForecastId", "ConfirmedCases", "Fatalities", "Usage"]]

ca_solution["ConfirmedCases"].fillna(1, inplace=True)

ca_solution["Fatalities"].fillna(1, inplace=True)

ca_solution.to_csv("ca_solution.csv", index=False)

ca_solution
ca_submission = ca_forecast[["ForecastId", "ConfirmedCases", "Fatalities"]]

ca_submission["ConfirmedCases"] = 1

ca_submission["Fatalities"] = 1

ca_submission.to_csv("ca_submission.csv", index=False)

ca_submission