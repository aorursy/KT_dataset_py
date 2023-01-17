from datetime import date, datetime, timedelta

import numpy as np

import pandas as pd



confirmed = pd.read_csv("../input/jhucovid19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv")

deaths   = pd.read_csv("../input/jhucovid19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv")
launch_date = date(2020, 4, 2)

latest_train_date = date(2020, 4, 1)



public_leaderboard_start_date = launch_date - timedelta(7)

close_date = launch_date + timedelta(7)

final_evaluation_start_date = launch_date + timedelta(8)
confirmed.columns = list(confirmed.columns[:4]) + [datetime.strptime(d, "%m/%d/%y").date().strftime("%Y-%m-%d") for d in confirmed.columns[4:]]

deaths.columns    = list(deaths.columns[:4])    + [datetime.strptime(d, "%m/%d/%y").date().strftime("%Y-%m-%d") for d in deaths.columns[4:]]
# Filter out problematic data points (The West Bank and Gaza had a negative value, cruise ships were associated with Canada, etc.)

removed_states = "Recovered|Grand Princess|Diamond Princess"

removed_countries = "US|The West Bank and Gaza"



confirmed.rename(columns={"Province/State": "Province_State", "Country/Region": "Country_Region"}, inplace=True)

deaths.rename(columns={"Province/State": "Province_State", "Country/Region": "Country_Region"}, inplace=True)

confirmed = confirmed[~confirmed["Province_State"].replace(np.nan, "nan").str.match(removed_states)]

deaths    = deaths[~deaths["Province_State"].replace(np.nan, "nan").str.match(removed_states)]

confirmed = confirmed[~confirmed["Country_Region"].replace(np.nan, "nan").str.match(removed_countries)]

deaths    = deaths[~deaths["Country_Region"].replace(np.nan, "nan").str.match(removed_countries)]



confirmed.drop(columns=["Lat", "Long"], inplace=True)

deaths.drop(columns=["Lat", "Long"], inplace=True)
confirmed
deaths
us_keys = pd.read_csv("../input/jhucovid19/csse_covid_19_data/csse_covid_19_daily_reports/04-01-2020.csv")

us_keys = us_keys[us_keys["Country_Region"]=="US"]

us_keys = us_keys.groupby(["Province_State", "Country_Region"])[["Confirmed", "Deaths"]].sum().reset_index()



us_keys = us_keys[~us_keys.Province_State.str.match("Diamond Princess|Grand Princess|Recovered|Northern Mariana Islands|American Samoa")].reset_index(drop=True)

us_keys
confirmed = confirmed.append(us_keys[["Province_State", "Country_Region"]], sort=False).reset_index(drop=True)

deaths = deaths.append(us_keys[["Province_State", "Country_Region"]], sort=False).reset_index(drop=True)
for col in confirmed.columns[2:]:

    confirmed[col].fillna(0, inplace=True)

    deaths[col].fillna(0, inplace=True)
confirmed
us_start_date = date(2020, 3, 10)

day_date = us_start_date



while day_date <= latest_train_date:

    day = pd.read_csv("../input/jhucovid19/csse_covid_19_data/csse_covid_19_daily_reports/%s.csv" % day_date.strftime("%m-%d-%Y"))

    

    if "Country/Region" in day.columns:

        day.rename(columns={"Country/Region": "Country_Region", "Province/State": "Province_State"}, inplace=True)

    

    us = day[day["Country_Region"]=="US"]

    us = us.groupby(["Province_State", "Country_Region"])[["Confirmed", "Deaths"]].sum().reset_index()

    

    unused_data = []

    untouched_states = set(confirmed[confirmed["Country_Region"]=="US"]["Province_State"])

    

    for (i, row) in us.iterrows():

        if confirmed[(confirmed["Country_Region"]=="US") & (confirmed["Province_State"]==row["Province_State"])].shape[0]==1:

            confirmed.loc[(confirmed["Country_Region"]=="US") & (confirmed["Province_State"]==row["Province_State"]), day_date.strftime("%Y-%m-%d")] = row["Confirmed"]

            deaths.loc[(deaths["Country_Region"]=="US") & (deaths["Province_State"]==row["Province_State"]), day_date.strftime("%Y-%m-%d")] = row["Deaths"]

            untouched_states.remove(row["Province_State"])

        else:

            unused_data.append(row["Province_State"])

            

    print(day_date, "Untouched", untouched_states)

    print(day_date, "Unused", unused_data)



    day_date = day_date + timedelta(1)
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
confirmed_melted = confirmed.melt(confirmed.columns[:2], confirmed.columns[2:], "Date", "ConfirmedCases")

#confirmed_melted.insert(5, "Type", "Confirmed")

deaths_melted = deaths.melt(deaths.columns[:2], deaths.columns[2:], "Date", "Fatalities")

#deaths_melted.insert(5, "Type", "Deaths")



confirmed_melted.sort_values(by=["Country_Region", "Province_State", "Date"], inplace=True)

deaths_melted.sort_values(by=["Country_Region", "Province_State", "Date"], inplace=True)



assert confirmed_melted.shape==deaths_melted.shape

assert list(confirmed_melted["Province_State"])==list(deaths_melted["Province_State"])

assert list(confirmed_melted["Country_Region"])==list(deaths_melted["Country_Region"])

assert list(confirmed_melted["Date"])==list(deaths_melted["Date"])



cases = confirmed_melted.merge(deaths_melted, on=["Province_State", "Country_Region", "Date"], how="inner")

cases = cases[["Country_Region", "Province_State", "Date", "ConfirmedCases", "Fatalities"]]



cases.sort_values(by=["Country_Region", "Province_State", "Date"], inplace=True)

cases.insert(0, "Id", range(1, cases.shape[0]+1))

cases
forecast = cases.loc[cases["Date"]>=public_leaderboard_start_date.strftime("%Y-%m-%d")]

forecast.drop(columns="Id", inplace=True)

forecast.insert(0, "ForecastId", range(1, forecast.shape[0]+1))

forecast.insert(6, "Usage", "Ignored")

forecast.loc[forecast["Date"]<launch_date.strftime("%Y-%m-%d"),"Usage"]="Public"

forecast.loc[forecast["Date"]>=final_evaluation_start_date.strftime("%Y-%m-%d"),"Usage"]="Private"

forecast
train = cases[cases["Date"]<launch_date.strftime("%Y-%m-%d")]

train.to_csv("train.csv", index=False)

train
test = forecast[forecast.columns[:-3]]

test.to_csv("test.csv", index=False)

test
solution = forecast[["ForecastId", "ConfirmedCases", "Fatalities", "Usage"]].copy()

solution["ConfirmedCases"].fillna(1, inplace=True)

solution["Fatalities"].fillna(1, inplace=True)

solution.to_csv("solution.csv", index=False)

solution
submission = forecast[["ForecastId", "ConfirmedCases", "Fatalities"]].copy()

submission["ConfirmedCases"] = 1

submission["Fatalities"] = 1

submission.to_csv("submission.csv", index=False)



submission