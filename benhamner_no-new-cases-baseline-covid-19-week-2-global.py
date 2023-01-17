# Read in the data



import numpy as np

import pandas as pd



train = pd.read_csv("../input/covid19-global-forecasting-week-2/train.csv")

test  = pd.read_csv("../input/covid19-global-forecasting-week-2/test.csv")
submission = test[["ForecastId"]]

submission.insert(1, "ConfirmedCases", 0)

submission.insert(2, "Fatalities", 0)
locations = list(set([(test.loc[i, "Province_State"], test.loc[i, "Country_Region"]) for i in test.index]))

locations
len(locations)
public_leaderboard_start_date = "2020-03-12"

last_public_leaderboard_train_date = "2020-03-11"

public_leaderboard_end_date  = "2020-03-26"



for loc in locations:

    if type(loc[0]) is float and np.isnan(loc[0]):

        confirmed=train[((train["Country_Region"]==loc[1]) & (train["Date"]==last_public_leaderboard_train_date))]["ConfirmedCases"].values[0]

        deaths=train[((train["Country_Region"]==loc[1]) & (train["Date"]==last_public_leaderboard_train_date))]["Fatalities"].values[0]

        submission.loc[((test["Country_Region"]==loc[1]) & (test["Date"]<=public_leaderboard_end_date)), "ConfirmedCases"] = confirmed

        submission.loc[((test["Country_Region"]==loc[1]) & (test["Date"]<=public_leaderboard_end_date)), "Fatalities"] = deaths

    else:

        confirmed=train[((train["Province_State"]==loc[0]) & (train["Country_Region"]==loc[1]) & (train["Date"]==last_public_leaderboard_train_date))]["ConfirmedCases"].values[0]

        deaths=train[((train["Province_State"]==loc[0]) & (train["Country_Region"]==loc[1]) & (train["Date"]==last_public_leaderboard_train_date))]["Fatalities"].values[0]

        submission.loc[((test["Country_Region"]==loc[1]) & (test["Date"]<=public_leaderboard_end_date)), "ConfirmedCases"] = confirmed

        submission.loc[((test["Country_Region"]==loc[1]) & (test["Date"]<=public_leaderboard_end_date)), "Fatalities"] = deaths



submission
last_train_date = max(train["Date"])



for loc in locations:

    if type(loc[0]) is float and np.isnan(loc[0]):

        confirmed=train[((train["Country_Region"]==loc[1]) & (train["Date"]==last_train_date))]["ConfirmedCases"].values[0]

        deaths=train[((train["Country_Region"]==loc[1]) & (train["Date"]==last_train_date))]["Fatalities"].values[0]

        submission.loc[((test["Country_Region"]==loc[1]) & (test["Date"]>public_leaderboard_end_date)), "ConfirmedCases"] = confirmed

        submission.loc[((test["Country_Region"]==loc[1]) & (test["Date"]>public_leaderboard_end_date)), "Fatalities"] = deaths

    else:

        confirmed=train[((train["Province_State"]==loc[0]) & (train["Country_Region"]==loc[1]) & (train["Date"]==last_train_date))]["ConfirmedCases"].values[0]

        deaths=train[((train["Province_State"]==loc[0]) & (train["Country_Region"]==loc[1]) & (train["Date"]==last_train_date))]["Fatalities"].values[0]

        submission.loc[((test["Country_Region"]==loc[1]) & (test["Date"]>public_leaderboard_end_date)), "ConfirmedCases"] = confirmed

        submission.loc[((test["Country_Region"]==loc[1]) & (test["Date"]>public_leaderboard_end_date)), "Fatalities"] = deaths



submission
submission.to_csv("submission.csv", index=False)