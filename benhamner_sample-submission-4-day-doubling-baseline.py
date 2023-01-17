# Read in the data



import numpy as np

import pandas as pd



train = pd.read_csv("../input/covid19-local-us-ca-forecasting-challenge-week-1/ca_train.csv")

test  = pd.read_csv("../input/covid19-local-us-ca-forecasting-challenge-week-1/ca_test.csv")
public_leaderboard_start_date = "2020-03-12"

last_public_leaderboard_train_date = "2020-03-11"

public_leaderboard_end_date  = "2020-03-26"



submission = test[["ForecastId"]]

submission.insert(1, "ConfirmedCases", 0)

submission.insert(2, "Fatalities", 0)



cases  = train[train["Date"]==last_public_leaderboard_train_date]["ConfirmedCases"].values[0] * (2**(1/4))

deaths = train[train["Date"]==last_public_leaderboard_train_date]["Fatalities"].values[0] * (2**(1/4))



for i in list(submission.index)[:15]:

    cases = cases * (2**(1/4))

    deaths = deaths * (2**(1/4))

    submission.loc[i, "ConfirmedCases"] = cases

    submission.loc[i, "Fatalities"] = deaths



submission
last_train_date = max(train["Date"])



cases  = train[train["Date"]==last_public_leaderboard_train_date]["ConfirmedCases"].values[0]

deaths = train[train["Date"]==last_public_leaderboard_train_date]["Fatalities"].values[0]



for i in submission.index:

    if test.loc[i, "Date"]>last_train_date: # Apply growth rule

        cases  = cases  * (2**(1/4))

        deaths = deaths * (2**(1/4))

    if test.loc[i, "Date"]>public_leaderboard_end_date: # Update submission value

        submission.loc[i, "ConfirmedCases"] = cases

        submission.loc[i, "Fatalities"] = deaths



submission
submission.to_csv("submission.csv", index=False)