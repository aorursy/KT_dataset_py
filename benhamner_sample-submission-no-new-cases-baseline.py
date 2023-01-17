# Read in the data



import pandas as pd



train = pd.read_csv("../input/covid19-local-us-ca-forecasting-challenge-week-1/ca_train.csv")

test  = pd.read_csv("../input/covid19-local-us-ca-forecasting-challenge-week-1/ca_test.csv")
public_leaderboard_start_date = "2020-03-12"

last_public_leaderboard_train_date = "2020-03-11"

public_leaderboard_end_date  = "2020-03-26"



submission = test[["ForecastId"]]

submission.insert(1, "ConfirmedCases", 0)

submission.insert(2, "Fatalities", 0)



submission.loc[test["Date"]<=public_leaderboard_end_date, "ConfirmedCases"] = train[train["Date"]==last_public_leaderboard_train_date]["ConfirmedCases"].values[0]

submission.loc[test["Date"]<=public_leaderboard_end_date, "Fatalities"] = train[train["Date"]==last_public_leaderboard_train_date]["Fatalities"].values[0]

submission
last_train_date = max(train["Date"])



submission.loc[test["Date"]>public_leaderboard_end_date, "ConfirmedCases"] = train[train["Date"]==last_train_date]["ConfirmedCases"].values[0]

submission.loc[test["Date"]>public_leaderboard_end_date, "Fatalities"] = train[train["Date"]==last_train_date]["Fatalities"].values[0]

submission
submission.to_csv("submission.csv", index=False)