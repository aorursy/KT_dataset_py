import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import random as r

from datetime import datetime

from scipy import stats
data_2016 = pd.read_csv("../input/2016-election-polls/presidential_polls.csv")

data_2016 = data_2016[["startdate", "enddate", "state", "pollster", "grade", "samplesize", "population", "adjpoll_clinton", "adjpoll_trump"]]

trump_clinton = data_2016.rename(columns = {"startdate": "start_date", "enddate": "end_date", "grade":"fte_grade", "samplesize":"sample_size", "adjpoll_clinton":"Clinton", "adjpoll_trump":"Trump"})

trump_clinton["start_date"] = pd.to_datetime(trump_clinton["start_date"])

trump_clinton["end_date"] = pd.to_datetime(trump_clinton["end_date"])

trump_clinton = trump_clinton.sort_values(by = ["end_date", "start_date"]) #Arranging the polls from most to least recent

trump_clinton["dem_lead"] = trump_clinton["Clinton"] - trump_clinton["Trump"] #lead of the democratic candidate (negative if they are losing)
trump_clinton.head()
trump_clinton.info()
data_2020 = pd.read_csv("../input/2020-general-election-polls/president_polls.csv")

data_2020 = data_2020[["poll_id","start_date", "end_date", "state", "pollster", "fte_grade", "sample_size", "population", "answer", "pct"]]
data_2020.head()
data_2020.info()
data_2020["state"] = data_2020.state.fillna("U.S.")
def trump_opponent(data_2020, opp):

    trump_vs = data_2020[(data_2020["answer"] == opp) | (data_2020["answer"] == "Trump")]

    trump_vs = trump_vs.pivot_table(values = "pct", index = ["poll_id", "start_date", "end_date", "state", "pollster", "fte_grade", "sample_size", "population"], columns = "answer")

    trump_vs = trump_vs.dropna(axis = 0, how = "any") #Drops the Trump polls against any opponent that isn't our opp parameter

    trump_vs = trump_vs.reset_index().drop(columns = ["poll_id"])

    trump_vs["start_date"] = pd.to_datetime(trump_vs["start_date"])

    trump_vs["end_date"] = pd.to_datetime(trump_vs["end_date"]) 

    trump_vs["dem_lead"] = trump_vs[opp] - trump_vs["Trump"] 

    trump_vs = trump_vs.sort_values(by = ["end_date", "start_date"]) #Arranging the polls from most to least recent

    return trump_vs
trump_biden = trump_opponent(data_2020, "Biden")



trump_biden.head()
trump_sanders = trump_opponent(data_2020, "Sanders")



trump_sanders.head()
trump_warren = trump_opponent(data_2020, "Warren")



trump_warren.head()
trump_buttigieg = trump_opponent(data_2020, "Buttigieg")



trump_buttigieg.head()
results_2016 = pd.read_csv("../input/2020-general-election-polls/nytimes_presidential_elections_2016_results_county.csv")

results_2016 = results_2016.groupby("State").sum()[["Clinton", "Trump"]]

results_2016.loc["U.S."] = [65853514, 62984828] #Adding a row for the national result

results_2016["Clinton_pct"] = 100 * results_2016["Clinton"] / (results_2016["Clinton"] + results_2016["Trump"])

results_2016["Trump_pct"] = 100 * results_2016["Trump"] / (results_2016["Clinton"] + results_2016["Trump"]) #percentages

results_2016["dem_lead"] = results_2016["Clinton_pct"] - results_2016["Trump_pct"]

results_2016["index"] = list(range(0,50))

results_2016["state"] = results_2016.index

results_2016 = results_2016.set_index("index")
results_2016.head()
results_2016.info()
def trump_vs_clinton(trump_clinton, state, results_2016, reliable = False, likely_voters = False):

    

    #getting polls for the specified state / U.S. and filtering if necessary

    match_up = trump_clinton

    match_up = match_up[match_up["state"] == state]

    

    if reliable == True:

        match_up = match_up[match_up["fte_grade"].isin(["A+", "A", "A-"])]

    if likely_voters == True:

        match_up = match_up[match_up["population"] == "lv"]

    

    #Accounting for repeated polls which have the same end date

    

    match_up = match_up.groupby(["end_date", "pollster", "fte_grade", "population"]).mean().reset_index()

    match_up.index = match_up["end_date"]

    

    #A rolling average of democrat lead/deficit in the past 14 days

    

    if state == "U.S.":

        match_up["average_lead"] = match_up["dem_lead"].rolling("14D", min_periods = 0).mean()

    else:

        match_up["average_lead"] = match_up["dem_lead"].rolling("30D", min_periods = 0).mean()

    

    #Plotting the time series

    

    polls_vs_final =  [match_up.iloc[-1]["average_lead"], results_2016[results_2016["state"] == state].iloc[0]["dem_lead"]]

    polls_df = pd.DataFrame(polls_vs_final)

    polls_df[1] = ["Final/Current Polling Average", "Actual Results"]

    

    plt.subplots(figsize = (9,6))

    plt.subplot(1,2,1)

    plt.plot(match_up["end_date"], match_up["average_lead"])

    plt.xlabel("Date")

    plt.xticks(rotation = 90)

    plt.ylabel("Lead or Deficit vs. Trump (%)")

    plt.title(f"Trump vs. Clinton in {state}")  

    plt.subplot(1,2,2)

    plt.bar(polls_df[1], polls_df[0])

    plt.xticks(rotation = 90)

    plt.title("Poll Accuracy Chart")

    plt.show()

    

    if reliable == True:

        return f"Percentage Points of Trump Underestimation in {state} From Historically Reliable Polls: {round(polls_vs_final[0] - polls_vs_final[1], 2)}%"

    if likely_voters == True:

        return f"Percentage Points of Trump Underestimation in {state} From Polls of Likely Voters: {round(polls_vs_final[0] - polls_vs_final[1], 2)}%"

    return f"Percentage Points of Trump Underestimation in {state} From All Polls: {round(polls_vs_final[0] - polls_vs_final[1], 2)}%"
trump_vs_clinton(trump_clinton, "U.S.", results_2016)
trump_vs_clinton(trump_clinton, "U.S.", results_2016, reliable = True)
trump_vs_clinton(trump_clinton, "U.S.", results_2016, likely_voters = True)
trump_vs_clinton(trump_clinton, "Florida", results_2016)
trump_vs_clinton(trump_clinton, "Florida", results_2016, reliable = True)
trump_vs_clinton(trump_clinton, "Florida", results_2016, likely_voters = True)
trump_vs_clinton(trump_clinton, "Pennsylvania", results_2016)
trump_vs_clinton(trump_clinton, "Pennsylvania", results_2016, reliable = True)
trump_vs_clinton(trump_clinton, "Pennsylvania", results_2016, likely_voters = True)
trump_vs_clinton(trump_clinton, "Ohio", results_2016)
trump_vs_clinton(trump_clinton, "Ohio", results_2016, reliable = True)
trump_vs_clinton(trump_clinton, "Ohio", results_2016, likely_voters = True)
trump_vs_clinton(trump_clinton, "Michigan", results_2016)
trump_vs_clinton(trump_clinton, "Michigan", results_2016, reliable = True)
trump_vs_clinton(trump_clinton, "Michigan", results_2016, likely_voters = True)
trump_vs_clinton(trump_clinton, "Wisconsin", results_2016)
trump_vs_clinton(trump_clinton, "Wisconsin", results_2016, reliable = True)
trump_vs_clinton(trump_clinton, "Wisconsin", results_2016, likely_voters = True)
trump_vs_clinton(trump_clinton, "Minnesota", results_2016)
trump_vs_clinton(trump_clinton, "Minnesota", results_2016, reliable = True)
trump_vs_clinton(trump_clinton, "Minnesota", results_2016, likely_voters = True)
trump_vs_clinton(trump_clinton, "California", results_2016)
trump_vs_clinton(trump_clinton, "California", results_2016, reliable = True)
trump_vs_clinton(trump_clinton, "California", results_2016, likely_voters = True)
trump_vs_clinton(trump_clinton, "Illinois", results_2016)
trump_vs_clinton(trump_clinton, "Illinois", results_2016, reliable = True)
trump_vs_clinton(trump_clinton, "Illinois", results_2016, likely_voters = True)
trump_vs_clinton(trump_clinton, "New York", results_2016)
trump_vs_clinton(trump_clinton, "New York", results_2016, reliable = True)
trump_vs_clinton(trump_clinton, "New York", results_2016, likely_voters = True)
trump_vs_clinton(trump_clinton, "Texas", results_2016)
trump_vs_clinton(trump_clinton, "Texas", results_2016, reliable = True)
trump_vs_clinton(trump_clinton, "Texas", results_2016, likely_voters = True)
def get_best_pollsters(trump_clinton, state, results_2016):

    

    #Getting polls only from the final two weeks in a given state

    

    final_polls = trump_clinton[(trump_clinton["end_date"] >= "2016-10-17") & (trump_clinton["state"] == state)]

    

    #Getting a variable to represent the true results in each state

    

    final_polls = pd.merge(final_polls, results_2016, on = "state", how = "inner")

    

    #Getting the average results in the state for each pollster and the difference from actual results

    

    by_pollster = final_polls.groupby(["pollster", "dem_lead_y"]).mean().reset_index()[["pollster", "dem_lead_x", "dem_lead_y"]]

    by_pollster["trump_underestimation"] = by_pollster["dem_lead_x"] - by_pollster["dem_lead_y"]

    graph_pollsters = by_pollster.sort_values("trump_underestimation", ascending = False)

    

    #Getting the number of polls from each pollster

    

    num_polls = final_polls.groupby("pollster").size().reset_index()

    

    #Finally, plotting our results

    

    plt.subplots(figsize = (10,10))

    sns.barplot(x = graph_pollsters["trump_underestimation"], y = graph_pollsters["pollster"])

    plt.xlabel("Percentage Point Difference From Actual Result (more positive number = more Trump underestimation)")

    plt.ylabel(None)

    plt.show()

    

    #Table of most to least accurate pollsters in terms of magnitude of inaccuracy

    

    best_pollsters = by_pollster

    best_pollsters["trump_underestimation"] = abs(by_pollster["dem_lead_x"] - by_pollster["dem_lead_y"])

    best_pollsters = pd.merge(best_pollsters, num_polls, on = "pollster") 

    best_pollsters["Number of Polls"] = best_pollsters[0]

    best_pollsters["Pct Pts Inaccuracy"] = best_pollsters["trump_underestimation"]

    best_pollsters["Pollster"] = best_pollsters["pollster"]

    best_pollsters = best_pollsters.sort_values("trump_underestimation")[["Pollster", "Pct Pts Inaccuracy", "Number of Polls"]]

    

    #Linear Regression of polls released the final two weeks vs final inaccuracy

    

    sns.lmplot(data = best_pollsters, x = "Number of Polls", y = "Pct Pts Inaccuracy")

    plt.title(f"Number of Polls Released vs. Inaccuracy in {state}")

    print(stats.linregress(best_pollsters["Number of Polls"], best_pollsters["Pct Pts Inaccuracy"]))
get_best_pollsters(trump_clinton, "U.S.", results_2016)
get_best_pollsters(trump_clinton, "Florida", results_2016)
get_best_pollsters(trump_clinton, "Pennsylvania", results_2016)
get_best_pollsters(trump_clinton, "Ohio", results_2016)
get_best_pollsters(trump_clinton, "Michigan", results_2016)
get_best_pollsters(trump_clinton, "Wisconsin", results_2016)
get_best_pollsters(trump_clinton, "Minnesota", results_2016)