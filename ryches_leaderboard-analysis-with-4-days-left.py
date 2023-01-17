%matplotlib inline

import pandas as pd

import matplotlib.pyplot as plt
lb_data = pd.read_csv("../input/leaderboard-4-days-out/jigsaw-unintended-bias-in-toxicity-classification-publicleaderboard.csv")
lb_data = lb_data.set_index("SubmissionDate")
#skimming off the top 15

top_15_teams = lb_data.groupby("TeamId").max().sort_values("Score")[-15:]["TeamName"].values
top_15_subs = lb_data.loc[lb_data["TeamName"].isin(top_15_teams)]
top_15_subs = top_15_subs.drop("TeamId", axis = 1)
top_15_subs.pivot(columns="TeamName", values="Score")
#looking at peoples top score over time. Interesting to see when people made their rise. Sometimes coincides with information going public in discussion

top_15_subs.pivot(columns="TeamName", values="Score").interpolate().plot(legend = True, ylim = (.93, .95), figsize = (12,12))
#viewing prints of peoples rising. Recent activity and the size of peoples jumps is interesting

for i in top_15_subs.pivot(columns="TeamName", values="Score").interpolate():

    print(top_15_subs.pivot(columns="TeamName", values="Score")[i].dropna())
#graphs showing individual teams trends. More easily readable than the above graph

for i in top_15_subs.pivot(columns="TeamName", values="Score").interpolate():

    top_15_subs.pivot(columns="TeamName", values="Score")[i].dropna().plot(legend = True, ylim = (.93, .95), figsize = (12,12), title = str(i))

    plt.show()
top_15_subs.index = pd.to_datetime(top_15_subs.index)
top_15_subs_last_7 = top_15_subs.loc[top_15_subs.index > '2019-6-15']
#looking at teams last 7 days worth of submissions. Some are active and yuval has no activity. 

for i in top_15_subs_last_7.pivot(columns="TeamName", values="Score").interpolate():

    top_15_subs_last_7.pivot(columns="TeamName", values="Score")[i].dropna().plot(legend = True, ylim = (.93, .95), figsize = (12,12), title = str(i))

    plt.show()