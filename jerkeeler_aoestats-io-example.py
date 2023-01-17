# Basic setup stuff

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        display(os.path.join(dirname, filename))



plt.rcParams['figure.figsize'] = (12.0, 8.0)  # default figure size

sns.set_context("notebook")

sns.set()
matches = pd.read_csv("/kaggle/input/age-of-empires-ii-de-match-data/matches.csv")

display(matches.info())

matches.head()
players = pd.read_csv("/kaggle/input/age-of-empires-ii-de-match-data/match_players.csv")

display(players.info())

players.head()
# What's the most popular color?

colors = players[~players["color"].isna()]

percent_colors = colors.groupby("color")["token"].count().reset_index().rename(columns={"token": "counts"})

percent_colors["percent"] = percent_colors["counts"] / len(colors)

plt.pie(percent_colors["percent"],labels=percent_colors["color"], colors=percent_colors["color"], autopct='%1.1f%%')

plt.title("Most Popular Player Colors")

plt.show()
# What's the most popular server?

servers = matches[~matches["server"].isna()]

percent_servers = servers.groupby("server")["token"].count().reset_index().rename(columns={"token": "counts"})

percent_servers["percent"] = percent_servers["counts"] / len(servers)

plt.pie(percent_servers["percent"],labels=percent_servers["server"], autopct='%1.1f%%')

plt.title("Most Popular Servers")

plt.show()
# What patches are included in the data?

matches["patch"].unique()
# As someone might note, 37650 was hotfixed by 37906! Let's replace 37650 with 37906 so they're together.

tidy_matches = matches.copy()

tidy_matches["patch"] = tidy_matches["patch"].replace(37650, 37906)



patches = [35584, 36202, 36906, 37906]

latest_patch = patches[-1]



# Looking at the .info() output there are some matches that don't have an average_rating

tidy_matches = tidy_matches.dropna(subset=["average_rating"])



# Let's convert duration to datetime objects

tidy_matches["duration"] = pd.to_timedelta(tidy_matches["duration"])

tidy_matches["duration_s"] = tidy_matches["duration"].dt.total_seconds()



# Great, now let's do some plots!

fig, axs = plt.subplots(1, 3, figsize=(30, 5))

sns.boxplot(data=tidy_matches, x="patch", y="duration_s", hue="ladder", order=patches, ax=axs[0])

sns.countplot(data=tidy_matches, x="patch", hue="ladder", order=patches, ax=axs[1])

sns.distplot(tidy_matches["average_rating"], ax=axs[2])

plt.show()
# It looks like we have a few outliers in match duration, let's filter to matches less than 2 hours

tidy_matches = tidy_matches[(tidy_matches["duration_s"] > 60 * 5) & (tidy_matches["duration_s"] < 60 * 60 * 2)]

sns.catplot(kind="box", data=tidy_matches[tidy_matches["duration_s"] < 7200], x="patch", y="duration_s", hue="ladder", order=patches)

plt.show()
# Better! We can live with those longer matches in the dataset

# Final data cleanup, converting some columns to category variables and verifying all of our civs and maps look valid

tidy_matches["map"] = tidy_matches["map"].astype("category")

tidy_matches["map_size"] = tidy_matches["map_size"].astype("category")

tidy_matches["ladder"] = tidy_matches["ladder"].astype("category")

tidy_matches["patch"] = tidy_matches["patch"].astype("category")

tidy_matches["server"] = tidy_matches["server"].astype("category")

players["civ"] = players["civ"].astype("category")

display(sorted(tidy_matches["map"].unique()))

sorted(players["civ"].unique())
# Last but not least we'll join the tidy_matches with the player data

joined_df = pd.merge(players, tidy_matches, left_on="match", right_on="token", suffixes=["_player", "_match"])

joined_df.head()
# Let's first explore which civs are played most often!

ordered_civs = sorted(joined_df["civ"].unique())



g = sns.catplot(x="civ", data=joined_df, col="ladder", kind="count", order=ordered_civs)

g.set_xticklabels(rotation=90)

plt.show()
# That's great! Let's convert to relative percentages instead of raw counts

joined_1v1 = joined_df[joined_df["ladder"] == "RM_1v1"]

joined_team = joined_df[joined_df["ladder"] == "RM_TEAM"]



def get_play_rates(df):

    counts = df.groupby("civ")["token_player"].count().reset_index()

    counts["play_rate"] = counts["token_player"] / len(df)

    return counts.sort_values("play_rate", ascending=False)



play_rate_1v1, play_rate_team = get_play_rates(joined_1v1), get_play_rates(joined_team)
from matplotlib.ticker import PercentFormatter





# Thought it might be cool to annotate with the actual percents, but the graph gets super cluttered...

# def autolabel(rects, ax):

#     for rect in rects:

#         height = rect.get_height()

#         ax.annotate('{}%'.format(round(height * 100, 1)),

#                     xy=(rect.get_x() + rect.get_width() / 2, height / 2),

#                     xytext=(0, 3),  # 3 points vertical offset

#                     textcoords="offset points",

#                     ha='center', va='bottom', rotation=90, fontsize='x-small')



fig, [ax1, ax2] = plt.subplots(ncols=2, sharey=True)

rects = ax1.bar(play_rate_1v1["civ"], play_rate_1v1["play_rate"])

ax1.set_xticklabels(play_rate_1v1["civ"], rotation=90)

ax1.yaxis.set_major_formatter(PercentFormatter(1))

ax1.set_title("RM_1v1")

ax1.set_ylabel("Play rate (%)")

ax1.set_xlabel("Civ")

# autolabel(rects, ax1)



rects = ax2.bar(play_rate_team["civ"], play_rate_team["play_rate"])

ax2.set_xticklabels(play_rate_team["civ"], rotation=90)

ax2.set_title("RM_TEAM")

ax2.set_xlabel("Civ")

fig.tight_layout()



fig.suptitle("Civ Play Rates by Ladder", y=1.02)

plt.show()
# Looks like Franks, Mongols, and Britons are the most popular civs! Now, which civs actually win the most games?

def get_win_rates(df):

    return (

        df.groupby("civ")["winner"]

            .mean()

            .reset_index()

            .rename(columns={"winner": "win_rate"})

            .sort_values("win_rate", ascending=False)

    )



win_rate_1v1, win_rate_team = get_win_rates(joined_1v1), get_win_rates(joined_team)
fig, [ax1, ax2] = plt.subplots(ncols=2, sharey=True)

rects = ax1.bar(win_rate_1v1["civ"], win_rate_1v1["win_rate"])

ax1.set_xticklabels(win_rate_1v1["civ"], rotation=90)

ax1.yaxis.set_major_formatter(PercentFormatter(1))

ax1.set_title("RM_1v1")

ax1.set_ylabel("Win rate (%)")

ax1.set_xlabel("Civ")

ax1.set_ylim(0.4, 0.55)



rects = ax2.bar(win_rate_team["civ"], win_rate_team["win_rate"])

ax2.set_xticklabels(win_rate_team["civ"], rotation=90)

ax2.set_title("RM_TEAM")

ax2.set_xlabel("Civ")

fig.tight_layout()



fig.suptitle("Civ Win Rates by Ladder", y=1.02)

plt.show()
# Awesome! The Goths are really good?! But this is across all the data! How do the win rates vary by patch?

# From here on out I will only be looking at 1v1 games

ax = sns.pointplot(data=joined_1v1[joined_1v1["civ"] == "Goths"], x="patch", order=patches, y="winner")

ax.yaxis.set_major_formatter(PercentFormatter(1))

ax.set_xlabel("Patch")

ax.set_ylabel("Win rate (%)")

plt.title("Goth Win Rate by Patch (RM_1v1)")

plt.show()
# The goths used to be even better! Looks like they were nerfed in patch 36202... 

# Checks out! https://www.ageofempires.com/news/aoe2de-update-36202/ Looks like their unique bonus was nerfed.

# Let's look at the latest patch and determine our 95% confidence interval in the Goth's win rate. Seaborn is nice

# and provides it on the graph, but let's calculate it ourselves to see the actual values

goths_latest = joined_1v1[(joined_1v1["civ"] == "Goths") & (joined_1v1["patch"] == latest_patch)]

def bootstrap(df, n=10000, p=1000):

    means = pd.DataFrame([df.sample(n=n, replace=True)["winner"].mean() for _ in range(p)])

    return means.quantile(0.025)[0] * 100, means.quantile(0.975)[0] * 100

bottom, top = bootstrap(goths_latest)

win_rate_avg = goths_latest["winner"].mean() * 100

print(f"Goths have an average win rate of {win_rate_avg:.2f}% with a 95% confidence interval between {bottom:.2f}% and {top:.2f}%")
# Ok, but what about other civs?

g = sns.catplot(kind="point", 

                data=joined_1v1, 

                y="winner", 

                x="patch", 

                col="civ", 

                order=patches,

                col_wrap=5, 

                sharey=True, 

                sharex=False)

for ax in g.axes:

    ax.yaxis.set_major_formatter(PercentFormatter(1))

g.fig.suptitle("Civ Win Rates by Patch (RM_1v1)", y=1.02)

g.set_xlabels("Patch")

g.set_titles("{col_name}")

g.set_ylabels("Win rate (%)")

plt.show()
latest_1v1 = joined_1v1[joined_1v1["patch"] == latest_patch].reset_index(drop=True)

latest_1v1.head()
renames = {

    "token_player": "opponent",

    "civ": "opponent_civ",

}

opponents = latest_1v1[["match", "token_player", "civ"]].rename(columns=renames)

vs_df = pd.merge(latest_1v1, opponents, left_on="match", right_on="match").rename(columns={"token_player": "player"})



# Since we've merged the same dataframe a player will be an opponent of itself, we should drop those rows

vs_df = vs_df[vs_df["player"] != vs_df["opponent"]]

assert(len(vs_df) == len(latest_1v1))  # ensure we didn't lose any data

vs_df[["match", "player", "opponent", "civ", "opponent_civ", "winner"]].head(6)
# Awesome, now that we have the civs and their opponents, we can easily calculate their win rates vs other civs!

win_vs = vs_df.pivot_table(values="winner", index="civ", columns="opponent_civ")

win_vs.head()
# What does this look like graphically?

sns.heatmap(win_vs)

plt.title("Win Rates of Each Civ vs Each Other")

plt.show()
# One point stands out on this heat map. What is it?

melted_win_vs = win_vs.unstack().reset_index().rename(columns={0: "win_rate"})

melted_win_vs.sort_values("win_rate").nlargest(5, columns="win_rate")

# What's the most played map?

latest_1v1_matches = tidy_matches[(tidy_matches["patch"] == latest_patch) & (tidy_matches["ladder"] == "RM_1v1")]

percent_maps = (

    latest_1v1_matches.groupby("map")["token"]

    .count()

    .reset_index()

    .rename(columns={"token": "counts"})

)

percent_maps = percent_maps[percent_maps["counts"] > 0]

percent_maps["percent"] = percent_maps["counts"] / len(latest_1v1_matches)

percent_maps = percent_maps.sort_values("percent", ascending=False)



fig, ax = plt.subplots()

rects = ax.bar(percent_maps["map"], percent_maps["percent"])

ax.set_xticklabels(percent_maps["map"], rotation=90)

ax.yaxis.set_major_formatter(PercentFormatter(1))

ax.set_title("Map Play Rate RM_1v1")

ax.set_ylabel("Play rate (%)")

ax.set_xlabel("Map")

plt.show()
# Looks like Arabia is pretty popular this patch! (And all patches, fwiw)

# How often do certain civs win on each map? Well, let's use the pivot table and heat map again

win_maps = latest_1v1.pivot_table(values="winner", index="civ", columns="map")

display(win_maps.head())

sns.heatmap(win_maps)

plt.show()