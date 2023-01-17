import matplotlib.pyplot as plt

import pandas as pd



%matplotlib inline
# Spiel and Kennerspiel des Jahres brand colours

red = "#E30613" # SdJ

black = "#193F4A" # KdJ
# load a list of all Spiel des Jahres winners plus the most recent nominees

sdj_all = pd.read_csv("https://blog.recommend.games/experiments/sdj.csv", index_col="bgg_id")

sdj_all.shape
sdj_all.head()
# load a list of all Kennerspiel des Jahres winners plus the most recent nominees

ksdj_all = pd.read_csv("https://blog.recommend.games/experiments/ksdj.csv", index_col="bgg_id")

ksdj_all.shape
ksdj_all.head()
# load data about all kinds of games scraped from BoardGameGeek

games = pd.read_csv("/kaggle/input/board-games/bgg_GameItem.csv", index_col="bgg_id")

games.shape
games.head()
# Join SdJ winners with more data from BGG

sdj = (

    sdj_all[sdj_all.winner == 1]

    .drop(columns=["url", "winner"])

    .join(games, how="left")

    .sort_values("sdj")

)

sdj.shape
# Join KdJ winners with more data from BGG

ksdj = (

    ksdj_all[ksdj_all.winner == 1]

    .drop(index=[203416, 203417])  # only keep one Exit game

    .drop(columns=["url", "winner"])

    .join(games, how="left")

    .sort_values("ksdj")

)

ksdj.shape
# columns we are interested in for our analysis

columns = [

    "name",

    # "year",

    # "designer",

    # "artist",

    # "publisher",

    "complexity",

    "avg_rating",

    "bayes_rating",

    "rank",

    "num_votes",

    "min_players",

    "max_players",

    "min_time",

    "max_time",

    "min_age",

]
sdj[["sdj"] + columns]
ksdj[["ksdj"] + columns]
# How did the geek rating develop over time?

plt.plot(ksdj.ksdj, ksdj.bayes_rating, color=black, linewidth=3)

plt.plot(sdj.sdj, sdj.bayes_rating, color=red, linewidth=3)

plt.legend(["Kennerspiel", "Spiel"])

plt.savefig("bayes_rating.svg")

plt.show()
# How did the complexity develop over time?

plt.plot(ksdj.ksdj, ksdj.complexity, color=black, linewidth=3)

plt.plot(sdj.sdj, sdj.complexity, color=red, linewidth=3)

plt.legend(["Kennerspiel", "Spiel"])

plt.savefig("complexity.svg")

plt.show()
# How did the play time develop over time?

plt.fill_between(ksdj.ksdj, ksdj.min_time, ksdj.max_time, color=black, alpha=0.5)

plt.plot(

    ksdj.ksdj,

    (ksdj.min_time + ksdj.max_time) / 2,

    color=black,

    linestyle="dashed",

    linewidth=3,

)

plt.fill_between(sdj.sdj, sdj.min_time, sdj.max_time, color=red, alpha=0.5)

plt.plot(

    sdj.sdj,

    (sdj.min_time + sdj.max_time) / 2,

    color=red,

    linestyle="dashed",

    linewidth=3,

)

plt.legend(["Kennerspiel", "Spiel"])

plt.savefig("time.svg")

plt.show()
# How did the player count develop over time?

plt.fill_between(ksdj.ksdj, ksdj.min_players, ksdj.max_players, color=black, alpha=0.5)

plt.fill_between(sdj.sdj, sdj.min_players, sdj.max_players, color=red, alpha=0.5)

plt.legend(["Kennerspiel", "Spiel"])

plt.savefig("players.svg")

plt.show()
# How did the player age develop over time?

plt.plot(ksdj.ksdj, ksdj.min_age_rec, color=black, linestyle="dotted", linewidth=3)

plt.plot(ksdj.ksdj, ksdj.min_age, color=black, linewidth=3)

plt.plot(sdj.sdj, sdj.min_age_rec, color=red, linestyle="dotted", linewidth=3)

plt.plot(sdj.sdj, sdj.min_age, color=red, linewidth=3)

plt.legend(["Kennerspiel (users)", "Kennerspiel (box)", "Spiel (users)", "Spiel (box)"])

plt.savefig("age.svg")

plt.show()
# Filter games from this year according to the SdJ criteria

# Sort them by their geek rating in order to find candidates for SdJ

games[

    (games.year >= 2019)

    & (games.year <= 2020)

    & (games.max_time <= 60)

    & (games.complexity <= 2)

    & (games.min_players <= 4)

    & (games.max_players >= 3)

    & ((games.min_age <= 14) | (games.min_age_rec <= 12))

][columns].sort_values("bayes_rating", ascending=False).head(50)
# Filter games from this year according to the KdJ criteria

# Sort them by their geek rating in order to find candidates for SdJ

games[

    (games.year >= 2019)

    & (games.year <= 2020)

    & (games.max_time <= 120)

    & (games.complexity >= 1.5)

    & (games.complexity <= 3.5)

    & (games.min_players <= 4)

    & (games.max_players >= 3)

    & ((games.min_age <= 14) | (games.min_age_rec <= 12))

][columns].sort_values("bayes_rating", ascending=False).head(50)