import numpy  as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.rcParams["figure.figsize"] = (20, 7)
games          = pd.read_csv("../input/games.csv")
game_events    = pd.read_csv("../input/game_events.csv")
game_factions  = pd.read_csv("../input/game_factions.csv")
score_tiles    = pd.read_csv("../input/game_scoring_tiles.csv")
FAV_SORT_ORDER  = [ "FAV" + str(num) for num in range(1, 13) ]

faction_palette = {
    "acolytes":       "#ff6103",
    "alchemists":     "black",
    "auren":          "green",
    "chaosmagicians": "red",
    "cultists":       "brown",
    "darklings":      "black",
    "dragonlords":    "#ff6103",
    "dwarves":        "gray",
    "engineers":      "gray",
    "fakirs":         "yellow",
    "giants":         "red",
    "halflings":      "brown",
    "icemaidens":     "#9adde9",
    "mermaids":       "blue",
    "nomads":         "yellow",
    "riverwalkers":   "#faebd7",
    "shapeshifters":  "#faebd7",
    "swarmlings":     "blue",
    "witches":        "green",
    "yetis":          "#9adde9"
}
favor_events = game_events[
     game_events["event"].str.startswith("favor:FAV") &
    (game_events["faction"] != "all")
].copy()

favor_events["tile"] = favor_events["event"].str[6:]
favor_events.drop(columns=["event", "num"], inplace=True)
favor_events.head()
def barplot_favor_counts(series, title, ax=None):
    if ax is None:
        f, ax = plt.subplots(figsize=(20, 5))

    sns.set_color_codes("muted")
    sns.barplot(
        x=series.values,
        y=series.index,
        order=FAV_SORT_ORDER,
        color="b",
        ax=ax
    )
    ax.set_title(title)
    
favor_counts = favor_events["tile"].value_counts()
barplot_favor_counts(favor_counts, "Number of times each favor tile has been taken across all games")
f, axs = plt.subplots(ncols=2)

fav_count_by_round = \
    favor_events.groupby(["tile", "round"]) \
        .size()                             \
        .reset_index(name="count")          \
        .pivot("tile", "round", "count")    \
        .reindex(index=FAV_SORT_ORDER)
        
fav_percent_by_round = fav_count_by_round.div(fav_count_by_round.sum(axis=0), axis=1)
fav_percent_by_favor = fav_count_by_round.div(fav_count_by_round.sum(axis=1), axis=0)

for ax, df, title in [
    (axs[0], fav_percent_by_round, "Distribution of tiles taken each round \n ie, each column sums up to 100%"),
    (axs[1], fav_percent_by_favor, "Distribution of rounds each tile was taken \n ie, each row sums up to 100%")
]:
    sns.heatmap(df, annot=True, fmt=".0%", ax=ax)
    ax.set(title=title, xlabel="Round", ylabel="")

plt.show()
icemaiden_favor_counts = \
    favor_events[
        (favor_events["faction"] == "icemaidens") &
        (favor_events["round"]   == 1)
    ]["tile"].value_counts()

barplot_favor_counts(icemaiden_favor_counts, "Ice Maiden Round 1 Favor Tiles")
fav11_round_counts = fav11_events.groupby(["game", "round"]).size().reset_index(name="count")
fav11_round_counts["cum_count"] = fav11_round_counts.groupby(["game"])["count"].apply(lambda x: x.cumsum())

fav11_gone = fav11_round_counts[fav11_round_counts["cum_count"] == 3]
fav11_gone_round_counts = fav11_gone["round"].value_counts()
fav11_gone_round_counts["Never"] = len(games) - sum(fav11_gone_round_counts)
fav11_gone_round_counts
player_order = game_events[
    game_events["event"].str.startswith("order") &
    (game_events["faction"] != "all")
].copy()

player_order["player_order"] = player_order["event"].str[6:].apply(int)
player_order.drop(columns=["event", "num", "turn"], inplace=True)
player_order.head()
def add_player_count_and_order(df):
    return pd.merge(
        pd.merge(df, games[["game", "player_count"]], on="game"),
        player_order,
        on=["game", "faction", "round"]
    )

fav11_events = favor_events[favor_events["tile"] == "FAV11"]
turn2_fav11  = fav11_events[(fav11_events["round"] == 1) & (fav11_events["turn"] == 2)]
turn2_fav11  = add_player_count_and_order(turn2_fav11)
turn2_fav11.head()
total_by_player_count = games["player_count"].value_counts().reset_index(name="total_by_player_count")
total_by_player_count.rename(columns={ "index": "player_count" }, inplace=True)
total_by_player_count
def calc_by_player_count_and_order(df):
    counts = df.groupby(["player_count", "player_order"]) \
        .size()                           \
        .drop([1, 6, 7], errors="ignore") \
        .reset_index(name="count")
    
    counts = pd.merge(counts, total_by_player_count, on="player_count")
    counts["percent_by_player_count"] = counts["count"] / counts["total_by_player_count"]
    return counts

turn2_fav11_counts = calc_by_player_count_and_order(turn2_fav11)
turn2_fav11_counts
def barplot_percentages_by_player_order(df, title, ax=None):
    if ax is None:
        ax = plt.subplot()

    sns.set_color_codes("muted")
    sns.barplot(
        data=df,
        x="player_count",
        y="percent_by_player_count",
        hue="player_order",
        ax=ax
    )
    ax.legend(loc="upper left", title="Player Order")
    ax.set(
        title=title,
        xlabel="Number of Players",
        ylabel="Percentage"
    )

barplot_percentages_by_player_order(
    turn2_fav11_counts,
    "Percentage of Players who went for Fav11 on Turn 2"
)
turn2_favors = favor_events[
    (favor_events["round"] == 1) &
    (favor_events["turn"]  == 2)
]
turn2_favors = add_player_count_and_order(turn2_favors)
turn2_favors.head()
turn2_temples = game_events[
    (game_events["round"] == 1) &
    (game_events["turn"]  == 2) &
    (game_events["event"]   == "upgrade:TE") &
    (game_events["faction"] != "all")
]
turn2_temples = add_player_count_and_order(turn2_temples)
turn2_temples.head()
f, axs = plt.subplots(ncols=2)

turn2_favors_p1_counts = turn2_favors[
    turn2_favors["player_order"] == 1
]["tile"].value_counts()
barplot_favor_counts(turn2_favors_p1_counts, "Turn 2 Favor Tiles of Player 1", ax=axs[0])

barplot_percentages_by_player_order(
    calc_by_player_count_and_order(turn2_temples),
    "Percentage of Players who went for Temples on Turn 2",
    ax=axs[1]
)

plt.show()
turn1_dwellings = game_events[
    (game_events["round"] == 1) &
    (game_events["turn"]  == 1) &
    (game_events["event"]   == "build:D") &
    (game_events["faction"] != "all")
]
turn1_dwellings = add_player_count_and_order(turn1_dwellings)

barplot_percentages_by_player_order(
    calc_by_player_count_and_order(turn1_dwellings),
    "Percentage of Players who went for Building Dwellings on Turn 1"
)












