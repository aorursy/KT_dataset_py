import numpy as np

import pandas as pd

%pylab inline



data = pd.read_csv("../input/steam-200k.csv", header=None, index_col=False, 

                   names=["UserID", "GameName", "Behavior", "Value"])

data[:5]
import missingno as msno

msno.matrix(data, figsize=(10,4))

# no missing values!
print("Data shape: ", data.shape)

print("Unique users in dataset: ", data.UserID.nunique())

print("Unique games in dataset: ", data.GameName.nunique())
# average number of games per user

games_transacted = data.groupby("UserID")["GameName"].nunique()

games_purchased = data[data.Behavior == "purchase"].groupby("UserID")["Value"].sum()

games_played = data[data.Behavior == "play"].groupby(["UserID","GameName"])["Value"].sum()



print("Games Transacted Stats:")

print(games_transacted.describe())



print("Games Purchased Stats:")

print(games_purchased.describe())



print("Games Played Stats:")

print(games_played.describe())
import seaborn as sns

f, ax = plt.subplots(figsize=(12, 8))

mean_v = np.mean(games_transacted)

std_v = np.std(games_transacted)

inliers = games_transacted[games_transacted < mean_v + 4* std_v]



g = sns.distplot(inliers, axlabel="Number of Games Transacted With", ax=ax, kde=False)

ax.set(yscale="log")

plt.show()
f, ax = plt.subplots(figsize=(12, 8))

mean_v = np.mean(games_purchased)

std_v = np.std(games_purchased)

inliers = games_purchased[games_purchased < mean_v + 4* std_v]



sns.distplot(inliers, ax=ax, axlabel="Number of Games Purchased", kde=False)

ax.set(yscale="log")

plt.show()
f, ax = plt.subplots(figsize=(12, 8))

mean_v = np.mean(games_played)

std_v = np.std(games_played)

inliers = games_played[games_played < mean_v + 4* std_v]



sns.distplot(inliers, ax=ax, axlabel="Number of Hours Played", kde=False)

ax.set(yscale="log")

plt.show()
# in terms of purchases

game_purchases = data[data.Behavior == "purchase"].groupby("GameName")["Value"].sum()

plt.figure(figsize=(10,8))

game_purchases.sort_values(ascending=False)[:20].plot(kind="barh")
# in terms of play time

game_plays = data[data.Behavior == "play"].groupby("GameName")["Value"].sum()

plt.figure(figsize=(10,8))

game_plays.sort_values(ascending=False)[:20].plot(kind="barh")
# for the top 10 most purchased games, what is the distribution of their play times?

top_10_most_purchased = game_purchases.sort_values(ascending=False)[:10].index.tolist()

top_10_most_purchased_playhours = data[(data.GameName.isin(top_10_most_purchased)) & (data.Behavior == "play")]

ax = sns.boxplot(x="GameName", y="Value", data=top_10_most_purchased_playhours)

plt.xticks(rotation=90)

ax.set(yscale='log')

plt.show()
# let's take care of the 1 ratings first

purchase_ratings = data[data.Behavior == "purchase"].groupby(["UserID", "GameName"])["Value"].count()

purchase_ratings.describe()
purchase_ratings[purchase_ratings > 1][:5]
data[(data.UserID==561758) & (data.GameName.str.contains("Civilization"))]
purchase_ratings[purchase_ratings > 1].shape
# making every value a 1

purchase_ratings[purchase_ratings > 1] = 1
games_played_per_user = data[data.Behavior=="play"].groupby(["UserID"])["GameName"].nunique()



average_games_played = games_played_per_user.mean()

average_games_played
games_played_per_user.describe()
weighted_games_played = games_played * (average_games_played / games_played_per_user)
def to_explicit_ratings(series):

    games = series.index.levels[1].tolist()

    hours_ratings = series.copy()

    for game_played in games:

        sliced_data = hours_ratings.xs(game_played, level=1)

        descr = sliced_data.describe()

        a = sliced_data[sliced_data >= descr["75%"]].index.tolist()

        hours_ratings.loc[(a, game_played)] = 5

        b = sliced_data[(sliced_data >= descr["50%"]) & (sliced_data < descr["75%"])].index.tolist()

        hours_ratings.loc[(b, game_played)] = 4

        c = sliced_data[(sliced_data >= descr["25%"]) & (sliced_data < descr["50%"])].index.tolist()

        hours_ratings.loc[(c, game_played)] = 3

        d = sliced_data[sliced_data < descr["25%"]].index.tolist()

        hours_ratings.loc[(d, game_played)] = 2

    

    return hours_ratings
hours_ratings = to_explicit_ratings(weighted_games_played)
sns.distplot(hours_ratings, kde=False)
mean_weighted_ratings = purchase_ratings.combine(hours_ratings, max)

print(mean_weighted_ratings.shape)

print(mean_weighted_ratings.describe())

sns.distplot(mean_weighted_ratings, kde=False)
non_weighted_ratings = to_explicit_ratings(games_played)

non_weighted_ratings = purchase_ratings.combine(non_weighted_ratings, max)

print(non_weighted_ratings.shape)

print(non_weighted_ratings.describe())

sns.distplot(non_weighted_ratings, kde=False)
games_played_per_user[games_played_per_user == 1][:5]
def display_before_and_after(user_id, before, after):

    print("====Before====")

    games_played = min(len(non_weighted_ratings.ix[user_id]), 10)

    idx = before.xs(user_id, level=0).sample(games_played).index.tolist()

    print(before.xs(user_id, level=0)[idx])

    print("====After====")

    print(after.xs(user_id, level=0)[idx])
display_before_and_after(62990992, non_weighted_ratings, mean_weighted_ratings)
display_before_and_after(16081636, non_weighted_ratings, mean_weighted_ratings)
display_before_and_after(5250, non_weighted_ratings, mean_weighted_ratings)
display_before_and_after(309167186, non_weighted_ratings, mean_weighted_ratings)
# cleanup

purchase_ratings = None

hours_ratings = None

data = None