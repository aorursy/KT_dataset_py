# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
bets = pd.read_csv("/kaggle/input/nba-historical-stats-and-betting-data/nba_betting_money_line.csv")

bets.head()
games = pd.read_csv("/kaggle/input/nba-historical-stats-and-betting-data/nba_games_all.csv")

games.head()
# Remove useless columns

bets = bets.loc[bets["book_name"] == "Bovada"]

bets = bets.drop(["book_name", "book_id"], axis=1)
# Merge game stats

bets = bets.merge(games[["game_id", "team_id", "a_team_id", "is_home", "wl", "season_year"]], on=["game_id", "team_id", "a_team_id"])
# Convert American Odds to Decimal odds

def conv_odds(row):

    absrow = abs(row)

    if row >= 0:

        odds = (absrow / 100) + 1

    else:

        odds = (100 / absrow) + 1

    return odds

bets["price1"] = bets["price1"].apply(conv_odds)

bets["price2"] = bets["price2"].apply(conv_odds)
# Convert wl to 0 or 1

def conv_wl(row):

    if row == "W":

        return 1

    elif row == "L":

        return 0

bets["wl"] = bets["wl"].apply(conv_wl)
# Where predictions are being generated, right now only random numbers

bets["pred"] = np.random.randint(0, 2, bets.shape[0])
# Find the profit with the given prediction model

def get_profit(row):

    profit = 0

    if row["wl"] == row["pred"]:

        if row["wl"] == 0:

            profit = row["price2"] - 1

        elif row["wl"] == 1:

            profit = row["price1"] - 1

    else:

        profit = -1

    return profit

bets["profit"] = bets.apply(get_profit, axis=1)

bets["profit"].sum()
year_index = []

year_profit = []

for y in bets["season_year"].unique():

    year_bets = bets.loc[bets["season_year"] == y]

    year_index.append(y)

    year_profit.append(year_bets["profit"].sum())

yearly_profit = pd.Series(data=year_profit, index=year_index)

yearly_profit.sort_index()

yearly_profit
sns.barplot(yearly_profit.index, yearly_profit)