import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Read fifa files by the individual year



fifa_15 = pd.read_csv("/kaggle/input/fifa-20-complete-player-dataset/players_15.csv")

fifa_16 = pd.read_csv("/kaggle/input/fifa-20-complete-player-dataset/players_16.csv")

fifa_17 = pd.read_csv("/kaggle/input/fifa-20-complete-player-dataset/players_17.csv")

fifa_18 = pd.read_csv("/kaggle/input/fifa-20-complete-player-dataset/players_18.csv")

fifa_19 = pd.read_csv("/kaggle/input/fifa-20-complete-player-dataset/players_19.csv")

fifa_20 = pd.read_csv("/kaggle/input/fifa-20-complete-player-dataset/players_20.csv")
# Tagging each file by the Year before concating into a single master file



fifa_15["Year"] = 2015

fifa_16["Year"] = 2016

fifa_17["Year"] = 2017

fifa_18["Year"] = 2018

fifa_19["Year"] = 2019

fifa_20["Year"] = 2020
# Generating a Master file for analysis



frames = (fifa_15, fifa_16, fifa_17, fifa_18, fifa_19, fifa_20)

master_data = pd.concat(frames, axis=0, join='outer', ignore_index=False, keys=None,

          levels=None, names=None, verify_integrity=False, copy=True)
master_data.shape
columns = master_data.columns.to_numpy()

columns
dist_subset = fifa_20[["overall", "wage_eur", "age", "pace"]]



dist_subset.hist(bins=25, figsize=(12,8))

plt.show()
class Club_compare:

    def __init__(self, teams, Year):

        self.year = Year

        self.teams = teams

        

    def best15(self):

        club_best_players = pd.DataFrame([])

        for team in self.teams:

            for year in self.year:

                club_top_15 = master_data[(master_data["club"]==team) & (master_data["Year"]==year)]

                club_top_15 = club_top_15.sort_values("overall", ascending=False)

                club_top_15 = club_top_15[:15]

                frames = [club_best_players, club_top_15]

                club_best_players = pd.concat(frames, axis=0, join='outer', ignore_index=False, keys=None,

                  levels=None, names=None, verify_integrity=False, copy=True)

        

        return club_best_players

    

    def stats(self):

        best15_data = Club_compare.best15(self)

        team_year_average = pd.DataFrame([])

        

        for team in self.teams:

            for year in self.year:

                team_year = best15_data[(best15_data["club"]==team) & (best15_data["Year"]==year)]

                if self.year == 2015:

                    wage = 0

                else:

                    wage = int(round(team_year[["wage_eur"]].mean()))

                overall = int(round(team_year[["overall"]].mean()))

                age = int(round(team_year[["age"]].mean()))              

                average_list = pd.DataFrame(np.array([[team, overall, wage, age, year]]), 

                                                     columns=("Club", "Overall", "Wage_avg", "Age_avg", "Year"))

                frames = (average_list, team_year_average)

                team_year_average = pd.concat(frames, axis=0, join='outer', ignore_index=False, keys=None,

                      levels=None, names=None, verify_integrity=False, copy=True)



        team_year_average = team_year_average.sort_values("Year")

        team_year_average = team_year_average.reset_index(drop=True)

        

        return team_year_average

                          

    def passing(self): #Passing was removed from the stats function since a lot of the values were missing and hence concatanation was creating an issue

        best15_data = Club_compare.best15(self)

        team_passing_average = pd.DataFrame([])

        for team in self.teams:

            for year in self.year:

                team_year = best15_data[(best15_data["club"]==team) & (best15_data["Year"]==year)]

                passing_avg = int(round(team_year[["passing"]].mean()))

                average_values = pd.DataFrame(np.array([[team, passing_avg, year]]), 

                                                     columns=("Club", "Passing_avg", "Year"))

                frames = (average_values, team_passing_average)

                team_passing_average = pd.concat(frames, axis=0, join='outer', ignore_index=False, keys=None,

                      levels=None, names=None, verify_integrity=False, copy=True)



        team_passing_average = team_passing_average.sort_values("Year")

        team_passing_average = team_passing_average.reset_index(drop=True)

        

        return team_passing_average
teams = ["Liverpool", "Chelsea", "Arsenal", "Manchester City", "Manchester United", "Tottenham Hotspur"]

Years = [2015, 2016, 2017, 2018, 2019, 2020]

EPL_top6 = Club_compare(teams, Years)
top6_stats = EPL_top6.stats()

top6_passing = EPL_top6.passing()

top6_players = EPL_top6.best15()

top6_stats # Note that the wage values for the year 2015 seems to be missing
top6_stats["Overall"] = top6_stats["Overall"].astype("int")

top6_stats["Age_avg"] = top6_stats["Age_avg"].astype("int")



plt.figure(figsize=(12,8))

plt.rcParams["axes.labelsize"] = 16

ax = sns.barplot(top6_stats["Year"], top6_stats["Overall"], hue=top6_stats["Club"], palette="muted")

ax.set(ylim=(78,88))

plt.xticks(fontsize=16)

plt.yticks(fontsize=14)

plt.show()
plt.figure(figsize=(12,8))

plt.rcParams["axes.labelsize"] = 16

ax = sns.barplot(top6_stats["Year"], top6_stats["Wage_avg"], hue=top6_stats["Club"], palette="muted")

# ax.set(ylim=(78,88))

plt.xticks(fontsize=16)

plt.yticks(fontsize=14)

plt.show()
top6_players.head()
top6_20 = top6_players[top6_players["Year"]==2020]



fig = plt.figure(figsize=(12,8))

fig.suptitle('Wages (Eur): 2020', fontsize=20)

sns.violinplot(top6_20["club"], top6_20["wage_eur"])

plt.xlabel("")

plt.ylabel("")

plt.xticks(fontsize=16, rotation=45)

plt.yticks(fontsize=16)

plt.show()
City_passing = top6_passing[top6_passing["Club"]=="Manchester City"]

pg_passing = City_passing.drop(City_passing.index[0]) #Dropping 2015 values since Pep joined City in 2016

pg_passing
pg_passing["Passing_avg"] = pg_passing["Passing_avg"].astype("int")



plt.figure(figsize=(12,8))

plt.suptitle("Passing under Guardiola", fontsize=20)

sns.set_style("darkgrid")

sns.lineplot(pg_passing["Year"], pg_passing["Passing_avg"])

plt.xlabel("")

plt.xticks(fontsize=16)

plt.yticks(fontsize=16)

plt.show()
# This class searches through the players for the inputted positions and then returns top 3 players for each players. 

# Note - Some players have a single position listed for them, whereas some players can play at different positions. Hence if we search for Messi under "CF" alone, we won't be able to find him.

# He would be found only under "RW, CF, ST". This algorithm takes this into account while searching for the top players and hence may have a single player listed multiple times 



class Dream_team:

    def __init__(self, year):

        self.positions = master_data["player_positions"].drop_duplicates()

        self.year = year

        self.data = master_data[master_data["Year"]==self.year]

    

    def position_search(self, position):

        players = pd.DataFrame([])

        for pos in position:

            positions = self.positions[self.positions.str.contains(pos)]

            positions= positions.tolist()

            for x in positions: 

                player_data = self.data[self.data["player_positions"]==x]

                player_data["searched_position"] = pos

                frames = (player_data, players)

                players = pd.concat(frames, axis=0, join='outer', ignore_index=False, keys=None,

                          levels=None, names=None, verify_integrity=False, copy=True)

        players = players.sort_values("overall", ascending=False)

        return players

    

    def top_players(self, positions):

        all_player_data = Dream_team.position_search(self, positions)

        dream_players = pd.DataFrame([])

        for position in positions:

            players = all_player_data[all_player_data["searched_position"]==position]

            sorting = players.sort_values("overall", ascending=False)

            chosen_player = sorting[:3]

            frames = (chosen_player, dream_players)

            dream_players = pd.concat(frames, axis=0, join="outer", ignore_index=False, keys=None,

                      levels=None, names=None, verify_integrity=False, copy=True)

        return dream_players
dream_team_2020 = Dream_team(2020)

player_positions = ["GK", "CB", "ST", "RB", "LB", "CM", "LM", "RM", "RW", "LW"]

players = dream_team_2020.top_players(player_positions)

players[["short_name", "club", "player_positions", "searched_position", "overall"]]
gk_players = dream_team_2020.position_search("GK")

wing_players = dream_team_2020.position_search(["RW", "LW"])

midfielders = dream_team_2020.position_search(["CM", "LM", "RM"])

cb_players = dream_team_2020.position_search(["CB", "RB", "LB"])

forward_players = dream_team_2020.position_search(["CF", "ST"])
gk_players["player_tag"] = "Goal Keepers"

cb_players["player_tag"] = "Defenders"

midfielders["player_tag"] = "Midfielders"

wing_players["player_tag"] = "Wingers"

forward_players["player_tag"] = "Strikers"

frames = (gk_players, cb_players, midfielders, wing_players, forward_players)

comparison_data = pd.concat(frames, axis=0, join="outer")
def comparison(data, x_data, y_data):

    plt.figure(figsize=(12,8))

    sns.boxenplot(data[x_data], data[y_data])

    plt.xlabel("")

    plt.xticks(fontsize=16)

    plt.yticks(fontsize=16)

    plt.show()
comparison(comparison_data, "player_tag", "weight_kg")
comparison(comparison_data, "player_tag", "height_cm")
comparison(comparison_data, "player_tag", "passing")
table = pd.crosstab(comparison_data["player_tag"], comparison_data["body_type"])

table
comparison_data["body"] = comparison_data["body_type"]

comparison_data[comparison_data["body"]=="PLAYER_BODY_TYPE_25"]
comparison_data["body"] = comparison_data["body"].replace(["Neymar", "C. Ronaldo", "Messi","Courtois","PLAYER_BODY_TYPE_25"], "Lean")

comparison_data["body"] = comparison_data["body"].replace(["Shaqiri", "Akinfenwa"], "Stocky")
table_1 = pd.crosstab(comparison_data["player_tag"], comparison_data["body"])

table_1
table_1.div(table.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(12,8))

plt.xlabel('')

plt.ylabel('Percentage')

plt.xticks(fontsize=16, rotation=45)

plt.yticks(fontsize=14)

plt.show()
players_unique = master_data.drop_duplicates(["sofifa_id"])

players_unique["player count"] = 1

abc = players_unique["nationality"].value_counts()[:20]
plt.figure(figsize=(12,8))

sns.set_context("talk")

sns.set_style('dark') #sets the size of the charts

sns.barplot(abc.values, abc.index, palette="muted")

plt.show()
# Extract all the different teams in Fifa



fifa_teams_15 = fifa_15["club"].drop_duplicates()

fifa_teams_16 = fifa_16["club"].drop_duplicates()

fifa_teams_17 = fifa_17["club"].drop_duplicates()

fifa_teams_18 = fifa_18["club"].drop_duplicates()

fifa_teams_19 = fifa_19["club"].drop_duplicates()

fifa_teams_20 = fifa_20["club"].drop_duplicates()
youngest_team_15 = Club_compare(fifa_teams_15, [2015])

youngest_team_16 = Club_compare(fifa_teams_16, [2016])

youngest_team_17 = Club_compare(fifa_teams_17, [2017])

youngest_team_18 = Club_compare(fifa_teams_18, [2018])

youngest_team_19 = Club_compare(fifa_teams_19, [2019])

youngest_team_20 = Club_compare(fifa_teams_20, [2020])
youngest_15 = youngest_team_15.stats()

youngest_16 = youngest_team_16.stats()

youngest_17 = youngest_team_17.stats()

youngest_18 = youngest_team_18.stats()

youngest_19 = youngest_team_19.stats()

youngest_20 = youngest_team_20.stats()
youngest_15 = youngest_15.sort_values("Age_avg")

youngest_16 = youngest_16.sort_values("Age_avg")

youngest_17 = youngest_17.sort_values("Age_avg")

youngest_18 = youngest_18.sort_values("Age_avg")

youngest_19 = youngest_19.sort_values("Age_avg")

youngest_20 = youngest_20.sort_values("Age_avg")
youngest_frames = [youngest_15[youngest_15["Age_avg"]==youngest_15["Age_avg"].min()], 

                               youngest_16[youngest_16["Age_avg"]==youngest_16["Age_avg"].min()],

                               youngest_17[youngest_17["Age_avg"]==youngest_17["Age_avg"].min()],

                               youngest_18[youngest_18["Age_avg"]==youngest_18["Age_avg"].min()],

                               youngest_19[youngest_19["Age_avg"]==youngest_19["Age_avg"].min()],

                               youngest_20[youngest_20["Age_avg"]==youngest_20["Age_avg"].min()]]

youngest = pd.concat(youngest_frames, axis=0, join="outer")
youngest  #Below are the youngest teams in each edition of the game
oldest_frames = [youngest_15[youngest_15["Age_avg"]==youngest_15["Age_avg"].max()], 

                               youngest_16[youngest_16["Age_avg"]==youngest_16["Age_avg"].max()],

                               youngest_17[youngest_17["Age_avg"]==youngest_17["Age_avg"].max()],

                               youngest_18[youngest_18["Age_avg"]==youngest_18["Age_avg"].max()],

                               youngest_19[youngest_19["Age_avg"]==youngest_19["Age_avg"].max()],

                               youngest_20[youngest_20["Age_avg"]==youngest_20["Age_avg"].max()]]

oldest = pd.concat(oldest_frames, axis=0, join="outer")
oldest  #Displaying the oldest teams in each edition
# Let us describe a good team by the criteria of its top 15 players averaging an overall of greater than 70 



youngest_good_15 = youngest_15[youngest_15["Overall"].astype("int")>70].sort_values("Age_avg")

youngest_good_16 = youngest_16[youngest_16["Overall"].astype("int")>70].sort_values("Age_avg")

youngest_good_17 = youngest_17[youngest_17["Overall"].astype("int")>70].sort_values("Age_avg")

youngest_good_18 = youngest_18[youngest_18["Overall"].astype("int")>70].sort_values("Age_avg")

youngest_good_19 = youngest_19[youngest_19["Overall"].astype("int")>70].sort_values("Age_avg")

youngest_good_20 = youngest_20[youngest_20["Overall"].astype("int")>70].sort_values("Age_avg")
youngest_good_frames = [youngest_good_15[youngest_good_15["Age_avg"]==youngest_good_15["Age_avg"].min()], 

                               youngest_good_16[youngest_good_16["Age_avg"]==youngest_good_16["Age_avg"].min()],

                               youngest_good_17[youngest_good_17["Age_avg"]==youngest_good_17["Age_avg"].min()],

                               youngest_good_18[youngest_good_18["Age_avg"]==youngest_good_18["Age_avg"].min()],

                               youngest_good_19[youngest_good_19["Age_avg"]==youngest_good_19["Age_avg"].min()],

                               youngest_good_20[youngest_good_20["Age_avg"]==youngest_good_20["Age_avg"].min()]]

youngest_good = pd.concat(youngest_good_frames, axis=0, join="outer")
youngest_good
oldest_good_frames = [youngest_good_15[youngest_good_15["Age_avg"]==youngest_good_15["Age_avg"].max()], 

                               youngest_good_16[youngest_good_16["Age_avg"]==youngest_good_16["Age_avg"].max()],

                               youngest_good_17[youngest_good_17["Age_avg"]==youngest_good_17["Age_avg"].max()],

                               youngest_good_18[youngest_good_18["Age_avg"]==youngest_good_18["Age_avg"].max()],

                               youngest_good_19[youngest_good_19["Age_avg"]==youngest_good_19["Age_avg"].max()],

                               youngest_good_20[youngest_good_20["Age_avg"]==youngest_good_20["Age_avg"].max()]]

oldest_good = pd.concat(oldest_good_frames, axis=0, join="outer")
oldest_good