# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
ipl_df = pd.read_csv("../input/ipldata/matches.csv")

ipl_df
ipl_df.info()
# Correct error for Rising Pune Supergiants. Some show "Rising Pune Supergiant" 

# and others 'Rising Pune Supergiants'

ipl_df['team1'] = ipl_df['team1'].str.replace("Rising Pune Supergiants","Rising Pune Supergiant")

ipl_df['team2'] = ipl_df['team2'].str.replace("Rising Pune Supergiants","Rising Pune Supergiant")

ipl_df['toss_winner'] = ipl_df['toss_winner'].str.replace("Rising Pune Supergiants","Rising Pune Supergiant")

ipl_df['winner'] = ipl_df['winner'].str.replace("Rising Pune Supergiants","Rising Pune Supergiant")



teams_list = ipl_df["team1"].unique().tolist()

teams_list

# Create short names list



teams_short_list = ['RCB','DD','MI','RPS','CSK','KXP','KKR','SH','RR','GL','DC','PW','KTK','DCAP']

teams_short_list

# Create short names

teams_short_dict = {'Sunrisers Hyderabad': 'SH','Mumbai Indians': 'MI','Gujarat Lions':'GL',

                    'Rising Pune Supergiant': 'RPS', 'Royal Challengers Bangalore': 'RCB',

                    'Kolkata Knight Riders':'KKR', 'Delhi Daredevils': 'DD', 'Kings XI Punjab':'KXP',

                    'Chennai Super Kings': 'CSK','Rajasthan Royals':'RR','Deccan Chargers':'DC',

                    'Kochi Tuskers Kerala':'KTK','Pune Warriors':'PW','Delhi Capitals':'DCAP'}

teams_short_dict
# Number of games won list

n_games_won = ipl_df["winner"].count

# Lists

n_games_played_by_team_list = []

n_games_won_by_team_list = []

n_wins_by_winning_toss_percentage_list = []

perc_of_team_winning_bat_1st_list = []

perc_of_team_winning_bowl_1st_list = []
from tabulate import tabulate

for team in teams_list:

    # Calculating number of games played

    games_played_by_team = ipl_df.loc[(ipl_df["team1"] == team) | (ipl_df["team2"] == team)]

    n_games_played_by_team = games_played_by_team["team1"].count()

    n_games_played_by_team_list.append(n_games_played_by_team)



    # Calculating number of games won

    games_won_by_team = ipl_df.loc[((ipl_df["team1"] == team) | (ipl_df["team2"] == team)) & (ipl_df["winner"] == team)]

    n_games_won_by_team = games_won_by_team["winner"].count()

    n_games_won_by_team_list.append(n_games_won_by_team)

    



    n_games_won_by_team_perc = round(n_games_won_by_team / n_games_played_by_team * 100, 2)

    

    

    text = str(team), str(n_games_played_by_team),str(n_games_won_by_team),str(n_games_won_by_team_perc)

    print (text)



# Create of new DataFrame for new data

new_df = pd.DataFrame()

new_df["short_name"] = teams_short_list

new_df
# Create dicts

per_games_won_dict = dict(zip(teams_list, n_games_won_by_team_list))

n_games_played_by_team_dict = dict(zip(teams_list, n_games_played_by_team_list))





# Map dicts to DataFrame

new_df["Team"] = n_games_played_by_team_dict.keys()

new_df["short_name"] = teams_short_dict.values()

new_df["Games"] = n_games_played_by_team_dict.values()

new_df["Wins"] = per_games_won_dict.values()

new_df["Win%"] = round(new_df["Wins"] / new_df["Games"] * 100,2)





new_df.set_index("short_name", inplace=True)



new_df
for team in teams_list:

    # Number of wins

    n_wins = ipl_df[(ipl_df["winner"]) == team]

    n_team_wins = n_wins["winner"].count()



    # Number of toss wins

    toss_winner = ipl_df.loc[(ipl_df["toss_winner"] == team)]

    toss_winner = toss_winner["toss_winner"].count()



    # Win toss and win match

    winning_team = ipl_df.loc[(ipl_df["toss_winner"] == team) & (ipl_df["winner"] == team)]

    n_wins_by_winning_toss = winning_team["toss_winner"].count()



    # Create a percentage of win/toss

    n_wins_by_winning_toss_percentage = round(n_wins_by_winning_toss / toss_winner * 100, 2)

    n_wins_by_winning_toss_percentage_list.append(n_wins_by_winning_toss_percentage)



# Create dict

n_wins_by_winning_toss_dict = dict(zip(teams_list, n_wins_by_winning_toss_percentage_list))

# Map dicts to DataFrame

new_df["Win % because won toss"] = n_wins_by_winning_toss_dict.values()

for team in teams_list:

    # Team who wins batting 1st

    team_bat_1st = ipl_df.loc[(ipl_df["team1"] == team)]

    times_team_bat_1st = team_bat_1st["toss_decision"].count()

    wins_team_bat_1st = ipl_df.loc[(ipl_df["team1"] == team) & (ipl_df["winner"] == team)]

    n_wins_team_bat_1st = wins_team_bat_1st["team1"].count()

    n_wins_team_bat_1st_per = round(n_wins_team_bat_1st / times_team_bat_1st * 100, 2)

    

    perc_of_team_winning_bat_1st_list.append(n_wins_team_bat_1st_per)

    



    # Team who wins bowling 1st

    team_bowl_1st = ipl_df.loc[(ipl_df["team2"] == team)]

    times_team_bowl_1st = team_bowl_1st["team2"].count()

    wins_team_bowl_1st = ipl_df.loc[(ipl_df["team2"] == team) & (ipl_df["winner"] == team)]

    n_wins_team_bowl_1st = wins_team_bowl_1st["team2"].count()

    n_wins_team_bowl_1st_per = round(n_wins_team_bowl_1st / times_team_bowl_1st * 100, 2)

    

    perc_of_team_winning_bowl_1st_list.append(n_wins_team_bowl_1st_per)

    





perc_of_team_winning_bat_dict = dict(zip(teams_list, perc_of_team_winning_bat_1st_list))

new_df["Win % bat 1st"] = perc_of_team_winning_bat_dict.values()



perc_of_team_winning_bowl_dict = dict(zip(teams_list, perc_of_team_winning_bowl_1st_list))

new_df["Win % bowl 1st"] = perc_of_team_winning_bowl_dict.values()



new_df

# Create new df for teams that played more than 100 games

more_than_100_games_df = new_df.loc[(new_df["Games"] > 100)]

more_than_100_games_df

import matplotlib.pyplot as plt



# Create new df for teams that played more than 100 games

more_than_100_games_df = new_df.loc[(new_df["Games"] > 100)]



more_than_100_games_df[["Games", "Wins"]].plot(kind="bar",figsize=(14,7))

plt.xlabel("IPL Teams")

plt.ylabel("Games")

plt.legend()

plt.title("IPL games and wins")

plt.xticks(rotation=45)

plt.show()

more_than_100_games_df[["Win % bat 1st", "Win % bowl 1st"]].plot(kind="bar",figsize=(14,7))

plt.xlabel("IPL Teams")

plt.ylabel("Percentage")

plt.legend()

plt.title("IPL bat and bowl win %")

plt.xticks(rotation=45)

plt.show()