import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))
# load the data set in pandas

df = pd.read_csv("../input/2012-18_officialBoxScore.csv")
df.shape
pd.set_option('display.max_columns', 130) # we want to see all 119 columns in the output

df.head(6)
df.tail(6)
df = df[["gmDate","gmTime", "seasTyp", "teamAbbr", "teamRslt", "teamPTS", "teamFGA", "teamFGM", "teamFG%", 

        "team2PA", "team2PM", "team2P%", "team3PA", "team3PM", "team3P%", "teamFTA", "teamFTM", "teamFT%", "teamPPS", 

        "opptAbbr", "opptRslt", "opptPTS", "opptFGA", "opptFGM", "opptFG%", "oppt2PA", "oppt2PM",

        "oppt2P%", "oppt3PA", "oppt3PM", "oppt3P%", "opptFTA", "opptFTM", "opptFT%", "opptPPS"]]
df["gmDate"] = pd.to_datetime(df["gmDate"])

#df.dtypes

df = df.sort_values("gmDate")



df = df[df["gmDate"]>"2017-10-17"]



df.head()



# set the index new

df = df.reset_index(drop=True)



#df.head()

#df.tail()
df.seasTyp.unique()
df = df.drop_duplicates()

df.shape
line_counter = 0

drop_list = []



for line_counter in range(0, len(df.index)):

    for delete_line_counter in range(line_counter+1, len(df.index)):

        # compare same date and teamAbbr must be the same as opptAbbr

        if df.iloc[line_counter, 0] == df.iloc[delete_line_counter, 0] and df.iloc[line_counter, 3] == df.iloc[delete_line_counter, 19]:

            drop_list.append(delete_line_counter)

            break

            

            

#print(drop_list)
df = df.drop(df.index[drop_list])

df.shape

df.head(10)
df.isnull().values.any()
# sort values with the help of game date

df = df.sort_values("gmDate")



# set the index new

df = df.reset_index(drop=True)



df.head()
# which teams we have in the NBA 2018/ 2019 season?

teams = df.teamAbbr.unique()

print(teams)
# let's prepare the dictionaries before using them



two_fga = {} # 2 point field goal attempts

three_fga = {} # 3 point field goal attempts

two_pfg_perc = {} # 2 point field goal percentage

three_pfg_perc = {} # 3 point field goal percentage



# set up the two_fga dictionary

for team in teams:

    if team not in two_fga:

        two_fga[team] = []



# set up the three_fga dictionary

for team in teams:

    if team not in three_fga:

        three_fga[team] = []

        

# set up the two_pfg_perc dictionary

for team in teams:

    if team not in two_pfg_perc:

        two_pfg_perc[team] = []

        

# set up the three_pfg_perc dictionary

for team in teams:

    if team not in three_pfg_perc:

        three_pfg_perc[team] = []



        

# e.g. the two_fga dictionary contains all teams as the key with an empty list

print(two_fga)
df.shape # take a look how many lines/ rows we have



line_counter = 0

k = 5 # number of games which we like to use for the average



for line_counter in range(0, len(df.index)):

    first_team = df.loc[line_counter, "teamAbbr"]

    second_team = df.loc[line_counter, "opptAbbr"]

    if len(two_fga[first_team]) == k and len(two_fga[second_team]) == k:

        # Prediction

        # Points first team

        pred_2P_first_team = np.mean(two_fga[first_team]) * np.mean(two_pfg_perc[first_team]) * 2

        pred_3P_first_team = np.mean(three_fga[first_team]) * np.mean(three_pfg_perc[first_team]) * 3

        pred_points_first_team = pred_2P_first_team + pred_3P_first_team # predicted points first team

        df.loc[line_counter, "teamPTSpred"] = pred_points_first_team

        # Points second team

        pred_2P_second_team = np.mean(two_fga[second_team]) * np.mean(two_pfg_perc[second_team]) * 2

        pred_3P_second_team = np.mean(three_fga[second_team]) * np.mean(three_pfg_perc[second_team]) * 3

        pred_points_second_team = pred_2P_second_team + pred_3P_second_team # predicted points second team

        df.loc[line_counter, "opptPTSpred"] = pred_points_second_team

        # prediction right or wrong

        if pred_points_first_team > pred_points_second_team and df.loc[line_counter, "teamPTS"] > df.loc[line_counter, "opptPTS"]:

            df.loc[line_counter, "predRslt"] = 1

        elif pred_points_first_team < pred_points_second_team and df.loc[line_counter, "teamPTS"] < df.loc[line_counter, "opptPTS"]:

            df.loc[line_counter, "predRslt"] = 1

        else:

            df.loc[line_counter, "predRslt"] = 0

        

        # delete oldest entry for prediction

        del two_fga[first_team][-1]

        del three_fga[first_team][-1]

        del two_pfg_perc[first_team][-1]

        del three_pfg_perc[first_team][-1]

        del two_fga[second_team][-1]

        del three_fga[second_team][-1]

        del two_pfg_perc[second_team][-1]

        del three_pfg_perc[second_team][-1]

    # collect data for average calculation

    if len(two_fga[first_team]) < k:

        # write data for first team

        two_fga[first_team].append(df.loc[line_counter, "team2PA"])

        three_fga[first_team].append(df.loc[line_counter, "team3PA"])

        two_pfg_perc[first_team].append(df.loc[line_counter, "team2P%"])

        three_pfg_perc[first_team].append(df.loc[line_counter, "team3P%"])

    if len(two_fga[second_team]) < k:

        # write data second_team

        two_fga[second_team].append(df.loc[line_counter, "oppt2PA"])

        three_fga[second_team].append(df.loc[line_counter, "oppt3PA"])

        two_pfg_perc[second_team].append(df.loc[line_counter, "oppt2P%"])

        three_pfg_perc[second_team].append(df.loc[line_counter, "oppt3P%"])        
df.tail()
df.isna().sum()
number_learn = df["predRslt"].isna().sum() # how many games we couldn't predict because of learning -> 79

number_right = df["predRslt"].sum() # how many games we predicted right

rows = len(df.index)

perc_right_pred = number_right / (rows - number_learn)

print(perc_right_pred)
# load the standing data set in pandas

standings = pd.read_csv("../input/2012-18_standings.csv")
standings.tail()
standings = standings[standings["stDate"]=="2018-04-11"]

standings = standings.sort_values(by="gameWon", ascending=False) # sorted by "gameWon" because the rank is per conference



display(standings)