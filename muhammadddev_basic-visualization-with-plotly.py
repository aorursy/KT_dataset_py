import os

import datetime



import numpy as np 

import pandas as pd 



import plotly.offline as po

import plotly.graph_objs as go



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
def integer_encoder(team_dataframe, team):

    FTR_encoded = []

    for date in team_dataframe["Date"]:

        team_dataframe_2 = team_dataframe[team_dataframe["Date"] == date]

        if (((team_dataframe_2["HomeTeam"].values == team) & (team_dataframe_2["FTR"].values == "H")) | ((team_dataframe_2["AwayTeam"].values == team) & (team_dataframe_2["FTR"].values == "A"))):

            FTR_encoded.append(int(1))

        elif (((team_dataframe_2["HomeTeam"].values == team) & (team_dataframe_2["FTR"].values == "A")) | ((team_dataframe_2["AwayTeam"].values == team) & (team_dataframe_2["FTR"].values == "H"))):

            FTR_encoded.append(int(-1))

        elif team_dataframe_2["FTR"].values == "D":

            FTR_encoded.append(int(0))



    return FTR_encoded
def syncing_FTR(main_dates, team_dates, team_FTR):

    synced_FTR = []

    for main_date in main_dates:

        if main_date in team_dates:

            for i, team_date in enumerate(team_dates):

                if main_date == team_date:

                    synced_FTR.append(team_FTR[i])

        else:

            synced_FTR.append(None)



    return synced_FTR
data_path = "/kaggle/input/english-premier-league-season-1819/season-1819_csv.csv"

df = pd.read_csv(data_path)

df.head()
df.columns
pl_teams = list(df["HomeTeam"].unique())

pl_teams
teams_df = []

FTRs_encoded = []

for team in pl_teams:

    team_df = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)]

    teams_df.append(team_df)

    FTR_encoded = integer_encoder(team_df, team)

    FTRs_encoded.append(FTR_encoded)
main_dates = sorted(df["Date"].unique(), key=lambda date: datetime.datetime.strptime(date, "%d/%m/%Y"))

main_dates
teams_synced_FTR = []

for i in range(len(teams_df)):

    teams_synced_FTR.append(syncing_FTR(main_dates, teams_df[i]["Date"].to_list(), FTRs_encoded[i]))
fig = go.Figure()

for i, team in enumerate(pl_teams):

    fig.add_trace(go.Scatter(

        x = main_dates,

        y = teams_synced_FTR[i],

        name = team,

        mode = 'lines+markers',

        connectgaps=True

    ))



fig.show()