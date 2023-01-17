# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import math
from plotnine import *
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
pd.__version__
## For some interactive, jazzy graphs

import plotly
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot , iplot

init_notebook_mode(connected=True)
cf.go_offline()
labelled = pd.read_csv('../input/LabelledDataset.csv')
categorical = pd.read_csv('../input/CategoricalDataset.csv')
continuous = pd.read_csv('../input/ContinousDataset.csv')
original = pd.read_csv('../input/originalDataset.csv')
continuous_unique = continuous.loc[~continuous[["Scorecard"]].duplicated(),:]
## Let's see which teams play a lot of ODI Cricket - We will use a simple pie chart for this.

%matplotlib inline
plt.rcParams['figure.figsize'] = 18,15

input_pie = continuous.pivot_table(index='Team 1',values='Scorecard',aggfunc='count').reset_index().rename(columns={'Scorecard':'Matches Played'})
plt.pie(input_pie["Matches Played"],labels=input_pie["Team 1"],autopct='%.1f%%')
plt.title("Share of ODIs played by every team")
## To do this, we need to manipulate the date variable - we need to change "Match Date" variable from Object to datetime
## But there are dates which are in the format "Jun 16-18, 1975"... we will strip off anything from "-" to "," so that dates
## follow a standard format and become easy to manipulate

## This helper function will clean the dates

def clean_date(x):
    if x.find("-") != -1:
        pos_dash = x.find("-",0,len(x))
        pos_comma = x.find(",",0,len(x))
        string_to_be_replaced = x[pos_dash:pos_comma]
        x = x[0:(pos_dash)] + x[(pos_comma):]
    
    else:
        pass
        
    return x
## Call the above function so that dates are in order

continuous["Match Date"] = continuous["Match Date"].apply(func=clean_date)
continuous["Match Date"] = pd.to_datetime(continuous["Match Date"])
## Let's derive a "Year" field in the data set

continuous["year"] = continuous["Match Date"].dt.year 
continuous["Day"] = continuous["Match Date"].dt.weekday_name
## ODIs have evolved over the years - Let's try and see where was the tipping point after which it gained immense popularity

plt.rcParams['figure.figsize'] = 18,8

yearwise = continuous.pivot_table(index='year',values='Scorecard',aggfunc='count').reset_index().rename(columns={'Scorecard':'Matches Played'})
plt.plot(yearwise['year'],yearwise["Matches Played"],color = 'indigo')
plt.title("Trend of ODI matches played year on year by all teams")
## We will do this using 2 helper functions 

def slice_data(team1,team2):
    '''
    This function takes a subset of data using the parameters passed and returns the resulting data frame.
    '''
    temp_frame = continuous_unique.loc[((continuous_unique["Team 1"] == team1) | (continuous_unique["Team 1"] == team2)) &
                                       ((continuous_unique["Team 2"] == team1) | (continuous_unique["Team 2"] == team2))
                                      ]
    return temp_frame
    
def h2h(temp_frame):
    '''
    This function takes a subset data frame (with match details of 2 teams) as input and outputs their head to head record
    '''
    win_loss = pd.DataFrame(temp_frame.pivot_table(index="Winner",values="Scorecard",aggfunc='count').reset_index()).rename(columns = ({"Scorecard":"Matches Won"}))
    return win_loss

def get_h2h_records(team1,team2):
    """
    This function calls slice_data() and h2h() functions to print the win loss record of the teams
    """
    temp = slice_data(team1,team2)
    win_loss = h2h(temp)
    return win_loss

get_h2h_records("India","Pakistan")
## Let's write some helper functions that will help us achieve above 5 things !! 

def get_team_data(team):
    """
    This function takes a subset of data from overall data and returns data frame with "Team 1" as the parameter passed
    
    Input: Team name for which we need information
    Output: team_data - A data frame with match information for the team that is passed as input parameter
    """
    team_data = continuous.loc[(continuous["Team 1"] == team),:]
    return team_data

def matches(team_data):
    """
    This function calculates number of matches played by one particular team against all other teams
    
    Input: Data frame with match information about ONE team
    Output: matches_table - Data frame with information on number of matches played by the team against all other teams
    """
    matches_table =  pd.DataFrame(team_data["Team 2"].value_counts()).reset_index()
    matches_table.rename(columns = {"index":"Opposition","Team 2":"Matches Played"},inplace=True)
    return matches_table

def wins(matches_table,team_data):
    """
    This function will calculate wins for the team against all other nations
    
    Input: matches_table and team_data
    Output: matches_table with one more column added - "Matches Won"
    """
    won = pd.DataFrame(team_data.pivot_table(index=["Team 2","Winner"],values="Team 1",aggfunc='count').reset_index())
    won.drop(won[won["Team 2"] == won["Winner"]].index,inplace=True)
    won.rename(columns={"Team 2":"Opposition","Team 1":"Matches Won"},inplace=True)
    matches_table = pd.merge(left=matches_table,right=won,how = 'right',on="Opposition")
    return matches_table

def losses(matches_table,team_data):
    """
    This function calculates losses of a team against all other teams
    
    Input: matches_table and team_data
    Output: matches_table with one more column added as "Matches Lost" 
    """
    lost = pd.DataFrame(team_data.pivot_table(index=["Winner"],values="Team 2",aggfunc='count').reset_index())
    lost.drop(lost[(lost["Team 2"] == lost["Winner"]) ].index,inplace=True)
    lost.rename(columns={"Winner":"Opposition"},inplace=True)
    matches_table = pd.merge(left=matches_table,right=lost,how = 'left',on="Opposition")
    matches_table.rename(columns={"Team 2":"Matches Lost"},inplace=True)
    matches_table["Matches Lost"] = matches_table["Matches Lost"].fillna(0)
    matches_table["Matches Lost"] = matches_table["Matches Lost"].astype('int64')
    return matches_table
    
    
## Let's create a main function which will subsequently call all helper functions above ...

def team_record(team):
    """
    This function is a wrapper to the helper functions which gives a detailed account of matches played,
    matches won, matches lost by a team
    """
    team_data = get_team_data(team)
    matches_table = matches(team_data)
    matches_table_w = wins(matches_table,team_data)
    matches_table_w_l = losses(matches_table_w,team_data)
    return matches_table_w_l
    
## Let's look at overall record of a team

team = "India"  ## Pass this as input - the team for which you need to see the record

record = team_record(team)
overall_wins = record["Matches Won"].sum()
overall_losses = record["Matches Lost"].sum()
overall_played = record["Matches Played"].sum()

d = { "Matches Played" : [overall_played],
      "Matches Won" : [overall_wins],
      "Matches Lost" : [overall_losses]
     }

overall = pd.DataFrame(data=d,index=["Matches Played","Matches Won","Matches Lost"])
overall.drop(labels=["Matches Won","Matches Lost"],axis=0,inplace=True)

## The below print statement will print the overall win-loss record of a team
print(overall.rename(index={"Matches Played": team}))

print("==========================================================================================")

## The below print statement will print the detailed win-loss record of a team against all other countries
print(record)

##  Create a function which takes team as an input and returns matches won / lost year on year .. will help us assess consistency
##  of the teams.

def year_on_year(team):
    """
    Input : any team
    Output: Data Frame with year on year record for the team
    """
    team_data = get_team_data(team)
    matches = pd.DataFrame(team_data.pivot_table(index='year',values='Scorecard',aggfunc='count').reset_index()).rename(columns = {'Scorecard':'Matches Played'})
    wins = pd.DataFrame(team_data.loc[(team_data["Winner"] == team),:].pivot_table(index='year',values = 'Winner',aggfunc = 'count')).reset_index().rename(columns = {'Winner':'Wins'})
    losses = pd.DataFrame(team_data.loc[(team_data["Winner"] != team),:].pivot_table(index='year',values='Winner',aggfunc='count')).reset_index().rename(columns= {'Winner':'Losses'})
    temp1 = pd.merge(left=matches,right=wins,on='year',how='left') 
    team_record = pd.merge(left=temp1,right=losses,on='year',how='left')     
    
    team_record.fillna(0,inplace=True)
    team_record["Wins"] = team_record["Wins"].astype('int64')
    team_record["Losses"] = team_record["Losses"].astype('int64')
    
    return team_record

def create_graphs(team_record):
    """
    This function creates a visulalization of year on year record for a particular team
    """
    plt.plot(team_record["year"],team_record["Matches Played"],color='black',linestyle='-')
    plt.plot(team_record["year"],team_record["Wins"],color='green',linestyle='--' )
    plt.plot(team_record["year"],team_record["Losses"],color='red',linestyle=':')
    plt.legend() 
    
def team_record_graphs(team):
    """
    This is a wrapper function that calls year_on_year and create_graphs functions 
    """
    team_record = year_on_year(team)
    create_graphs(team_record)
    
%matplotlib inline
plt.rcParams['figure.figsize'] = 18,8

team = "South Africa"  ### Input the name of the team for which we need to see the year on year record graphically

title = "Graphs showing year on year record for " + team
plt.title(title,fontsize=15)

team_record_graphs(team)  ##  Call the main function that generates the graphs for the team
## Let's create a can help us set of functions that can help us visualize away win percent year on year for a team

def get_away_win_percent(team):
    team_data = get_team_data(team)
    matches = pd.DataFrame(team_data.loc[(team_data["Team 1"] == team),:].pivot_table(index='year',values='Scorecard',columns='Venue_Team1',aggfunc='count').reset_index()).rename(columns={'Away':'Away Matches','Home':'Home Matches','Neutral':'Neutral Matches'})
    matches.fillna(0,inplace=True)
    wins = pd.DataFrame(team_data.loc[(team_data["Winner"] == team),:].pivot_table(index='year',values='Scorecard',columns='Venue_Team1',aggfunc='count').reset_index()).rename(columns={'Away':'Away Wins','Home':'Home Wins','Neutral':'Neutral Wins'})
    wins.fillna(0,inplace=True)
    combined = pd.merge(left=matches,right=wins,on='year',how='left')
    combined["Total Away Matches"] = combined["Away Matches"] + combined["Neutral Matches"]
    combined["Total Away Wins"] = combined["Away Wins"] + combined["Neutral Wins"]
    combined["Away Win%"] = np.round(((combined["Total Away Wins"] / combined["Total Away Matches"]) * 100),decimals=2) 
    combined['Home Win%'] = np.round(((combined["Home Wins"] / combined["Home Matches"]) * 100),decimals=2)
    combined.fillna(0,inplace=True)
    
    return combined

def plot_away_win_percent(away_record):
    
    plt.plot(away_record['year'],away_record['Away Win%'],color='green')
    plt.plot(away_record['year'],away_record['Home Win%'],color='blue')
    plt.legend()
    
def show_away_trend(team):
    away_record = get_away_win_percent(team)
    plot_away_win_percent(away_record)
    
%matplotlib inline
plt.rcParams['figure.figsize'] = 18,9

team = "India" ## Input the team for which you want to see the away win% trendline

title = "Year on year Home / Away win percent trend for " + team
plt.title(title)

show_away_trend(team)
## Let's merge some information from Original dataset into continuous data set .. we will merge Margin colum from original dataset
## First, drop "Margin" column from continuous dataset

continuous.drop(columns='Margin',inplace=True)
continuous = pd.merge(left=continuous,right=original[["Margin","Scorecard"]],on='Scorecard',how='left')
## Q1. Have the teams batting first won more matches or the teams that have batted second ?
## We need to add a derived variable for this
def derive_result(x):
    
    if "wickets" in np.str(x):
        return "batting second"
    else:
        return "batting first"
continuous["Victory"] = continuous["Margin"].apply(lambda x : derive_result(x))
continuous.dropna(inplace=True)
continuous.loc[(~continuous["Scorecard"].duplicated()),"Victory"].value_counts()
## Q2. Which team has won most number of consecutive matches ?

all_teams = continuous["Team 1"].unique().tolist()
consecutive_wins_record = {}

for x in all_teams:
    
    team_data = get_team_data(x)   
    consec_win = 0
    wins_var = [0]
    prev_max_wins = 0
    
    for j,k in team_data.iterrows():
        
        if k["Winner"] == x:
            consec_win = consec_win + 1
            if consec_win > prev_max_wins:
                wins_var.pop(0)
                wins_var.insert(0,consec_win) 
                prev_max_wins = consec_win
            else:
                pass
        else:
            consec_win = 0

    consecutive_wins_record.update({x:wins_var[0]})
print("Consecutive wins record:",consecutive_wins_record)    
## Q3. Which day of the week has been the luckiest for teams ie on which day do teams have best win%

team = "India"
by_day = pd.DataFrame(continuous.loc[(continuous["Team 1"] == team) & (continuous["Winner"] == team),:].pivot_table(index='Day',values='Winner',aggfunc='count')).reset_index()
played_by_day = continuous.loc[(continuous["Team 1"] == team),:].pivot_table(index='Day',values='Scorecard',aggfunc='count').reset_index().rename(columns={'Scorecard':'Matches Played'})
by_day["Matches Played"] = played_by_day["Matches Played"]
by_day["Win%"] = np.round(((by_day["Winner"] / by_day["Matches Played"]) * 100), 1) 

## Let's plot this visually to understand this better

my_dpi = 150
f, ax = plt.subplots(figsize=(12,6),dpi=100)

labels = by_day["Matches Played"].tolist() + by_day["Winner"].tolist()

(sns.barplot(
              x = "Day",     
              y = "Matches Played",
              data = by_day,
              color = "grey"
            ),
sns.barplot(
              x = "Day",    
              y = "Winner",
              data = by_day,
              color = 'green'
)
)

plt.xlabel("Days of the week")

sns.despine(top=True,right=True,left=True)
cur_axes = plt.gca()
cur_axes.axes.get_yaxis().set_ticks([])
plt.ylabel("Matches Played / Matches Won")
rects = ax.patches

for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height + 5, label,
            ha='center', va='bottom')

subtitle = "Graph showing number of matches played and matches won by weekday for " + team
f.suptitle(subtitle,fontsize = 12)
## Q4. Which are the grounds where team should opt to bat first or bowl first ?

by_ground = (continuous.pivot_table(index='Ground',values='Scorecard',aggfunc='count')).reset_index().rename(columns={'Scorecard':'Matches Played'}).sort_values(by='Matches Played',ascending=False)
by_ground = by_ground.loc[(by_ground["Matches Played"] > 47),:]  ## Take only those grounds where matches played is > avg matches / ground

b1_2_wins = (continuous.loc[(continuous["Ground"].isin(by_ground["Ground"])),:].pivot_table(index=['Ground','Victory'],values='Scorecard',aggfunc='count')).reset_index().rename(columns={'Scorecard':'Wins'})

by_ground["Wins batting 1st"] = 0
by_ground["Wins batting 2nd"] = 0

for i , j in by_ground.iterrows():
    
    by_ground["Wins batting 1st"][i] = b1_2_wins.loc[(b1_2_wins["Ground"] == j["Ground"]) & (b1_2_wins["Victory"] == "batting first"),"Wins"]
    by_ground["Wins batting 2nd"][i] = b1_2_wins.loc[(b1_2_wins["Ground"] == j["Ground"]) & (b1_2_wins["Victory"] == "batting second"),"Wins"]

by_ground["Wins bat 1st%"] = np.round((by_ground["Wins batting 1st"] / by_ground["Matches Played"] * 100),1)
by_ground["Wins bat 2nd%"] = np.round((by_ground["Wins batting 2nd"] / by_ground["Matches Played"] * 100),1)
by_ground.sort_values(by='Wins bat 1st%', ascending=False)[:5] ## Gives top 5 grounds where batting 1st yeilds more wins
by_ground.sort_values(by='Wins bat 2nd%', ascending=False)[:5] ## Gives top 5 grounds where batting 2nd yields more wins
temp = get_team_data("India")
temp.to_csv("India_matches.csv")
