# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import pandasql as ps
import math
from plotnine import *
import fastai
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
labelled = pd.read_csv('../input/LabelledDataset.csv')
categorical = pd.read_csv('../input/CategoricalDataset.csv')
continuous = pd.read_csv('../input/ContinousDataset.csv')
original = pd.read_csv('../input/originalDataset.csv')
continuous.head(5)
continuous.loc[(continuous["Match Date"] == "Jan 5, 1971"),:] 

## Every match as two occurances, one each corresponding to "Team 1" and "Team 2"
continuous.head(5)
print(continuous.shape)
continuous_unique = continuous.loc[~continuous[["Scorecard"]].duplicated(),:]
print(continuous_unique.shape)

## We have created 2 data frames - the first one "continuous" has all records ... The second one "continuous_unique" has only
## one record per match !!
## First let's try to understand how many matches has every team played in total

temp = continuous.groupby(["Team 1"])
temp_1 = temp["Scorecard"].agg({"Matches Played":'count'}).reset_index()
print("This table gives a view of total matches played by all teams")
print("============================================================")
print(temp_1.sort_values(by="Matches Played",ascending=False))
continuous.loc[(continuous["Team 1"] == "U.S.A.") | (continuous["Team 2"] == "U.S.A.") ,:]
continuous["Winner"].unique()
## Now, let's . try and calculate number of matches won by every team in the data.

temp = continuous_unique["Winner"].groupby(continuous_unique["Winner"])
wins = temp.agg({"Wins":'count'}).reset_index()
print(wins)

## Let's merge temp_1 and wins 

wins = wins.rename(columns=({"Winner":"Team 1"})) 
records = pd.merge(left=temp_1,right=wins,how='left',on='Team 1')
records["Win percent"] = (records["Wins"] / records["Matches Played"]) * 100 
print("This table gives a summary of number of matches played and number of matches won by every team")
print("===============================================================================================")
print(records.sort_values(by="Win percent",ascending=False))
## Let's try and graphically visualize the above information 
## Let's pick top 10 countries basis the number of matches played and then plot them in a graph


my_dpi = 150
f, ax = plt.subplots(figsize=(19, 6),dpi=100)
##plt.rcParams['figure.figsize'] = 19,6
(sns.barplot(
              x = "Team 1",     
              y = "Matches Played",
              data = records.sort_values(by="Matches Played",ascending=False).head(15),
              color = "DarkBlue"
            ),
sns.barplot(
              x = "Team 1",    
              y = "Wins",
              data = records.sort_values(by="Matches Played",ascending=False).head(15),
              color = 'DeepSkyBlue'
           ))
f.suptitle(" Graph showing number of matches played and number of matches won by teams",fontsize = 20)

## Subset data for India matches only

india = continuous.loc[((continuous["Team 1"] == "India") | (continuous["Team 2"] == "India") ),:]
india_unique = india.loc[~india["Scorecard"].duplicated(),:]

print(india.shape)
print(india_unique.shape)
## Let's see India's performance against arch rival Pakistan - what has been their head-to-head record !!! 

india_pak = india_unique.loc[((india_unique["Team 1"] == "Pakistan") | (india_unique["Team 2"] == "Pakistan")),:]
india_pak.shape
pd.DataFrame(india_pak.pivot_table(index="Winner",values="Scorecard",aggfunc='count').reset_index()).rename(columns = ({"Scorecard":"Matches Won"}))

## Head to Head Pakistan has won more matches against India.
india_pak.head(10)
## Let's add "Venue" colum in the data - It will help us categorize India's wins in "Home", "Away" and "Neutral" venues
## We can then compare how has India fared against Pakistan in all 3 Venue categories

india_pak["venue"] = np.where((india_pak["Winner"] == india_pak["Host_Country"]),"Home_win",
                              np.where(((india_pak["Host_Country"] != "India") & (india_pak["Host_Country"] != "Pakistan")),"Neutral_win","Away_win"))

india_pak["country"] = "India"
india_pak["opposition"] = "Pakistan"
india_pak.head(15)
## Let's analyze India's performance against Pakistan at home, away and Neutral venues

home_wins_india = (india_pak.loc[((india_pak["venue"] == "Home_win") & (india_pak["Winner"] == "India")),:]).shape[0]
home_matches_india = (india_pak.loc[(india_pak["Host_Country"] == "India"),:]).shape[0]

away_wins_india = (india_pak.loc[((india_pak["venue"] == "Away_win") & (india_pak["Winner"] == "India")),:]).shape[0]
away_matches_india = (india_pak.loc[(india_pak["Host_Country"] == "Pakistan"),:]).shape[0]

neutral_wins_india = (india_pak.loc[((india_pak["venue"] == "Neutral_win") & (india_pak["Winner"] == "India")),:]).shape[0]
neutral_matches_india = (india_pak.loc[((india_pak["Host_Country"] != "India") & (india_pak["Host_Country"] != "Pakistan" )),:]).shape[0]

## We will create a Data Frame that will store this information so that we can retrieve it at once
## The structure of the dats should be
##                Matches Played     Matches Won
##  Home
##  Away 
##  Neutral

d = {"Matches played":[home_matches_india,away_matches_india,neutral_matches_india],"India wins":[home_wins_india,away_wins_india,neutral_wins_india] }
india_pak_results = pd.DataFrame(data=d,index=["Home (in India)","Away (in Pakistan)","Neutral"])
india_pak_results["win percent"] = (india_pak_results["India wins"] / india_pak_results["Matches played"]) * 100 
print(india_pak_results)
## Let's see how India has fared in Sharjah against Pakistan

sharjah = india_pak.loc[india_pak["Ground"] == "Sharjah",:]
pd.DataFrame(sharjah.pivot_table(index="Winner",values="Scorecard",aggfunc='count').reset_index()).rename(columns = ({"Scorecard":"Matches Won"}))
## Let's see how India has fared on Fridays 

india_pak.info()
## The above table shows that "Match Date" is "Object" Type - we need to convert this to date type

india_pak["Match Date"] = pd.to_datetime(india_pak["Match Date"])

india_pak.info()
## The above table now shows that "Match Date" has been converted to datetime data type ... 
## We now need to extract the weekday to see how many matches have been played on a Friday

india_pak["Match Day"] = india_pak["Match Date"].dt.weekday_name
india_pak.head(10)
fridays =india_pak.loc[india_pak["Match Day"] == "Friday",:]
pd.DataFrame(fridays.pivot_table(index="Ground",values="Scorecard",columns="Winner",aggfunc='count').reset_index()).rename(columns = ({"Scorecard":"Matches Won"}))
## It is generally perceived that after Sourav Ganguly took over as captain of India, our record overall improved.
## Let's see after 2000, when Ganguly took over, how has India fared against Pakistan

ganguly_onward = india_pak.loc[india_pak["Match Date"] > "2000-03-31" ,:]
pd.DataFrame(ganguly_onward.pivot_table(index="Winner",values="Scorecard",aggfunc='count').reset_index()).rename(columns = ({"Scorecard":"Matches Won"}))
## Let's see how India has fared against all teams - Matches played, Matches won and win % against all nations

india = continuous.loc[(continuous["Team 1"] == "India"),:]

## india.pivot_table(index="Team 2",values="Winner",aggfunc='count').reset_index()
temp = india.groupby(["Team 2","Winner"])
pd.DataFrame(temp["Winner"].agg({"Wins":'count'},margin=True))

india["Team 2"].value_counts()
## Although the above table is useful, but it's not good visually ! Let's create something that is better to use later.

temp_played = india.groupby(["Team 2"])
india_played = temp_played["Scorecard"].agg({"Matches Played":'count'}).reset_index()
india_played["Team 1"] = "India"
india_played = india_played.iloc[:,[2,0,1]]
india_played
def wins(x):
    matches_won = (india.loc[(india["Team 2"] == x)&(india["Winner"]=="India"),:]).shape[0]
    return matches_won

def losses(x):
    matches_lost = (india.loc[(india["Team 2"] == x)&(india["Winner"]== x),:]).shape[0]
    return matches_lost
india_played["India Wins"] = india_played["Team 2"].apply(func=wins)
india_played["India Losses"] = india_played["Team 2"].apply(func = losses)
india_played["India Win%"] = np.round(((india_played["India Wins"] / india_played["Matches Played"]) * 100),decimals=2)
india_played.iloc[:,[0,1,2,3] + [-1] + [4]]      ## Rearranging column order
## Let's try and plot the above table in visual format !

my_dpi = 150
f, ax = plt.subplots(figsize=(19, 6),dpi=100)
##plt.rcParams['figure.figsize'] = 19,6
(sns.barplot(
              x = "Team 2",     
              y = "Matches Played",
              data = india_played.sort_values(by="Matches Played",ascending=False).head(15),
              color = "Green"
            ),
sns.barplot(
              x = "Team 2",    
              y = "India Wins",
              data = india_played.sort_values(by="Matches Played",ascending=False).head(15),
              color = 'Yellow'
           ))
f.suptitle(" Graph showing India's record against all countries",fontsize = 20,color = 'Orange')
india["Match Date"].head(5)

## In the below table we see there are dates that do not conform to standard date convention. We need to fix this.
## We will have to remove anything beyond a "-" sign

india.loc[(india["Match Date"] == 'Jun 16-18, 1979'),:]

## The below date is a little messed up. We need to clean this so that we can convert this value to a date format
## To convert "Match Date" into a timestamp data type, we need to do some data massaging

def clean_date(x):
    if x.find("-") != -1:
        pos_dash = x.find("-",0,len(x))
        pos_comma = x.find(",",0,len(x))
        string_to_be_replaced = x[pos_dash:pos_comma]
        x = x[0:(pos_dash)] + x[(pos_comma):]
    
    else:
        pass
        
    return x
india["Match Date"] = india["Match Date"].apply(func=clean_date)
india["Match Date"] = pd.to_datetime(india["Match Date"])
## Now that the dates are cleaned, let's see how India performed before and after Saurav as a captain
## We will take 2000-03-31 as the cut off date, assuming after which Saurav took over 

before_saurav =  india.loc[india["Match Date"] <= "2000-03-31" ,:]
after_saurav =   india.loc[india["Match Date"] > "2000-03-31" ,:]
india_wins = (india.loc[(india["Winner"] == "India"),:].shape[0])
india_losses = india.loc[(india["Winner"] != "India") ,:].shape[0]
matches_total = india["Scorecard"].count()
india_wins_before_sg = before_saurav.loc[(india["Winner"] == "India"),:].shape[0]
india_losses_before_sg = before_saurav.loc[(india["Winner"] != "India"),:].shape[0]
matches_before_sg = before_saurav["Scorecard"].count()
india_wins_after_sg = after_saurav.loc[(india["Winner"] == "India"),:].shape[0]
india_losses_after_sg = after_saurav.loc[(india["Winner"] != "India"),:].shape[0]
matches_after_sg = after_saurav["Scorecard"].count()

print(matches_total)
d = {"Matches Played":[matches_total,matches_before_sg,matches_after_sg],
     "Matces Won":[india_wins,india_wins_before_sg,india_wins_after_sg],
     "Matches Lost":[india_losses,india_losses_before_sg,india_losses_after_sg]
    }

records = pd.DataFrame(data=d,index=["Matches Played","Matches Won","Matches Lost"])
records.rename(columns={"Matches Played":"Total","Matces Won":"Before SG","Matches Lost":"After SG"},inplace=True)
records = records.T 
records["Win percent"] = (records["Matches Won"] / records["Matches Played"]) * 100 
print(records)
## It would be cool to write a generic function which takes 2 teams as input(s) and returns their H2H record as output
## We will need to use "Continuous" and "continuous_unique" Data Frames for this.

def slice_data(team1,team2):
    temp_frame = continuous_unique.loc[((continuous_unique["Team 1"] == team1) | (continuous_unique["Team 1"] == team2)) &
                                       ((continuous_unique["Team 2"] == team1) | (continuous_unique["Team 2"] == team2))
                                      ]
    return temp_frame
    
def h2h(temp_frame):
    win_loss = pd.DataFrame(temp_frame.pivot_table(index="Winner",values="Scorecard",aggfunc='count').reset_index()).rename(columns = ({"Scorecard":"Matches Won"}))
    return win_loss
test = slice_data("South Africa","India")
h2h(test)
test.shape
