import numpy as np # importing numpy.
import pandas as pd # imorting pandas.
import sqlite3 # Sql Database libary
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
with sqlite3.connect('../input/soccer/database.sqlite') as con:
    countries = pd.read_sql_query("SELECT * from Country", con)
    matches = pd.read_sql_query("SELECT * from Match", con)
    leagues = pd.read_sql_query("SELECT * from League", con)
    teams = pd.read_sql_query("SELECT * from Team", con)
    player = pd.read_sql_query("SELECT * from Player",con)
    player_attributes = pd.read_sql_query("SELECT * from Player_Attributes",con)
    sequence = pd.read_sql_query("SELECT * from sqlite_sequence",con)
    team_attributes = pd.read_sql_query("SELECT * from Team_Attributes",con)
countries.head(1) # To investigate the columns and rows
leagues.head(1) # To investigate the columns and rows
matches.head(1) # To investigate the columns and rows 
matches.info() # to see the informatiom about the dataframe No of Columns, Total Rows and the datatypes of the columns 
#Merge country and leauge data
countries_leagues = countries.merge(leagues,left_on="id",right_on="id",how="outer") # joining the countries and league dataframe together through outer join
countries_leagues = countries_leagues.drop("id",axis = 1) # dropping the id from the dataframe
countries_leagues = countries_leagues.rename(columns={'name_x':"country", 'name_y':"league"}) # renaming the name_x and name_y as country and league
countries_leagues.head()
#subsetting data with necessary columns
matches_new = matches[['id', 'country_id', 'league_id', 'season', 'stage', 'date',
                   'match_api_id', 'home_team_api_id', 'away_team_api_id',
                    'home_team_goal', 'away_team_goal']] # creating the new dataframe with the existing data from the matches dataframe

matches_new = matches_new.drop("id",axis=1) # dropping the id column from the dataframe.
matches_new.head(1)  # to inspect the columns and rows of merged dataframe
teams.head(1) # to inspect the columns and rows of the dataframe
team_attributes.head(1) # to inspect the columns and rows of dataframe
teams_new = teams.merge(team_attributes,left_on="team_api_id",right_on="team_api_id",how="left") # merging the team and team attributed dataframe together with left join.
teams_new = teams_new.drop(['id_x','id_y', 'team_fifa_api_id_y'],axis=1) # dropping the columns in the dataframe
teams_new["date"] = pd.to_datetime(teams_new["date"],format="%Y-%m-%d") # changing the datetime format 
teams_new = teams_new.rename(columns={'team_fifa_api_id_x':"team_fifa_api_id"}) # renaming the column
teams_new.shape # checking the shape of dataframe
teams_new.head(1) # to see the informatiom about the dataframe No of Columns, Total Rows and the datatypes of the columns 
player.head(1) # to see the informatiom about the dataframe No of Columns, Total Rows and the datatypes of the columns 
player_attributes.head(1) # inspect  the dataframe
player_info = player.merge(player_attributes,left_on="player_api_id",right_on="player_api_id",how="left") # merging the player and player attributes dataframe
player_info = player_info.drop(['id_x','id_y', 'player_fifa_api_id_y'],axis=1) # dropping the columns
player_info["date"] = pd.to_datetime(player_info["date"],format="%Y-%m-%d") # Changing the time and data format
player_info = player_info.rename(columns={'player_fifa_api_id_x':"player_fifa_api_id"}) # renaming the column names
player_info.head(1) # to see the informatiom about the dataframe No of Columns, Total Rows and the datatypes of the columns 
# Load your data and print out a few lines. Perform operations to inspect data
#   types and look for instances of missing or possibly errant data.
matches_new.info() #checking the datatypes of the columns
matches_new.duplicated().sum().any() # checking the duplicates 
# No duplicate data in the Matches_new Dataframe.
teams_new.info() # checking the shape and datatypes of the team_new Dataframe.
teams_new.duplicated().sum().any() # checking for duplicate datas.
# Duplicate datas found in the dataframe.
teams_new.isnull().sum().any() # checking null values in the dataframe.
player_info.info() # checking shape and datatype of the dataframe.
player_info.duplicated().sum().any()
player_info.isnull().sum().any()
# After discussing the structure of the data and any problems that need to be
#   cleaned, perform those cleaning steps in the second part of this section.

(matches.country_id == matches.league_id).any() # country_id and league_id are same.
matches_new = matches_new.drop(['league_id'],axis=1) # Dropping league_id because it is same as country_id.
matches_new.head(1)
matches_new = matches_new.rename(columns={'country_id':"id"}) # renaming country_id as id
matches_new.head()
matches_new.info() # checking shape and datatypes after cleaning the matches_new dataframe.
teams_new.head(1)
teams_new.drop_duplicates(inplace=True) # dropping the duplicates in the dataframe.
teams_new.info() # checking for duplicates after cleaning.
teams_new = teams_new.drop(['buildUpPlayDribbling'],axis=1) # dropping the buildUpPlayDribbling because the column has more null values.
teams_new.info()
teams_new.fillna(teams_new.mean(axis=0)) # filling the na values with mean
teams_new.head()
teams_new.info()
teams_new.dropna(inplace=True) #dropping the null values 
teams_new.info()
player_info.info()
player_info.dropna(inplace=True) # dropping the null values.
player_info.duplicated().sum() # checking for duplicates
player_info.info() # checking shape and datatypes.
player_info.player_name.nunique() # unique no of values in player_name.
player_info.head()
player_info.preferred_foot.unique() # Checking Unique values in preferred_foot column.
x = player_info[player_info["attacking_work_rate"] == 'high'] # attacking_work_rate of the player with preferred_foot.
x = x.groupby(["player_api_id","player_name","preferred_foot"]).count().reset_index() #counting the preferred_foot values.
plt.figure(figsize=(12,6))
plt.subplot(121)
x["preferred_foot"].value_counts().plot.pie(autopct = "%1.0f%%")
plt.ylabel("")
plt.title("Preferred foot by  player with High attacking rate")

x = player_info[player_info["defensive_work_rate"] == 'high'] # defensive_work_rate of the player with preferred_foot.
x = x.groupby(["player_api_id","player_name","preferred_foot"]).count().reset_index() #counting the preferred_foot values.
plt.figure(figsize=(12,6))
plt.subplot(122)
x["preferred_foot"].value_counts().plot.pie(autopct = "%1.0f%%")
plt.ylabel("")
plt.title("Preferred foot by  player with High defensive rate")
plt.show()
player_info['overall_rating'].corr(player_info['free_kick_accuracy']) # correlation between overall rating and free kick accuracy.
boxplot = player_info.boxplot(['overall_rating', 'free_kick_accuracy']) # plot the boxplot
sns.set(style="white", color_codes=True)
sns.jointplot(player_info["overall_rating"], player_info["free_kick_accuracy"], kind='kde', color="blue") # joint plot
player_info['overall_rating'].corr(player_info['penalties']) # correlation between overall rating and penalties.
boxplot = player_info.boxplot(['overall_rating', 'penalties']) #plotting boxplot.
sns.set(style="white", color_codes=True)
sns.jointplot(player_info["overall_rating"], player_info["penalties"], kind='kde', color="yellow")
bmi = player_info.iloc[:,[4,5,7]] # creating new dataframe with height, weight and overall rating.
bmi.head()
bmi['weight_kg'] = bmi.apply(lambda row: row.weight/2.20462 , axis =1) #creating new column with weight_kg 
# converting Weight in pounds to kg.
bmi.drop(['weight'],axis =1, inplace=True) # dropping the weight in pounds column.
bmi.rename(columns ={'weight_kg' : "weight"}) # renaming the column
bmi.info() # checking shape and datatype.
bmi['index_bmi'] = bmi.apply(lambda row: (row.weight_kg / (row.height**2))*10000 , axis =1) # creating new column with Bmi index by calculating.
bmi.head()
bmi.info()
bmi.overall_rating.corr(bmi.index_bmi)
def getBmiresults(index_bmi): # function to convert the bmi values into string
    if(index_bmi < 16):
        bmi_result = "severely underweight"
    elif(index_bmi >= 16 and index_bmi < 18.5):
        bmi_result = "underweight"
 
    elif(index_bmi >= 18.5 and index_bmi < 25):
        bmi_result = "healthy"
 
    elif(index_bmi >= 25 and index_bmi < 30):
        bmi_result = "overweight"
 
    elif(index_bmi >=30):
        bmi_result = "severely overweight"
    return bmi_result

bmi['bmi_result'] = bmi['index_bmi'].apply(getBmiresults) # to convert the BMI values in to String like Healthy, Overweight, Underweight and Severly Overweight.
bmi.head()
sns.set(style="white", color_codes=True)
sns.jointplot(bmi["overall_rating"], bmi["index_bmi"], kind='kde', color="red") # Plot joint plot
sns.barplot(bmi["overall_rating"], bmi["bmi_result"])
bmi_23 = bmi.groupby('bmi_result').count()
bmi_23.head()
bmi.head(1)
bmi.groupby("bmi_result").count().describe()
sns.boxplot(bmi["bmi_result"],bmi["overall_rating"])
sns.stripplot(bmi["bmi_result"],bmi["overall_rating"])
sns.countplot(bmi["bmi_result"])