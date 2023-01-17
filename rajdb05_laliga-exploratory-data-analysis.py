import pandas as pd

import numpy as np

df=pd.read_csv('/kaggle/input/laliga-history/Laliga.csv',header=1)
df.head()
#Replacing all '-' with '0'

df_new=df.replace('-',0)

df_new.tail()
#Creating a temporary column,'Debut_temp', that reformats data in 'Debut' column

df_new['Debut_temp']=df_new['Debut'].apply(lambda x: str(x).split('-'))

df_new.head()
#Function to convert Debut_temp from 'yyyy,yy' to 'yyyy,yyyy'

def format_year(x):

    if len(x)>1:

        if int(x[0])<1999:

            return [x[0],int(x[1])+1900]

        else:

            return [x[0],int(x[1])+2000]

    else:

        return x
#Applying the function to the Debut_temp column for conversion from 'yyyy,yy' to 'yyyy,yyyy'

df_new['Debut_temp']=df_new['Debut_temp'].apply(format_year)
#Function to check if any of the years in 'Debut_temp' >= 1930 and <=1980

def check_debut(x):

    if len(x)==1:

        return int(x[0])>=1930 and int(x[0])<=1980

    else:

        return ((int(x[0])>=1930 and int(x[0])<=1980)or(int(x[1])>=1930 and int(x[1])<=1980))
#Printing the teams that debuted between 1930 and 1980 (includes 1930 and 1980)

df_new[df_new['Debut_temp'].apply(check_debut)]['Team']
#String columns that are required to be converted to numeric for calculation purposes

cols_tonumeric=['Pos', 'Seasons', 'Points', 'GamesPlayed', 'GamesWon',

       'GamesDrawn', 'GamesLost', 'GoalsFor', 'GoalsAgainst', 'Champion',

       'Runner-up', 'Third', 'Fourth', 'Fifth', 'Sixth', 'T','BestPosition']
#Converting the above mentioned columns to numeric

df_new[cols_tonumeric]=df_new[cols_tonumeric].apply(pd.to_numeric)
#Top 5 teams in terms of points

df_new.sort_values(by='Points',ascending=False).loc[:,['Team','Points']].head()
#Function to calculate goal difference

def Goal_diff_count(Goals_For,Goals_Against):

    return Goals_For-Goals_Against
#Applying the Goal_diff_count function to each element in the dataset

df_new['Goal_diff']=df_new.apply(lambda x: Goal_diff_count(x['GoalsFor'],x['GoalsAgainst']),axis=1)
#Question 4ii - Team with the maximum goal difference

df_new[df_new['Goal_diff']==df_new['Goal_diff'].max()][['Team','Goal_diff']]
#Question 4iii - Team with the minimum goal difference

df_new[df_new['Goal_diff']==df_new['Goal_diff'].min()][['Team','Goal_diff']]
#Question 5i - Winning percentage of each team

df_new['Winning Percent']=(df_new['GamesWon']/df_new['GamesPlayed'])*100

df_new['Winning Percent']=df_new['Winning Percent'].fillna(value=0)
#Question 5i - Top 5 teams in terms of winning percentage

df_new.sort_values(by='Winning Percent',ascending=False).loc[:,['Team','Winning Percent']].head()
#Question 6 - Points grouped by best position

df_new.groupby(by='BestPosition')['Points'].sum()