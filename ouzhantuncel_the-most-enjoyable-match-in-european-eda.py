# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sqlite3





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        



        



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_data= sqlite3.connect('/kaggle/input/soccer/database.sqlite') 
match=pd.read_sql_query("select id,country_id,league_id,season,stage,date,match_api_id,home_team_api_id,away_team_api_id,home_team_goal,away_team_goal  from Match",df_data)

match.head()
plymatch=pd.read_sql_query("select t.id,t.team_api_id,t.team_long_name,m.id,m.country_id,m.league_id,m.season,m.stage from Team t left join Match m on m.home_team_api_id=t.team_api_id ",df_data)

plymatch.head()

plymatch=pd.read_sql_query("select t.id,t.team_api_id,t.team_long_name,m.id,m.country_id,m.league_id,m.season,m.stage From Team t left join Match m on m.away_team_api_id=t.team_api_id ",df_data)

plymatch.head()
matchgoal=pd.read_sql_query("Select season,home_team_goal,away_team_goal from Match",df_data)

matchgoal

goalcount=pd.read_sql_query("Select season, league_id,home_team_api_id,max(home_team_goal),away_team_api_id,max(away_team_goal) from Match",df_data)

goalcount
kindteam=pd.read_sql_query("Select country_id,name from League where id=4769",df_data)

kindteam
match11=pd.read_sql_query("Select home_team_api_id,away_team_api_id from Match where league_id=4769",df_data)

match11
teamname=pd.read_sql_query("Select team_api_id,team_long_name from Team where team_api_id=10242",df_data)

teamname
teamname1=pd.read_sql_query("Select team_api_id,team_long_name from Team where team_api_id=9847",df_data)

teamname1
matchhh=pd.read_sql_query("Select season,home_team_api_id,away_team_api_id from Match where home_team_api_id=10242 and away_team_api_id=9847",df_data)

matchhh