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
import pandas as pd
import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
%matplotlib inline

pd.set_option('display.max_columns',None)
df_match = pd.read_csv('/kaggle/input/indian-premier-league-csv-dataset/Match.csv')
df_playermatch = pd.read_csv('/kaggle/input/indian-premier-league-csv-dataset/Player_Match.csv')
df_ballbyball =  pd.read_csv('/kaggle/input/indian-premier-league-csv-dataset/Ball_by_Ball.csv')
df_player = pd.read_csv('/kaggle/input/indian-premier-league-csv-dataset/Player.csv')
df_season = pd.read_csv('/kaggle/input/indian-premier-league-csv-dataset/Season.csv')
df_team = pd.read_csv('/kaggle/input/indian-premier-league-csv-dataset/Team.csv')
df_ballbyball = df_ballbyball.merge(df_match,on='Match_Id')
df_ballbyball.head(3)
df_ballbyball = df_ballbyball.merge(df_season,on='Season_Id')
df_ballbyball.head(3)
df_playermatch = df_playermatch.merge(df_player,on='Player_Id')
df_playermatch.head(3)
df_playermatch = df_playermatch.merge(df_team,on='Team_Id')
df_playermatch.head(3)
df_playermatch = df_playermatch.merge(df_ballbyball,on = 'Match_Id')
df_playermatch.head()
df = df_playermatch.copy()
df = df.drop(['Unnamed: 7'],axis = 1)
df = df.dropna()
df_2010 = df.loc[(df['Season_Year'] == 2010)]
df_2011 = df.loc[(df['Season_Year'] == 2011)]
df_2012 = df.loc[(df['Season_Year'] == 2012)]
df_2013 = df.loc[(df['Season_Year'] == 2013)]
df_2014 = df.loc[(df['Season_Year'] == 2014)]
df_2015 = df.loc[(df['Season_Year'] == 2015)]
df_2016 = df.loc[(df['Season_Year'] == 2016)]
df_2010['series'] = np.where(df_2010['Player_Id'] == df_2010['Man_of_the_Series_Id'],1,0)
df_2011['series'] = np.where(df_2011['Player_Id'] == df_2011['Man_of_the_Series_Id'],1,0)
df_2012['series'] = np.where(df_2012['Player_Id'] == df_2012['Man_of_the_Series_Id'],1,0)
df_2013['series'] = np.where(df_2013['Player_Id'] == df_2013['Man_of_the_Series_Id'],1,0)
df_2014['series'] = np.where(df_2014['Player_Id'] == df_2014['Man_of_the_Series_Id'],1,0)
df_2015['series'] = np.where(df_2015['Player_Id'] == df_2015['Man_of_the_Series_Id'],1,0)
df_2016['series'] = np.where(df_2016['Player_Id'] == df_2016['Man_of_the_Series_Id'],1,0)


df_2010['orange'] = np.where(df_2010['Player_Id'] == df_2010['Orange_Cap_Id'],1,0)
df_2011['orange'] = np.where(df_2011['Player_Id'] == df_2011['Orange_Cap_Id'],1,0)
df_2012['orange'] = np.where(df_2012['Player_Id'] == df_2012['Orange_Cap_Id'],1,0)
df_2013['orange'] = np.where(df_2013['Player_Id'] == df_2013['Orange_Cap_Id'],1,0)
df_2014['orange'] = np.where(df_2014['Player_Id'] == df_2014['Orange_Cap_Id'],1,0)
df_2015['orange'] = np.where(df_2015['Player_Id'] == df_2015['Orange_Cap_Id'],1,0)
df_2016['orange'] = np.where(df_2016['Player_Id'] == df_2016['Orange_Cap_Id'],1,0)


df_2010['purple'] = np.where(df_2010['Player_Id'] == df_2010['Purple_Cap_Id'],1,0)
df_2011['purple'] = np.where(df_2011['Player_Id'] == df_2011['Purple_Cap_Id'],1,0)
df_2012['purple'] = np.where(df_2012['Player_Id'] == df_2012['Purple_Cap_Id'],1,0)
df_2013['purple'] = np.where(df_2013['Player_Id'] == df_2013['Purple_Cap_Id'],1,0)
df_2014['purple'] = np.where(df_2014['Player_Id'] == df_2014['Purple_Cap_Id'],1,0)
df_2015['purple'] = np.where(df_2015['Player_Id'] == df_2015['Purple_Cap_Id'],1,0)
df_2016['purple'] = np.where(df_2016['Player_Id'] == df_2016['Purple_Cap_Id'],1,0)

df_2010.head(3)
df_2010_series = df_2010.loc[(df_2010['series'] == 1)]
df_2011_series = df_2011.loc[(df_2011['series'] == 1)]
df_2012_series = df_2012.loc[(df_2012['series'] == 1)]
df_2013_series = df_2013.loc[(df_2013['series'] == 1)]
df_2014_series = df_2014.loc[(df_2014['series'] == 1)]
df_2015_series = df_2015.loc[(df_2015['series'] == 1)]
df_2016_series = df_2016.loc[(df_2016['series'] == 1)]

df_2010_series['Player_Name'].unique()
name_df = [df_2010_series,df_2011_series,df_2012_series,df_2013_series,df_2014_series,df_2015_series,df_2016_series]
name_list = []
for i in name_df:
    name_list.append(i['Player_Name'].unique())
    
year_list = [x for x in range(2010,2017)]
year_list
df_2010_orange = df_2010.loc[(df_2010['orange'] == 1)]
df_2011_orange = df_2011.loc[(df_2011['orange'] == 1)]
df_2012_orange = df_2012.loc[(df_2012['orange'] == 1)]
df_2013_orange = df_2013.loc[(df_2013['orange'] == 1)]
df_2014_orange = df_2014.loc[(df_2014['orange'] == 1)]
df_2015_orange = df_2015.loc[(df_2015['orange'] == 1)]
df_2016_orange = df_2016.loc[(df_2016['orange'] == 1)]

orange_df = [df_2010_orange,df_2011_orange,df_2012_orange,df_2013_orange,df_2014_orange,df_2015_orange,df_2016_orange]
df_2010_orange['Batsman_Scored'].unique()
df_2010_orange = df_2010_orange.loc[(df_2010_orange['Batsman_Scored']!=' ')]
df_2010_orange['Batsman_Scored'].unique()
df_2011_orange['Batsman_Scored'].unique()
for i in orange_df:
    print(i['Batsman_Scored'].unique())
    print("="*100)
df_2010_orange = df_2010_orange.loc[(df_2010_orange['Batsman_Scored']!=' ')]
df_2010_orange['Batsman_Scored'].unique()
df_2016_orange = df_2016_orange.loc[(df_2016_orange['Batsman_Scored']!='Do_nothing')]
df_2016_orange['Batsman_Scored'].unique()
df_2016_orange = df_2016_orange.loc[(df_2016_orange['Batsman_Scored']!=' ')]
df_2016_orange['Batsman_Scored'].unique()
df_2011_orange = df_2011_orange.loc[(df_2011_orange['Batsman_Scored']!='Do_nothing')]
df_2012_orange = df_2012_orange.loc[(df_2012_orange['Batsman_Scored']!='Do_nothing')]
orange_df = [df_2010_orange,df_2011_orange,df_2012_orange,df_2013_orange,df_2014_orange,df_2015_orange,df_2016_orange]
for i in orange_df:
    print(i['Batsman_Scored'].unique())
    print("="*100)
for i in orange_df:
    i['Batsman_Scored'] = i['Batsman_Scored'].astype('int64')
df_2010_orange = df_2010_orange.loc[(df_2010_orange['Striker_Id'] == df_2010_orange['Player_Id'])]
df_2011_orange = df_2011_orange.loc[(df_2011_orange['Striker_Id'] == df_2011_orange['Player_Id'])]
df_2012_orange = df_2012_orange.loc[(df_2012_orange['Striker_Id'] == df_2012_orange['Player_Id'])]
df_2013_orange = df_2013_orange.loc[(df_2013_orange['Striker_Id'] == df_2013_orange['Player_Id'])]
df_2014_orange = df_2014_orange.loc[(df_2014_orange['Striker_Id'] == df_2014_orange['Player_Id'])]
df_2015_orange = df_2015_orange.loc[(df_2015_orange['Striker_Id'] == df_2015_orange['Player_Id'])]
df_2016_orange = df_2016_orange.loc[(df_2016_orange['Striker_Id'] == df_2016_orange['Player_Id'])]
orange_df = [df_2010_orange,df_2011_orange,df_2012_orange,df_2013_orange,df_2014_orange,df_2015_orange,df_2016_orange]
for i in orange_df:
    print(i['Striker_Id'].unique())
    print("="*100)
df_2010_orange4and6 = df_2010_orange.loc[(df_2010_orange['Batsman_Scored'] == 4) | (df_2010_orange['Batsman_Scored'] == 6)]
df_2011_orange4and6 = df_2011_orange.loc[(df_2011_orange['Batsman_Scored'] == 4) | (df_2011_orange['Batsman_Scored'] == 6)]
df_2012_orange4and6 = df_2012_orange.loc[(df_2012_orange['Batsman_Scored'] == 4) | (df_2012_orange['Batsman_Scored'] == 6)]
df_2013_orange4and6 = df_2013_orange.loc[(df_2013_orange['Batsman_Scored'] == 4) | (df_2013_orange['Batsman_Scored'] == 6)]
df_2014_orange4and6 = df_2014_orange.loc[(df_2014_orange['Batsman_Scored'] == 4) | (df_2014_orange['Batsman_Scored'] == 6)]
df_2015_orange4and6 = df_2015_orange.loc[(df_2015_orange['Batsman_Scored'] == 4) | (df_2015_orange['Batsman_Scored'] == 6)]
df_2016_orange4and6 = df_2016_orange.loc[(df_2016_orange['Batsman_Scored'] == 4) | (df_2016_orange['Batsman_Scored'] == 6)]
df4and6 = [df_2010_orange4and6,df_2011_orange4and6,df_2012_orange4and6,df_2013_orange4and6,df_2014_orange4and6,df_2015_orange4and6,df_2016_orange4and6]
name_list = []
for i in orange_df:
    name_list.append(i['Player_Name'].unique())
def countplots(df):
    plt.figure(figsize=(16,9))
    sn.countplot(data=df,x = df['Batsman_Scored'],palette='plasma')
    plt.show()
for i,j in zip(orange_df,name_list):
    print(f' player <{j}> has scored these many runs ↓')
    countplots(i)
    print("="*75)

for i,j in zip(df4and6,name_list):
    print(f' player <{j}> has scored these many runs in boundaries ↓')
    countplots(i)
    print("="*75)
total_runs = []
for i in orange_df:
    total_runs.append(i['Batsman_Scored'].sum())
run_map = dict(zip(total_runs,name_list))
run_map
boundary = []
for i in df4and6:
    boundary.append(i['Batsman_Scored'].sum())
boundary_map = dict(zip(boundary,name_list))
boundary_map
top_df = pd.DataFrame(name_list,columns=['Player_Name'])
top_df
total_runs = np.array(total_runs)
boundary =  np.array(boundary)
top_df['Season_Total_runs'] = total_runs
top_df['Runs_scored_in_boundries'] = boundary
top_df
team_name = []
for i in orange_df:
    team_name.append(i['Team_Name'].unique())
team_name = np.array(team_name)
top_df['Batsman_team'] = team_name
top_df
player_id = df['Player_Id'].unique()
player_id
player_name = df['Player_Name'].unique()
player_name
player_map = dict(zip(player_id,player_name))
player_map
bowler10 = df_2010_orange['Bowler_Id'].unique()
bowler10
random = [x for x in range(0,56)]
random = random[1:]
len(random)
bowler10 = dict(zip(bowler10,random))
bowler10
value_list = []
for key,values in player_map.items():
    if key in bowler10.keys():
        value_list.append(values)
        
value_list
bowlerkeys = bowler10.keys()
bowlermap = dict(zip(bowlerkeys,value_list))
bowlermap
df_2010_orange['bowler_name'] = df_2010_orange['Bowler_Id'].map(bowlermap)
df_2010_orange.head(2)
px.bar(data_frame=df_2010_orange,x = 'bowler_name',y = 'Batsman_Scored',color_discrete_sequence=['magenta'],opacity=1)
def bowlerid(df):
    bowler = []
    bowler.append(df['Bowler_Id'].unique())
    return bowler
bowler11 = []
bowler12 = []
bowler13 = []
bowler14 = []
bowler15 = []
bowler16 = []
bowler11=df_2011_orange['Bowler_Id'].unique()
bowler12=df_2012_orange['Bowler_Id'].unique()
bowler13=df_2013_orange['Bowler_Id'].unique()
bowler14=df_2014_orange['Bowler_Id'].unique()
bowler15=df_2015_orange['Bowler_Id'].unique()
bowler16=df_2016_orange['Bowler_Id'].unique()
def counts(lists):
    dumm = [x for x in range(0,len(bowler11))]
    return dumm
count11 = counts(bowler11)
count12 = counts(bowler12)
count13 = counts(bowler13)
count14 = counts(bowler14)
count15 = counts(bowler15)
count16 = counts(bowler16)
def dummydict(list1,list2):
    dummy = dict(zip(list1,list2))
    return dummy
    
dummy11 = dummydict(bowler11,count11)
dummy12 = dummydict(bowler12,count12)
dummy13 = dummydict(bowler13,count13)
dummy14 = dummydict(bowler14,count14)
dummy15 = dummydict(bowler15,count15)
dummy16 = dummydict(bowler16,count16)

def maps(dummy):
    values_list = []
    for key,value in player_map.items():
        if key in dummy.keys():
            values_list.append(value)
    return values_list
map11 = maps(dummy11)
map12 = maps(dummy12)
map13 = maps(dummy13)
map14 = maps(dummy14)
map15 = maps(dummy15)
map16 = maps(dummy16)
bowler11map = dummydict(bowler11,map11)
bowler12map = dummydict(bowler12,map12)
bowler13map = dummydict(bowler13,map13)
bowler14map = dummydict(bowler14,map14)
bowler15map = dummydict(bowler15,map15)
bowler16map = dummydict(bowler16,map16)

bowlerlist = [bowler11map,bowler12map,bowler13map,bowler14map,bowler15map,bowler16map]
orangedf =  [df_2011_orange,df_2012_orange,df_2013_orange,df_2014_orange,df_2015_orange,df_2016_orange]
for i,j in zip(orangedf,bowlerlist):
    i['Bowler_name'] = i['Bowler_Id'].map(j)
df_2011_orange.head(3)
df_2010_orange = df_2010_orange.rename(columns={'bowler_name':'Bowler_name'})
df_2010_orange.columns
def scoreplots(df):
    fig = px.bar(data_frame=df,x = 'Bowler_name',y = 'Batsman_Scored',color_discrete_sequence=['magenta'],opacity=1)
    fig.show()
orange_df = [df_2010_orange,df_2011_orange,df_2012_orange,df_2013_orange,df_2014_orange,df_2015_orange,df_2016_orange]
for i,j,k in zip(orange_df,name_list,year_list):
    print(f' batsman <{j}> scores against bowlers in the season <{k}> is shown below ↓')
    scoreplots(i)
    print("="*75)
from plotly.offline import iplot
import plotly.graph_objects as go 
def charts(df):
    chart = px.pie(df,values = 'Batsman_Scored',names = 'Bowler_name',height = 600)
    chart.update_traces(textposition = 'inside',textinfo = 'percent+label')
    
    chart.update_layout(title_x = 0.5,
                       geo =dict(
        showframe = False,
        showcoastlines = False,
    ))
    
    chart.show()
for i,j,k in zip(orange_df,name_list,year_list):
    print(f' batsman <{j}> scores against bowlers in the season <{k}> is shown below ↓')
    charts(i)
    print("="*75)
series_lists = [df_2010_series,df_2011_series,df_2012_series,df_2013_series,df_2014_series,df_2015_series,df_2016_series]
series_man = []
for i in series_lists:
    series_man.append(i['Player_Name'].unique())
series_man = np.array(series_man)
top_df['Man_of_the_Series'] = series_man
top_df
top_df = top_df.rename(columns={'Player_Name':'Orange_Cap_Owner'})
top_df
top_df['Season_year'] = year_list
top_df


df_2010_purple = df_2010.loc[(df_2010['purple'] == 1)]
df_2011_purple = df_2011.loc[(df_2011['purple'] == 1)]
df_2012_purple = df_2012.loc[(df_2012['purple'] == 1)]
df_2013_purple = df_2013.loc[(df_2013['purple'] == 1)]
df_2014_purple = df_2014.loc[(df_2014['purple'] == 1)]
df_2015_purple = df_2015.loc[(df_2015['purple'] == 1)]
df_2016_purple = df_2016.loc[(df_2016['purple'] == 1)]

purple_df = [df_2010_purple,df_2011_purple,df_2012_purple,df_2013_purple,df_2014_purple,df_2015_purple,df_2016_purple]
for i in purple_df:
    print(i['Dissimal_Type'].unique())
    print("="*100)
df_2010_purple = df_2010_purple.loc[(df_2010_purple['Dissimal_Type']!= ' ')]
df_2010_purple = df_2010_purple.loc[(df_2010_purple['Dissimal_Type']!='run out')]
df_2010_purple = df_2010_purple.loc[ (df_2010_purple['Dissimal_Type']!= 'retired hurt')]
df_2010_purple['Dissimal_Type'].unique()
df_2011_purple = df_2011_purple.loc[(df_2011_purple['Dissimal_Type']!= ' ')]
df_2012_purple = df_2012_purple.loc[(df_2012_purple['Dissimal_Type']!= ' ')]
df_2013_purple = df_2013_purple.loc[(df_2013_purple['Dissimal_Type']!= ' ')]
df_2014_purple = df_2014_purple.loc[(df_2014_purple['Dissimal_Type']!= ' ')]
df_2015_purple = df_2015_purple.loc[(df_2015_purple['Dissimal_Type']!= ' ')]
df_2016_purple = df_2016_purple.loc[(df_2016_purple['Dissimal_Type']!= ' ')]
df_2011_purple = df_2011_purple.loc[(df_2011_purple['Dissimal_Type']!= 'run out')]
df_2012_purple = df_2012_purple.loc[(df_2012_purple['Dissimal_Type']!= 'run out')]
df_2013_purple = df_2013_purple.loc[(df_2013_purple['Dissimal_Type']!= 'run out')]
df_2014_purple = df_2014_purple.loc[(df_2014_purple['Dissimal_Type']!= 'run out')]
df_2015_purple = df_2015_purple.loc[(df_2015_purple['Dissimal_Type']!= 'run out')]
df_2016_purple = df_2016_purple.loc[(df_2016_purple['Dissimal_Type']!= 'run out')]
purple_df = [df_2010_purple,df_2011_purple,df_2012_purple,df_2013_purple,df_2014_purple,df_2015_purple,df_2016_purple]

for i in purple_df:
    print(i['Dissimal_Type'].unique())
    print("="*100)
purple_cap = []
for i in purple_df:
    purple_cap.append(i['Player_Name'].unique())
purple_cap = np.array(purple_cap)
top_df['Purple_cap_Owner'] = purple_cap
top_df
def bowlcounts(df):
    plt.figure(figsize=(16,9))
    sn.countplot(data=df,x = df['Dissimal_Type'],palette='plasma')
    plt.show()
for i,j in zip(purple_df,purple_cap):
    print(f' stats of <{j}> is shown below ↓')
    bowlcounts(i)
    print("="*75)
df_2010_purple = df_2010_purple.loc[(df_2010_purple['Bowler_Id'] == df_2010_purple['Player_Id'])]
df_2011_purple = df_2011_purple.loc[(df_2011_purple['Bowler_Id'] == df_2011_purple['Player_Id'])]
df_2012_purple = df_2012_purple.loc[(df_2012_purple['Bowler_Id'] == df_2012_purple['Player_Id'])]
df_2013_purple = df_2013_purple.loc[(df_2013_purple['Bowler_Id'] == df_2013_purple['Player_Id'])]
df_2014_purple = df_2014_purple.loc[(df_2014_purple['Bowler_Id'] == df_2014_purple['Player_Id'])]
df_2015_purple = df_2015_purple.loc[(df_2015_purple['Bowler_Id'] == df_2015_purple['Player_Id'])]
df_2016_purple = df_2016_purple.loc[(df_2016_purple['Bowler_Id'] == df_2016_purple['Player_Id'])]
purple_df = [df_2010_purple,df_2011_purple,df_2012_purple,df_2013_purple,df_2014_purple,df_2015_purple,df_2016_purple]
for i,j,k in zip(purple_df,purple_cap,year_list):
    print(f' bowler <{j}> has taken these many wickets in the season <{k}> ↓')
    bowlcounts(i)
    print("="*75)
bowler_team = []
for i in purple_df:
    bowler_team.append(i['Team_Name'].unique())
bowler_team = np.array(bowler_team)
top_df['Bowler_Team'] = bowler_team
top_df
bowl10 = []
bowl11 = []
bowl12 = []
bowl13 = []
bowl14 = []
bowl15 = []
bowl16 = []
bowl10 = df_2010_purple['Dissimal_Type']
bowl11 = df_2011_purple['Dissimal_Type']
bowl12 = df_2012_purple['Dissimal_Type']
bowl13 = df_2013_purple['Dissimal_Type']
bowl14 = df_2014_purple['Dissimal_Type']
bowl15 = df_2015_purple['Dissimal_Type']
bowl16 = df_2016_purple['Dissimal_Type']
bowl_stats = [bowl10,bowl11,bowl12,bowl13,bowl14,bowl15,bowl16]
wickets_list = []
for i in bowl_stats:
    wickets_list.append(len(i))
wickets_list[2] = 23
wickets_list
wickets_list = np.array(wickets_list)
top_df['Wickets_taken_by_Purple_cap'] = wickets_list
top_df
px.bar(data_frame=top_df,x = 'Orange_Cap_Owner',y = 'Season_Total_runs',color_discrete_sequence=['orange'],opacity=1)
px.bar(data_frame=top_df,x = 'Purple_cap_Owner',y = 'Wickets_taken_by_Purple_cap',color_discrete_sequence=['magenta'],opacity=1)
