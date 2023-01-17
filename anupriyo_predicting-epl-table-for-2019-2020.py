# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

epl=pd.read_csv('/kaggle/input/epl-stats-20192020/epl2020.csv')

epl.head()

epl.columns

epl=epl[['date','teamId','scored','missed','xpts','result','wins','loses','draws','pts','tot_points','tot_goal','tot_con','round']]

epl.head()

epl['wins']

#epl[epl['teamId']=='Liverpool']['tot_goal']

#How does the current team looks like

teams=epl['teamId'].unique()

teams

team_list=[]

position=np.array(range(1,21))

for i in teams:

    team_data=epl[epl['teamId']==i]

    

    team_win=team_data['wins'].sum()

    team_loses=team_data['loses'].sum()

    team_draw=team_data['draws'].sum()

    team_points=team_data['tot_points'].max()

    team_goals=team_data['tot_goal'].max()

    team_conceded=team_data['tot_con'].max()

    team_match=team_data['round'].max()

    goal_diff=team_goals-team_conceded

    team_list.append([i,team_match,team_win,team_draw,team_loses,team_points,team_goals,team_conceded,goal_diff])

table=pd.DataFrame(team_list,columns=['Club','P','W','D','L','Points','GF','GA','GD'])   

table=table.tail(20)

table.sort_values(by=['Points','GD','GA'],ascending=False,inplace=True)

table

table.set_index(position,inplace=True)

table

#We wold find recent form of last 5 games for each team and calculate the points taken

recent_form=[]

for i in teams:

    team_datas=epl[epl['teamId']==i].tail(5)

    wins=team_datas['wins'].sum()

    loses=team_datas['loses'].sum()

    draws=team_datas['draws'].sum()

    point=(3*wins)+draws

    team_goals=team_datas['tot_goal'].max()

    team_conceded=team_datas['tot_con'].max()

    goal_diff=team_goals-team_conceded

    recent_form.append([i,point,goal_diff])

    recent_form.sort(key=lambda x:x[1])

recent_form  

team_points=[]

for team in teams:

    points_per_game=[x for x in recent_form if x[0]==team][0][1]/5

    team_data=table[table['Club']==team]

    games_left=38-team_data['P']

    new_points=int(team_data['Points']+round(points_per_game*games_left))

    team_points.append([team,new_points])

team_points    

predicted_table=pd.DataFrame(team_points,columns=['Club','NewPoints'])

predicted_table.sort_values(by='NewPoints',ascending=False,inplace=True)

predicted_table

predicted_table.set_index(position,inplace=True)

predicted_table

    # Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

os.listdir('../input')

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session