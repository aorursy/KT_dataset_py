# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


nbadata = pd.read_csv("../input/nba-all-star-game-20002016/NBA All Stars 2000-2016 - Sheet1.csv")



nbadata = pd.DataFrame(data = nbadata)







nbadata.head()
nbadata['Conference'] = np.where(nbadata['Selection Type'].str.startswith("Eastern"),"East","West")



nbadata
nbadata['Conference'].unique()
#nbadata = nbadata.reset_index()



#nbadata = nbadata.columns = ['Year','Player','Pos','HT','WT','Team','Selection Type','NBA Draft Status','Nationality','Conference']



type(nbadata)

nbadata.describe()
nbadata.info()
team_all = nbadata['Team'].value_counts()



team_counts = team_all.reset_index()



team_counts.columns=['Team','Count']



team_counts

import plotly.express as px



fig = px.bar(team_counts,x='Team',y='Count')



fig.show()
import plotly.express as px



teams = nbadata['Team']



fig = px.histogram(teams, x="Team")



fig.show()



player_nation = nbadata['Nationality'].value_counts()



nation_count = player_nation.reset_index()



nation_count.columns=['Nationality','Count']



nation_count



fig = px.bar(nation_count,x='Nationality',y='Count')



fig.show()
team_weight = nbadata.loc[:,['Team','WT']]



team_weight = team_weight.groupby('Team').mean().sort_values(by=['WT'],ascending=False,inplace=False)



team_weight = team_weight.reset_index()



fig = px.bar(team_weight,x='Team',y='WT')



fig.show()
weight_pos = nbadata.loc[:,['WT','Pos']]



weight_pos = weight_pos.groupby('Pos').mean().sort_values(by=['WT'],ascending=False,inplace=False).reset_index()





fig = px.bar(weight_pos,x='Pos',y='WT')



fig.show()



import plotly.graph_objects as go









team_years = nbadata.loc[:,['Year','Team','Conference']]





team_years_east = team_years[team_years['Conference'].str.match('East')]



team_years_west = team_years[team_years['Conference'].str.match('West')]







team_years_east = team_years_east.groupby('Year')['Team'].value_counts()



team_years_west = team_years_west.groupby('Year')['Team'].value_counts()





team_years_east = pd.DataFrame(data=team_years_east)



team_years_west = pd.DataFrame(data=team_years_west)





team_years_east
team_years_east = team_years_east.rename(columns={'Team': 'Count'})

team_years_west = team_years_west.rename(columns={'Team': 'Count'})

team_years_west = team_years_west.reset_index(level='Team')

team_years_east = team_years_east.reset_index(level='Team')









team_years_east
team_years_east = team_years_east.reset_index(level='Year')



team_years_east
fig = px.scatter(team_years_east,x=team_years_east['Year'],y=team_years_east['Team'],size=(team_years_east.Count),color=team_years_east['Count'])





fig.show()
team_years_west = team_years_west.reset_index(level='Year')



team_years_west
fig = px.scatter(team_years_west,x=team_years_west['Year'],y=team_years_west['Team'],size=(team_years_west.Count),color=team_years_west['Count'])





fig.show()
top_west = team_years_west.loc[team_years_west['Count'] >= 2]

top_east = team_years_east.loc[team_years_east['Count'] >= 2]

fig = px.scatter(top_west,x=top_west['Year'],y=top_west['Team'],size=(top_west.Count),color=top_west['Count'])





fig.show()
fig = px.scatter(top_east,x=top_east['Year'],y=top_east['Team'],size=(top_east.Count),color=top_east['Count'])





fig.show()