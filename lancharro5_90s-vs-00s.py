# ms-python.python added

import os

try:

	os.chdir(os.path.join(os.getcwd(), 'Kaggle\\NBA Players Stats since 1950'))

	print(os.getcwd())

except:

	pass



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import plotly

import plotly.offline as pyo

import plotly.graph_objs as go

pyo.init_notebook_mode(connected=True)
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

seasons_stats = pd.read_csv("../input/seasons-stats-50-19/Seasons_stats_complete.csv")

#remove duplicated players who played in more than 1 team in a year and keep the TOT

duplic = seasons_stats[seasons_stats.duplicated(['Player','Year'],keep='first')]

indexx = list(duplic.index)

seasons_stats = seasons_stats.drop(indexx)
# Function to create 'xGame' column, to calculate the average for stats per game

# x == DataFrame ; y == Column



def pergame(x, y):

    x[y + 'xG'] = round(x[y]/x['G'], 2)



pergame(seasons_stats, 'ORB')

pergame(seasons_stats, 'DRB')

pergame(seasons_stats, 'TRB')

pergame(seasons_stats, 'AST')

pergame(seasons_stats, 'STL')

pergame(seasons_stats, 'BLK')

pergame(seasons_stats, 'TOV')

pergame(seasons_stats,'PF')

pergame(seasons_stats,'2PA')

pergame(seasons_stats,'3P')

pergame(seasons_stats,'3PA')

pergame(seasons_stats,'FGA')

pergame(seasons_stats, 'PTS')



seasons_stats.columns





def duel_decades(x,y1,y2,z1,z2):

    # x = dataframe ; y1 = first year decade1(yyyy) ; y2 = first year decade2 (yyyy) ; z1 = main attribute ; z2 = graph attribute 

    x_y1 = x.loc[(x['Year'] < (y1+10)) & (x['Year'] > (y1-1))]

    x_y1 = x_y1.assign(decade='('+str(y1)+')')

    x_y1['Player'] = x_y1['Player'] + x_y1['decade']

    x_y1 = x_y1.drop('decade', axis=1) 

    x_y1_mean = round(x_y1.groupby('Player').mean(), 2)

    x_y1_sum =  round(x_y1.groupby('Player').sum(), 2)

    x_y1_mean[z1+'tot'] = x_y1_sum[z1]

    x_y1_mean = x_y1_mean.drop([str(z1)], axis=1)

    x_y1_mean_top = x_y1_mean.sort_values(z1+'tot', ascending=False).iloc[:30]

    x_y1_mean_top_z2 = x_y1_mean_top.sort_values(z2, ascending=False).iloc[:10]

    colors_y1 = ('#000096', '#000096', '#000096', '#1B00A1', '#3600A1', '#5000A1', '#6B00A1', '#8600A1', '#A100A1', '#A10086')

    top10_z2_y1 = go.Bar(x=x_y1_mean_top_z2.index,y=x_y1_mean_top_z2[str(z2)],name='Total'+str(z2),marker=dict(color=colors_y1))

    data_y1=[top10_z2_y1]

    layout_y1=go.Layout(title='Top10 Best '+str(z2)+' '+str(y1), xaxis=dict(title='Player', titlefont=dict(size=16, color='black'), tickfont=dict(size=10, color='#0080ff'), showline=True, linewidth=1, linecolor='black'),	yaxis=dict(title='Total '+str(z2), titlefont=dict(size=16, color='black'), tickfont=dict(size=14, color='#0080ff'), showgrid=True, gridwidth=0.2, gridcolor='#D7DBDD',showline=True, linewidth=1, linecolor='black'), plot_bgcolor='white', barmode='group', bargap=0.6, bargroupgap=0.1)

    fig_y1 = go.Figure(data=data_y1, layout=layout_y1)

    x_y2 = x.loc[(x['Year'] < (y2+10)) & (x['Year'] > (y2-1))]

    x_y2 = x_y2.assign(decade='('+str(y2)+')')

    x_y2['Player'] = x_y2['Player'] + x_y2['decade']

    x_y2 = x_y2.drop('decade', axis=1)

    x_y2_mean = round(x_y2.groupby('Player').mean(), 2)

    x_y2_sum =  round(x_y2.groupby('Player').sum(), 2)

    x_y2_mean[z1+'tot'] = x_y2_sum[z1]

    x_y2_mean = x_y2_mean.drop([str(z1)], axis=1)

    x_y2_mean_top = x_y2_mean.sort_values(z1+'tot', ascending=False).iloc[:30]

    x_y2_mean_top_z2 = x_y2_mean_top.sort_values(z2, ascending=False).iloc[:10]

    colors_y2 = ('#E41B17', '#E41B17', '#E41B17', '#E33C17', '#E35E17', '#E38017', '#E3A217', '#E3C417', '#E0E317', '#BEE317')

    top10_z2_y2 = go.Bar(x=x_y2_mean_top_z2.index,y=x_y2_mean_top_z2[str(z2)],name='Total'+str(z2),marker=dict(color=colors_y2))

    data_y2=[top10_z2_y2]

    layout_y2=go.Layout(title='Top10 Best '+str(z2)+' '+str(y2), xaxis=dict(title='Player', titlefont=dict(size=16, color='black'), tickfont=dict(size=10, color='#0080ff'), showline=True, linewidth=1, linecolor='black'),	yaxis=dict(title='Total '+str(z2), titlefont=dict(size=16, color='black'), tickfont=dict(size=14, color='#0080ff'), showgrid=True, gridwidth=0.2, gridcolor='#D7DBDD',showline=True, linewidth=1, linecolor='black'), plot_bgcolor='white', barmode='group', bargap=0.6, bargroupgap=0.1)

    fig_y2 = go.Figure(data=data_y2, layout=layout_y2)

    z2_mean_top10 = x_y1_mean_top_z2

    z2_y2_mean_top10 = x_y2_mean_top_z2

    z2_mean_top10 = z2_mean_top10.append(z2_y2_mean_top10)

    z2_mean_top10 = z2_mean_top10.sort_values(z2, ascending=False).iloc[:10]

    colors3 = ('#006747', '#006747', '#009639', '#009639', '#78be20', '#78be20', '#00a499', '#00a499', '#404e4d', '#404e4d')

    top10_z2_y1_y2_comp = go.Bar(x=z2_mean_top10.index,y=z2_mean_top10[z2],name=z2,marker=dict(color=colors3))

    data3=[top10_z2_y1_y2_comp]

    layout3=go.Layout(title='Top10 Best '+z2+' '+str(y1)+' vs '+str(y2), xaxis=dict(title='Player',titlefont=dict(size=16, color='black'), tickfont=dict(size=10, color='#0080ff'), showline=True, linewidth=1, linecolor='black'), yaxis=dict(title='Total '+z2,titlefont=dict(size=16, color='black'), tickfont=dict(size=14, color='#0080ff'), showgrid=True, gridwidth=0.2, gridcolor='#D7DBDD',showline=True, linewidth=1, linecolor='black'),plot_bgcolor='white', barmode='group', bargap=0.6, bargroupgap=0.1)

    fig3 = go.Figure(data=data3, layout=layout3)

    print (pyo.iplot(fig_y1), pyo.iplot(fig_y2), pyo.iplot(fig3))

   



rebs = seasons_stats[['Year', 'Player', 'Pos', 'Age', 'Tm', 'G', 'PER', 'USG%', 'ORB%', 'DRB%', 'TRB%', 'ORB', 'DRB', 'TRB', 'ORBxG', 'DRBxG', 'TRBxG', 'PTSxG']]

duel_decades(rebs,1990,2000,'TRB','TRBxG')

ast = seasons_stats[['Year', 'Player', 'Pos', 'Age', 'Tm', 'G', 'PER', 'USG%', 'AST%', 'AST', 'ASTxG', 'PTSxG']]



duel_decades(ast,1990,2000,'AST','ASTxG')

pts = seasons_stats[['Year', 'Player', 'Pos', 'Age', 'Tm', 'G', 'PER', 'USG%', 'PTS', 'PTSxG']]



duel_decades(pts,1990,2000,'PTS','PTSxG')

blk = seasons_stats[['Year', 'Player', 'Pos', 'Age', 'Tm', 'G', 'PER', 'USG%', 'BLK', 'BLKxG']]



duel_decades(blk,1990,2000,'BLK','BLKxG')

stl = seasons_stats[['Year', 'Player', 'Pos', 'Age', 'Tm', 'G', 'PER', 'USG%', 'STL', 'STLxG']]



duel_decades(stl,1990,2000,'STL','STLxG')

shot3p = seasons_stats[['Year', 'Player', 'Pos', 'Age', 'Tm', 'G', 'PER', 'USG%', '3P', '3PxG', '3P%']]



duel_decades(shot3p,1990,2000,'3P','3PxG')

duel_decades(shot3p,1990,2000,'3P','3P%')

per = seasons_stats[['Year', 'Player', 'Pos', 'Age', 'Tm', 'G', 'PER', 'PTS', 'USG%']]



duel_decades(per,1990,2000, 'PTS', 'PER')


