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
seasons_stats = seasons_stats.fillna(0)

seasons_stats = seasons_stats[seasons_stats.Tm != 'TOT']

seasons_stats1980 = seasons_stats.loc[seasons_stats['Year'] > 1979]
# Function to create 'xGame' column, to calculate the average for stats per game

# x == DataFrame ; y == Column



def pergame(x, y):

    x[y + 'xG'] = round(x[y]/x['G'], 2)

seasons_stats3p = seasons_stats.loc[seasons_stats['Year'] > 1979]
seasons_stats3p = seasons_stats3p[['Year', 'Pos', 'Player', 'G', '2PA', '3PA', 'FGA']]

seasons_stats3p_years = seasons_stats3p.groupby('Year').sum()

totyears = seasons_stats3p_years.index



tot2 = go.Bar(

	x=totyears,

	y=seasons_stats3p_years['2PA'],

	name='2 points attempts',

	marker=dict(color='#2828ff')

)

tot3 = go.Bar(

	x=totyears,

	y=seasons_stats3p_years['3PA'],

	name='3 points attempts',

	marker=dict(color='#ff3939')

)

data=[tot2,tot3]

layout=go.Layout(

	title='Total 2p vs 3p',

	xaxis=dict(

		title='Years',

		titlefont=dict(size=16, color='black'),

		tickfont=dict(size=14, color='#0080ff')

	),

	yaxis=dict(

		title='Total attempts',

		titlefont=dict(size=16, color='black'),

		tickfont=dict(size=14, color='#0080ff'),

        showgrid=True, gridwidth=0.2, gridcolor='#D7DBDD'

	),	

	legend=dict(

		x=1,

		y=1.0,

		bgcolor='white',

		bordercolor='white'

	),

    plot_bgcolor='white',

	barmode='group',

	bargap=0.2,

	bargroupgap=0.1

)



fig = go.Figure(data=data, layout=layout)

pyo.iplot(fig)

seasons_stats3p_years['2PA/FGA%'] = round(seasons_stats3p_years['2PA'] / seasons_stats3p_years['FGA'] * 100, 2)

seasons_stats3p_years['3PA/FGA%'] = round(seasons_stats3p_years['3PA'] / seasons_stats3p_years['FGA'] * 100, 2)

perc2 = go.Bar(

	x=totyears,

	y=seasons_stats3p_years['2PA/FGA%'],

	name='2 points attempts %',

	marker=dict(color='#2828ff')

)

perc3 = go.Bar(

	x=totyears,

	y=seasons_stats3p_years['3PA/FGA%'],

	name='3 points attempts %',

	marker=dict(color='#ff3939')

)

data=[perc2,perc3]

layout=go.Layout(

	title='2p% vs 3p%',

	xaxis=dict(

		title='Years',

		titlefont=dict(size=16, color='black'),

		tickfont=dict(size=14, color='#0080ff')

	),

	yaxis=dict(

		title='Total attempts',

		titlefont=dict(size=16, color='black'),

		tickfont=dict(size=14, color='#0080ff'),

        showgrid=True, gridwidth=0.2, gridcolor='#D7DBDD'

	),	

	legend=dict(

		x=1,

		y=1.0,

		bgcolor='white',

		bordercolor='white'

	),

    plot_bgcolor='white',

	barmode='group',

	bargap=0.2,

	bargroupgap=0.1

)



fig = go.Figure(data=data, layout=layout)

pyo.iplot(fig)

seasons_stats3p_game = seasons_stats[['Year', 'Player', 'Pos', 'G', 'MP', '2PA', '3PA', 'FGA', 'eFG%']]



# filtered by player who played more than 24 games in a season

seasons_stats3p_game = seasons_stats3p_game.loc[(seasons_stats3p_game['Year'] > 1979) & (seasons_stats3p_game['G'] > 24)]



#MinutsxGame column and filter by  > 17 MPxG

pergame(seasons_stats3p_game, 'MP')



seasons_stats3p_game = seasons_stats3p_game.loc[seasons_stats3p_game['MPxG'] > 17]



#Attemps per game

pergame(seasons_stats3p_game, '2PA')

pergame(seasons_stats3p_game, '3PA')

pergame(seasons_stats3p_game, 'FGA')







seasons_stats3p_game['eFG%'] = round(seasons_stats3p_game['eFG%'] * 100, 2)

shots_game_c = round(seasons_stats3p_game.loc[seasons_stats3p_game['Pos'] == 'C'].groupby('Year').mean(),2)

shots_game_pf = round(seasons_stats3p_game.loc[seasons_stats3p_game['Pos'] == 'PF'].groupby('Year').mean(),2)

shots_game_sf = round(seasons_stats3p_game.loc[seasons_stats3p_game['Pos'] == 'SF'].groupby('Year').mean(),2)

shots_game_sg = round(seasons_stats3p_game.loc[seasons_stats3p_game['Pos'] == 'SG'].groupby('Year').mean(),2)

shots_game_pg = round(seasons_stats3p_game.loc[seasons_stats3p_game['Pos'] == 'PG'].groupby('Year').mean(),2)

shots_game_gen = round(seasons_stats3p_game.groupby('Year').mean(),2)



years_shotsgame = shots_game_c.index

p2c = go.Scatter(x=years_shotsgame, y=shots_game_c['2PAxG'], name='Centers', marker=dict(color='blue'))

p2pf = go.Scatter(x=years_shotsgame, y=shots_game_pf['2PAxG'], name='PowerForwards', marker=dict(color='red'))

p2sf = go.Scatter(x=years_shotsgame, y=shots_game_sf['2PAxG'], name='SmallForwards', marker=dict(color='green'))

p2sg = go.Scatter(x=years_shotsgame, y=shots_game_sg['2PAxG'], name='ShootingGuards', marker=dict(color='purple'))

p2pg = go.Scatter(x=years_shotsgame, y=shots_game_pg['2PAxG'], name='PointGuards', marker=dict(color='orange'))

p2gen = go.Bar(x=years_shotsgame, y=shots_game_gen['2PAxG'], name='Average', marker=dict(color='#FFD5C1'))



data=[p2c,p2pf,p2sf,p2sg,p2pg,p2gen]

layout=go.Layout(

	title='2P attempts per game by position',

	xaxis=dict(

		title='Years',

		titlefont=dict(size=16, color='#000000'),

		tickfont=dict(size=14, color='#000000')

	),

	yaxis=dict(

		title='2 points attempts per game',

		titlefont=dict(size=16, color='#000000'),

		tickfont=dict(size=14, color='#000000'),

        showgrid=True, gridwidth=0.2, gridcolor='#D7DBDD'

	),	

	legend=dict(

		x=1,

		y=1.0,

		bgcolor='white',

		bordercolor='black'

	),

    plot_bgcolor='white',

	barmode='group',

	bargap=0.15,

	bargroupgap=0.1

)



fig = go.Figure(data=data, layout=layout)

pyo.iplot(fig)

p3c = go.Scatter(x=years_shotsgame, y=shots_game_c['3PAxG'], name='Centers', marker=dict(color='blue'))

p3pf = go.Scatter(x=years_shotsgame, y=shots_game_pf['3PAxG'], name='PowerForwards', marker=dict(color='red'))

p3sf = go.Scatter(x=years_shotsgame, y=shots_game_sf['3PAxG'], name='SmallForwards', marker=dict(color='green'))

p3sg = go.Scatter(x=years_shotsgame, y=shots_game_sg['3PAxG'], name='ShootingGuards', marker=dict(color='purple'))

p3pg = go.Scatter(x=years_shotsgame, y=shots_game_pg['3PAxG'], name='PointGuards', marker=dict(color='orange'))

p3gen = go.Bar(x=years_shotsgame, y=shots_game_gen['3PAxG'], name='Average', marker=dict(color='#FFD5C1'))



data=[p3c,p3pf,p3sf,p3sg,p3pg,p3gen]

layout=go.Layout(

	title='3P attempts per game by position',

	xaxis=dict(

		title='Years',

		titlefont=dict(size=16, color='#000000'),

		tickfont=dict(size=14, color='#000000'),

	),

	yaxis=dict(

		title='3 points attempts per game',

		titlefont=dict(size=16, color='#000000'),

		tickfont=dict(size=14, color='#000000'),

        showgrid=True, gridwidth=0.2, gridcolor='#D7DBDD'

	),	

	legend=dict(

		x=1,

		y=1.0,

		bgcolor='white',

		bordercolor='black'

	),

	plot_bgcolor='white',

    barmode='group',

	bargap=0.15,

	bargroupgap=0.1

)



fig = go.Figure(data=data, layout=layout)

pyo.iplot(fig)

FGAc = go.Scatter(x=years_shotsgame, y=shots_game_c['FGAxG'], name='Centers', marker=dict(color='blue'))

FGApf = go.Scatter(x=years_shotsgame, y=shots_game_pf['FGAxG'], name='PowerForwards', marker=dict(color='red'))

FGAsf = go.Scatter(x=years_shotsgame, y=shots_game_sf['FGAxG'], name='SmallForwards', marker=dict(color='green'))

FGAsg = go.Scatter(x=years_shotsgame, y=shots_game_sg['FGAxG'], name='ShootingGuards', marker=dict(color='purple'))

FGApg = go.Scatter(x=years_shotsgame, y=shots_game_pg['FGAxG'], name='PointGuards', marker=dict(color='orange'))

FGAgen = go.Bar(x=years_shotsgame, y=shots_game_gen['FGAxG'], name='Average', marker=dict(color='#FFD5C1'))



data=[FGAc,FGApf,FGAsf,FGAsg,FGApg,FGAgen]

layout=go.Layout(

	title='Shot attempts per game by position',

	xaxis=dict(

		title='Years',

		titlefont=dict(size=16, color='#000000'),

		tickfont=dict(size=14, color='#000000')

	),

	yaxis=dict(

		title='Shot attempts per game',

		titlefont=dict(size=16, color='#000000'),

		tickfont=dict(size=14, color='#000000'),

        showgrid=True, gridwidth=0.2, gridcolor='#D7DBDD'

	),	

	legend=dict(

		x=1,

		y=1,

		bgcolor='white',

		bordercolor='black'

	),

    plot_bgcolor='white',

	barmode='group',

	bargap=0.15,

	bargroupgap=0.1

)



fig = go.Figure(data=data, layout=layout)

pyo.iplot(fig)

eFGc = go.Scatter(x=years_shotsgame, y=shots_game_c['eFG%'], name='Centers', marker=dict(color='blue'))

eFGpf = go.Scatter(x=years_shotsgame, y=shots_game_pf['eFG%'], name='PowerForwards', marker=dict(color='red'))

eFGsf = go.Scatter(x=years_shotsgame, y=shots_game_sf['eFG%'], name='SmallForwards', marker=dict(color='green'))

eFGsg = go.Scatter(x=years_shotsgame, y=shots_game_sg['eFG%'], name='ShootingGuards', marker=dict(color='purple'))

eFGpg = go.Scatter(x=years_shotsgame, y=shots_game_pg['eFG%'], name='PointGuards', marker=dict(color='orange'))

eFGgen = go.Bar(x=years_shotsgame, y=shots_game_gen['eFG%'], name='Average', marker=dict(color='#FFD5C1'))



data=[eFGc,eFGpf,eFGsf,eFGsg,eFGpg,eFGgen]

layout=go.Layout(

	title='Effective Field Goal Percentage by positions',

	xaxis=dict(

		title='Years',

		titlefont=dict(size=16, color='#000000'),

		tickfont=dict(size=14, color='#000000')

	),

	yaxis=dict(

		title='Effective Field Goal Percentage',

		titlefont=dict(size=16, color='#000000'),

		tickfont=dict(size=14, color='#000000'),

        showgrid=True, gridwidth=0.2, gridcolor='#D7DBDD'

	),	

	legend=dict(

		x=1,

		y=1.0,

		bgcolor='white',

		bordercolor='black'

	),

    plot_bgcolor='white',

	barmode='group',

	bargap=0.15,

	bargroupgap=0.1

)



fig = go.Figure(data=data, layout=layout)

pyo.iplot(fig)
main_stats = seasons_stats[['Year', 'Player', 'Tm', 'Pos', 'G', 'MP', 'PER', 'ORB%', 'DRB%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PTS']]

main_stats = main_stats.loc[(main_stats['Year'] > 1979) & (main_stats['G'] > 24)]



#MinutsxGame column and filter by  > 17 MPxG

pergame(main_stats, 'MP')



main_stats = main_stats.loc[main_stats['MPxG'] > 17]

xgame = main_stats[['Year', 'Player', 'Tm', 'Pos', 'G', 'USG%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PTS', 'MPxG']]

pergame(xgame, 'ORB')

pergame(xgame, 'DRB')

pergame(xgame, 'TRB')

pergame(xgame, 'AST')

pergame(xgame, 'STL')

pergame(xgame, 'BLK')

pergame(xgame, 'TOV')

pergame(xgame, 'PTS')



xgame = xgame.drop(['ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PTS'], axis=1)
xgame_c = round(xgame.loc[xgame['Pos'] == 'C'].groupby('Year').mean(),2)

xgame_pf = round(xgame.loc[xgame['Pos'] == 'PF'].groupby('Year').mean(),2)

xgame_sf = round(xgame.loc[xgame['Pos'] == 'SF'].groupby('Year').mean(),2)

xgame_sg = round(xgame.loc[xgame['Pos'] == 'SG'].groupby('Year').mean(),2)

xgame_pg = round(xgame.loc[xgame['Pos'] == 'PG'].groupby('Year').mean(),2)

xgame_gen = round(xgame.groupby('Year').mean(),2)



years_xgame = xgame_c.index
offreb = go.Scatter(x=years_xgame, y=xgame_gen['ORBxG'], name='Offensive Rebounds', marker=dict(color='red'))

defreb = go.Scatter(x=years_xgame, y=xgame_gen['DRBxG'], name='Defensive Rebounds', marker=dict(color='blue'))

totreb = go.Bar(x=years_xgame, y=xgame_gen['TRBxG'], name='Rebounds', marker=dict(color='#FFD5C1'))



data=[offreb,defreb,totreb]

layout=go.Layout(

	title='Rebound per game',

	xaxis=dict(

		title='Years',

		titlefont=dict(size=16, color='#000000'),

		tickfont=dict(size=14, color='#000000')

	),

	yaxis=dict(

		title='Rebounds per game',

		titlefont=dict(size=16, color='#000000'),

		tickfont=dict(size=14, color='#000000'),

        showgrid=True, gridwidth=0.2, gridcolor='#D7DBDD'

	),	

	legend=dict(

		x=1,

		y=1.0,

		bgcolor='white',

		bordercolor='black'

	),

    plot_bgcolor='white',

	barmode='group',

	bargap=0.15,

	bargroupgap=0.1

)



fig = go.Figure(data=data, layout=layout)

pyo.iplot(fig)
rebc = go.Scatter(x=years_xgame, y=xgame_c['TRBxG'], name='Centers', marker=dict(color='blue'))

rebpf = go.Scatter(x=years_xgame, y=xgame_pf['TRBxG'], name='PowerForwards', marker=dict(color='red'))

rebsf = go.Scatter(x=years_xgame, y=xgame_sf['TRBxG'], name='SmallForwards', marker=dict(color='green'))

rebsg = go.Scatter(x=years_xgame, y=xgame_sg['TRBxG'], name='ShootingGuards', marker=dict(color='purple'))

rebpg = go.Scatter(x=years_xgame, y=xgame_pg['TRBxG'], name='PointGuards', marker=dict(color='orange'))

rebgen = go.Bar(x=years_xgame, y=xgame_gen['TRBxG'], name='Average', marker=dict(color='#FFD5C1'))



data=[rebc,rebpf,rebsf,rebsg,rebpg,rebgen]

layout=go.Layout(

	title='Total Rebound per game by position',

	xaxis=dict(

		title='Years',

		titlefont=dict(size=16, color='#000000'),

		tickfont=dict(size=14, color='#000000')

	),

	yaxis=dict(

		title='Total Rebounds per game',

		titlefont=dict(size=16, color='#000000'),

		tickfont=dict(size=14, color='#000000'),

        showgrid=True, gridwidth=0.2, gridcolor='#D7DBDD'

	),	

	legend=dict(

		x=1,

		y=1.0,

		bgcolor='white',

		bordercolor='black'

	),

    plot_bgcolor='white',

	barmode='group',

	bargap=0.15,

	bargroupgap=0.1

)



fig = go.Figure(data=data, layout=layout)

pyo.iplot(fig)
def_rebc = go.Scatter(x=years_xgame, y=xgame_c['DRBxG'], name='Centers', marker=dict(color='blue'))

def_rebpf = go.Scatter(x=years_xgame, y=xgame_pf['DRBxG'], name='PowerForwards', marker=dict(color='red'))

def_rebsf = go.Scatter(x=years_xgame, y=xgame_sf['DRBxG'], name='SmallForwards', marker=dict(color='green'))

def_rebsg = go.Scatter(x=years_xgame, y=xgame_sg['DRBxG'], name='ShootingGuards', marker=dict(color='purple'))

def_rebpg = go.Scatter(x=years_xgame, y=xgame_pg['DRBxG'], name='PointGuards', marker=dict(color='orange'))

def_rebgen = go.Bar(x=years_xgame, y=xgame_gen['DRBxG'], name='Average', marker=dict(color='#FFD5C1'))



data=[def_rebc,def_rebpf,def_rebsf,def_rebsg,def_rebpg,def_rebgen]

layout=go.Layout(

	title='Defensive Rebounds per game by position',

	xaxis=dict(

		title='Years',

		titlefont=dict(size=16, color='#000000'),

		tickfont=dict(size=14, color='#000000')

	),

	yaxis=dict(

		title='Defensive Rebounds per game',

		titlefont=dict(size=16, color='#000000'),

		tickfont=dict(size=14, color='#000000'),

        showgrid=True, gridwidth=0.2, gridcolor='#D7DBDD'

	),	

	legend=dict(

		x=1,

		y=1.0,

		bgcolor='white',

		bordercolor='black'

	),

    plot_bgcolor='white',

	barmode='group',

	bargap=0.15,

	bargroupgap=0.1

)



fig = go.Figure(data=data, layout=layout)

pyo.iplot(fig)
off_rebc = go.Scatter(x=years_xgame, y=xgame_c['ORBxG'], name='Centers', marker=dict(color='blue'))

off_rebpf = go.Scatter(x=years_xgame, y=xgame_pf['ORBxG'], name='PowerForwards', marker=dict(color='red'))

off_rebsf = go.Scatter(x=years_xgame, y=xgame_sf['ORBxG'], name='SmallForwards', marker=dict(color='green'))

off_rebsg = go.Scatter(x=years_xgame, y=xgame_sg['ORBxG'], name='ShootingGuards', marker=dict(color='purple'))

off_rebpg = go.Scatter(x=years_xgame, y=xgame_pg['ORBxG'], name='PointGuards', marker=dict(color='orange'))

off_rebgen = go.Bar(x=years_xgame, y=xgame_gen['ORBxG'], name='Average', marker=dict(color='#FFD5C1'))



data=[off_rebc,off_rebpf,off_rebsf,off_rebsg,off_rebpg,off_rebgen]

layout=go.Layout(

	title='Offensive Rebounds per game by position',

	xaxis=dict(

		title='Years',

		titlefont=dict(size=16, color='#000000'),

		tickfont=dict(size=14, color='#000000')

	),

	yaxis=dict(

		title='Offensive Rebounds per game',

		titlefont=dict(size=16, color='#000000'),

		tickfont=dict(size=14, color='#000000'),

        showgrid=True, gridwidth=0.2, gridcolor='#D7DBDD'

	),	

	legend=dict(

		x=1,

		y=1.0,

		bgcolor='white',

		bordercolor='black'

	),

    plot_bgcolor='white',

	barmode='group',

	bargap=0.15,

	bargroupgap=0.1

)



fig = go.Figure(data=data, layout=layout)

pyo.iplot(fig)
blk_c = go.Scatter(x=years_xgame, y=xgame_c['BLKxG'], name='Centers', marker=dict(color='blue'))

blk_pf = go.Scatter(x=years_xgame, y=xgame_pf['BLKxG'], name='PowerForwards', marker=dict(color='red'))

blk_sf = go.Scatter(x=years_xgame, y=xgame_sf['BLKxG'], name='SmallForwards', marker=dict(color='green'))

blk_sg = go.Scatter(x=years_xgame, y=xgame_sg['BLKxG'], name='ShootingGuards', marker=dict(color='purple'))

blk_pg = go.Scatter(x=years_xgame, y=xgame_pg['BLKxG'], name='PointGuards', marker=dict(color='orange'))

blk_gen = go.Bar(x=years_xgame, y=xgame_gen['BLKxG'], name='Average', marker=dict(color='#FFD5C1'))



data=[blk_c,blk_pf,blk_sf,blk_sg,blk_pg,blk_gen]

layout=go.Layout(

	title='Blocks per game by position',

	xaxis=dict(

		title='Years',

		titlefont=dict(size=16, color='#000000'),

		tickfont=dict(size=14, color='#000000')

	),

	yaxis=dict(

		title='Blocks per game',

		titlefont=dict(size=16, color='#000000'),

		tickfont=dict(size=14, color='#000000'),

        showgrid=True, gridwidth=0.2, gridcolor='#D7DBDD'

	),	

	legend=dict(

		x=1,

		y=1.0,

		bgcolor='white',

		bordercolor='black'

	),

    plot_bgcolor='white',

	barmode='group',

	bargap=0.15,

	bargroupgap=0.1

)



fig = go.Figure(data=data, layout=layout)

pyo.iplot(fig)
ast_c = go.Scatter(x=years_xgame, y=xgame_c['ASTxG'], name='Centers', marker=dict(color='blue'))

ast_pf = go.Scatter(x=years_xgame, y=xgame_pf['ASTxG'], name='PowerForwards', marker=dict(color='red'))

ast_sf = go.Scatter(x=years_xgame, y=xgame_sf['ASTxG'], name='SmallForwards', marker=dict(color='green'))

ast_sg = go.Scatter(x=years_xgame, y=xgame_sg['ASTxG'], name='ShootingGuards', marker=dict(color='purple'))

ast_pg = go.Scatter(x=years_xgame, y=xgame_pg['ASTxG'], name='PointGuards', marker=dict(color='orange'))

ast_gen = go.Bar(x=years_xgame, y=xgame_gen['ASTxG'], name='Average', marker=dict(color='#FFD5C1'))



data=[ast_c,ast_pf,ast_sf,ast_sg,ast_pg,ast_gen]

layout=go.Layout(

	title='Assists per game by position',

	xaxis=dict(

		title='Years',

		titlefont=dict(size=16, color='#000000'),

		tickfont=dict(size=14, color='#000000')

	),

	yaxis=dict(

		title='Assists per game',

		titlefont=dict(size=16, color='#000000'),

		tickfont=dict(size=14, color='#000000'),

        showgrid=True, gridwidth=0.2, gridcolor='#D7DBDD'

	),	

	legend=dict(

		x=1,

		y=1.0,

		bgcolor='white',

		bordercolor='black'

	),

    plot_bgcolor='white',

	barmode='group',

	bargap=0.15,

	bargroupgap=0.1

)



fig = go.Figure(data=data, layout=layout)

pyo.iplot(fig)
pts_c = go.Scatter(x=years_xgame, y=xgame_c['PTSxG'], name='Centers', marker=dict(color='blue'))

pts_pf = go.Scatter(x=years_xgame, y=xgame_pf['PTSxG'], name='PowerForwards', marker=dict(color='red'))

pts_sf = go.Scatter(x=years_xgame, y=xgame_sf['PTSxG'], name='SmallForwards', marker=dict(color='green'))

pts_sg = go.Scatter(x=years_xgame, y=xgame_sg['PTSxG'], name='ShootingGuards', marker=dict(color='purple'))

pts_pg = go.Scatter(x=years_xgame, y=xgame_pg['PTSxG'], name='PointGuards', marker=dict(color='orange'))

pts_gen = go.Bar(x=years_xgame, y=xgame_gen['PTSxG'], name='Average', marker=dict(color='#FFD5C1'))



data=[pts_c,pts_pf,pts_sf,pts_sg,pts_pg,pts_gen]

layout=go.Layout(

	title='Points per game by position',

	xaxis=dict(

		title='Years',

		titlefont=dict(size=16, color='#000000'),

		tickfont=dict(size=14, color='#000000')

	),

	yaxis=dict(

		title='Points per game',

		titlefont=dict(size=16, color='#000000'),

		tickfont=dict(size=14, color='#000000'),

        showgrid=True, gridwidth=0.2, gridcolor='#D7DBDD'

	),	

	legend=dict(

		x=1,

		y=1.0,

		bgcolor='white',

		bordercolor='black'

	),

    plot_bgcolor='white',

	barmode='group',

	bargap=0.15,

	bargroupgap=0.1

)



fig = go.Figure(data=data, layout=layout)

pyo.iplot(fig)
usg_c = go.Scatter(x=years_xgame, y=xgame_c['USG%'], name='Centers', marker=dict(color='blue'))

usg_pf = go.Scatter(x=years_xgame, y=xgame_pf['USG%'], name='PowerForwards', marker=dict(color='red'))

usg_sf = go.Scatter(x=years_xgame, y=xgame_sf['USG%'], name='SmallForwards', marker=dict(color='green'))

usg_sg = go.Scatter(x=years_xgame, y=xgame_sg['USG%'], name='ShootingGuards', marker=dict(color='purple'))

usg_pg = go.Scatter(x=years_xgame, y=xgame_pg['USG%'], name='PointGuards', marker=dict(color='orange'))

usg_gen = go.Bar(x=years_xgame, y=xgame_gen['USG%'], name='Average', marker=dict(color='#FFD5C1'))



data=[usg_c,usg_pf,usg_sf,usg_sg,usg_pg,usg_gen]

layout=go.Layout(

	title='USG% by position',

	xaxis=dict(

		title='Years',

		titlefont=dict(size=16, color='#000000'),

		tickfont=dict(size=14, color='#000000')

	),

	yaxis=dict(

		title='USG%',

		titlefont=dict(size=16, color='#000000'),

		tickfont=dict(size=14, color='#000000'),

        showgrid=True, gridwidth=0.2, gridcolor='#D7DBDD'

	),	

	legend=dict(

		x=1,

		y=1.0,

		bgcolor='white',

		bordercolor='black'

	),

    plot_bgcolor='white',

	barmode='group',

	bargap=0.15,

	bargroupgap=0.1

)



fig = go.Figure(data=data, layout=layout)

pyo.iplot(fig)