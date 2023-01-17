# Library Initializations

import pandas as pd

import plotly.graph_objs as go

import numpy as np

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from plotly.tools import FigureFactory as FF



init_notebook_mode(connected=True)

pd.set_option('display.max_columns', 40)



# Read data from files

# The file is stored at the same location as the notebook

matches = pd.read_csv("../input/matches.csv")

deliveries = pd.read_csv("../input/deliveries.csv")



matches.head(2)
deliveries.head(2)
no_of_wins = matches.groupby(['winner','season']).season.count()

# No of Wins by Chennai Super Kings

data = [go.Bar(

            x=pd.DataFrame(no_of_wins['Chennai Super Kings'].index)['season'],

            y=pd.DataFrame(no_of_wins['Chennai Super Kings'].values)[0],

            marker={

                'color': pd.DataFrame(no_of_wins['Chennai Super Kings'].values)[0],

                'colorscale': 'Viridis'

            }

    )]



iplot(data, filename='basic-line')
venue_list = matches.groupby('venue').venue.count()



data = [go.Bar(

            y=pd.DataFrame(venue_list.index)['venue'],

            x=pd.DataFrame(venue_list.values)[0],

            orientation = 'h',

            marker={

                'color': pd.DataFrame(venue_list.values)[0],

                'colorscale': 'Jet'

            }

    )]

iplot(data, filename='basic-bar')
df = pd.merge(matches, deliveries,how='inner',left_on='id',right_on='match_id')

top_10_batsman = df.groupby(['batsman']).batsman_runs.sum().sort_values(ascending=False)[0:10].index



no_of_fours = pd.DataFrame(df[df.batsman.isin(top_10_batsman) & (df.batsman_runs==4)].groupby(['batsman']).batsman_runs.count())

no_of_sixes = pd.DataFrame(df[df.batsman.isin(top_10_batsman) & (df.batsman_runs==6)].groupby(['batsman']).batsman_runs.count())





no_of_fours.reset_index(inplace=True)

no_of_sixes.reset_index(inplace=True)



no_of_fours.columns = ['batsman','no of fours']

no_of_sixes.columns = ['batsman','no of sixes']



total_no_fours_sixes = no_of_fours.merge(no_of_sixes, left_on='batsman',right_on='batsman')



trace1 = go.Bar(

            x=total_no_fours_sixes['batsman'],

            y=total_no_fours_sixes['no of fours'],

            text=total_no_fours_sixes['no of fours'],

            textposition='auto',

            name = 'No of 4s'

        )

trace2= go.Bar(        

            x=total_no_fours_sixes['batsman'],

            y=total_no_fours_sixes['no of sixes'],

            text=total_no_fours_sixes['no of sixes'],

            textposition='auto',

            name = 'No of 6s'

        )



data = [trace1,trace2]

layout = go.Layout(

    barmode='group'

)



fig = go.Figure(data=data, layout=layout)



iplot(fig, filename='group-bar')
from IPython.core.display import HTML





winner_by_shortest_margin = pd.DataFrame(matches[(matches.win_by_runs<5) & (matches.win_by_wickets==0)].groupby(['season','winner']).id.count()).reset_index()

winner_by_shortest_margin.columns=['season','winner','no of times']



display(HTML(winner_by_shortest_margin.to_html()))

 

winner_by_least_wickets = pd.DataFrame(matches[(matches.win_by_runs==0) & (matches.win_by_wickets<=2)].groupby(['season','winner']).id.count()).reset_index()

winner_by_least_wickets.columns=['season','winner','no of times']



display(HTML(winner_by_least_wickets.to_html()))

start_season, end_season = 2008, 2018

while start_season < end_season:

    wins_percity = pd.DataFrame(matches[matches['season'] == start_season].groupby(['winner', 'city'])['id'].count().unstack()).reset_index()

    wins_percity.fillna(value=0,inplace=True)

    wins_percity.dropna(axis='columns',how='all',inplace=True)

    wins_percity.dropna(inplace=True)



    city_in_season = matches[matches.season == start_season].city.unique()

    data = list();

    

    for each_city in city_in_season:

        trace = go.Bar(

            x=wins_percity['winner'],

            y=wins_percity[each_city],

            text=wins_percity[each_city],

            textposition='auto',

            name=each_city,

        )

        data.append(trace)

    layout = go.Layout(

        barmode='stack',

        title='Wins Per City in Season '+str(start_season),

    )

    fig = go.Figure(data=data, layout=layout)

    iplot(fig, filename='stacked-bar')

    start_season += 1
#Toss won vs Match won

win_count = pd.DataFrame(matches[matches.toss_winner == matches.winner].groupby('season').winner.count()).reset_index()

win_count.columns = ['season','both_toss_n_match_winner']



total_no_of_matches = pd.DataFrame(matches.groupby('season').id.count()).reset_index()

total_no_of_matches.columns = ['season','no_of_matches']



percentage_wins = win_count.merge(total_no_of_matches)

percentage_wins['percentage_wins'] = round(percentage_wins['both_toss_n_match_winner']*100/percentage_wins['no_of_matches'],2)



percentage_wins.set_index('season',inplace=True)



percentage_wins.plot.line()
type_of_dismissal = ['caught','bowled', 'lbw', 'caught and bowled','stumped', 'hit wicket']

valid_wickets_df = pd.DataFrame(df[df.dismissal_kind.isin(type_of_dismissal)].groupby(['season','bowler']).inning.count())



top_5_wicket_takers = pd.DataFrame(valid_wickets_df.sort_values(by=["season","inning"],ascending=False).groupby('season').head(5)).reset_index()

top_5_wicket_takers
balls_per_season = pd.DataFrame(df[df.is_super_over == 1].groupby('season').id.count()).reset_index()

balls_per_season.columns = ['season','no_of_superover_balls']

data = [go.Bar(

            x=balls_per_season['no_of_superover_balls'],

            y=balls_per_season['season'],

            text=balls_per_season['no_of_superover_balls'],

            textposition='auto',

            orientation = 'h',

            marker={

                'color': balls_per_season['season'],

                'colorscale': 'Jet'

            }

    )]

layout = go.Layout(

        title='No of balls bowled in SuperOver '

    )

fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='basic-bar')