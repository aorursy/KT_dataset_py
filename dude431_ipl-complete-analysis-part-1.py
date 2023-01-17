import numpy as np 

import pandas as pd

import os



import plotly.plotly as py

from plotly import tools

from plotly.offline import init_notebook_mode,iplot

init_notebook_mode(connected=False)

import plotly.figure_factory as ff

import plotly.graph_objs as go



print(os.listdir("../input"))
deliveries = pd.read_csv('../input/deliveries.csv')

matches = pd.read_csv('../input/matches.csv')
#Since umpire3 contains all null values we can omit the column

matches.drop('umpire3',axis = 1, inplace=True)
x=['Sunrisers Hyderabad', 'Mumbai Indians', 'Gujarat Lions',

    'Rising Pune Supergiant', 'Royal Challengers Bangalore',

    'Kolkata Knight Riders', 'Delhi Daredevils', 'Kings XI Punjab',

    'Chennai Super Kings', 'Rajasthan Royals', 'Deccan Chargers',

    'Kochi Tuskers Kerala', 'Pune Warriors', 'Rising Pune Supergiants', 'Delhi Capitals']



y = ['SRH','MI','GL','RPS','RCB','KKR','DC','KXIP','CSK','RR','SRH','KTK','PW','RPS','DC']



matches.replace(x,y,inplace = True)

deliveries.replace(x,y,inplace = True)
matches['season'].value_counts().head(3)
data = [go.Histogram(x=matches['season'], marker=dict(color='#EB89B5'),opacity=0.75)]

layout = go.Layout(title='Matches In Every Season ',xaxis=dict(title='Season',tickmode='linear'),

                    yaxis=dict(title='Count'),bargap=0.2)



fig = go.Figure(data=data, layout=layout)

iplot(fig)
matches_played=pd.concat([matches['team1'],matches['team2']])

matches_played=matches_played.value_counts().reset_index()

matches_played.columns=['Team','Total Matches']

matches_played['wins']=matches['winner'].value_counts().reset_index()['winner']



matches_played.set_index('Team',inplace=True)
matches_played.reset_index().head(8)
win_percentage = round(matches_played['wins']/matches_played['Total Matches'],3)*100

win_percentage.head(3)
trace1 = go.Bar(x=matches_played.index,y=matches_played['Total Matches'],

                name='Total Matches',opacity=0.4)



trace2 = go.Bar(x=matches_played.index,y=matches_played['wins'],

                name='Matches Won',marker=dict(color='red'),opacity=0.4)



trace3 = go.Bar(x=matches_played.index,

               y=(round(matches_played['wins']/matches_played['Total Matches'],3)*100),

               name='Win Percentage',opacity=0.6,marker=dict(color='gold'))



data = [trace1, trace2, trace3]



layout = go.Layout(title='Match Played, Wins And Win Percentage',xaxis=dict(title='Team'),

                   yaxis=dict(title='Count'),bargap=0.2,bargroupgap=0.1)



fig = go.Figure(data=data, layout=layout)

iplot(fig)
venue_matches=matches.groupby('venue').count()[['id']].sort_values(by='id',ascending=False).head()

ser = pd.Series(venue_matches['id']) 

ser
venue_matches=matches.groupby('venue').count()[['id']].reset_index()



data = [{"x": venue_matches['id'],"y": venue_matches['venue'], 

          "marker": {"color": "lightblue", "size": 12},

         "line": {"color": "red","width" : 2,"dash" : 'dash'},

          "mode": "markers+lines", "name": "Women", "type": "scatter"}]



layout = {"title": "Stadiums and Matches", 

          "xaxis": {"title": "Matches Played", }, 

          "yaxis": {"title": "Stadiums"},

          "autosize":False,"width":900,"height":1000,

          "margin": go.layout.Margin(l=340, r=0,b=100,t=100,pad=0)}



fig = go.Figure(data=data, layout=layout)

iplot(fig)
ump=pd.concat([matches['umpire1'],matches['umpire2']])

ump=ump.value_counts()

umps=ump.to_frame().reset_index()
ump.head()
data = [go.Bar(x=umps['index'],y=umps[0],opacity=0.4)]



layout = go.Layout(title='Umpires in Matches',

                   yaxis=dict(title='Matches'),bargap=0.2)



fig = go.Figure(data=data, layout=layout)

iplot(fig)
batsmen = matches[['id','season']].merge(deliveries, left_on = 'id', right_on = 'match_id', how = 'left').drop('id', axis = 1)

season=batsmen.groupby(['season'])['total_runs'].sum().reset_index()



avgruns_each_season=matches.groupby(['season']).count().id.reset_index()

avgruns_each_season.rename(columns={'id':'matches'},inplace=1)

avgruns_each_season['total_runs']=season['total_runs']

avgruns_each_season['average_runs_per_match']=avgruns_each_season['total_runs']/avgruns_each_season['matches']
fig = {"data" : [{"x" : season["season"],"y" : season["total_runs"],

                  "name" : "Total Run","marker" : {"color" : "lightblue","size": 12},

                  "line": {"width" : 3},"type" : "scatter","mode" : "lines+markers" },

        

                 {"x" : season["season"],"y" : avgruns_each_season["average_runs_per_match"],

                  "name" : "Average Run","marker" : {"color" : "brown","size": 12},

                  "type" : "scatter","line": {"width" : 3},"mode" : "lines+markers",

                  "xaxis" : "x2","yaxis" : "y2",}],

       

        "layout" : {"title": "Total and Average run per Season",

                    "xaxis2" : {"domain" : [0, 1],"anchor" : "y2",

                    "showticklabels" : False},"margin" : {"b" : 111},

                    "yaxis2" : {"domain" : [.55, 1],"anchor" : "x2","title": "Average Run"},                    

                    "xaxis" : {"domain" : [0, 1],"tickmode":'linear',"title": "Year"},

                    "yaxis" : {"domain" :[0, .45], "title": "Total Run"}}}



iplot(fig)
avgruns_each_season.sort_values(by='total_runs', ascending=False).head(2)
Season_boundaries=batsmen.groupby("season")["batsman_runs"].agg(lambda x: (x==6).sum()).reset_index()

fours=batsmen.groupby("season")["batsman_runs"].agg(lambda x: (x==4).sum()).reset_index()

Season_boundaries=Season_boundaries.merge(fours,left_on='season',right_on='season',how='left')

Season_boundaries=Season_boundaries.rename(columns={'batsman_runs_x':'6"s','batsman_runs_y':'4"s'})
Season_boundaries['6"s'] = Season_boundaries['6"s']*6

Season_boundaries['4"s'] = Season_boundaries['4"s']*4

Season_boundaries['total_runs'] = season['total_runs']
trace1 = go.Bar(

    x=Season_boundaries['season'],

    y=Season_boundaries['total_runs']-(Season_boundaries['6"s']+Season_boundaries['4"s']),

    name='Remaining runs',opacity=0.6)



trace2 = go.Bar(

    x=Season_boundaries['season'],

    y=Season_boundaries['4"s'],

    name='Run by 4"s',opacity=0.7)



trace3 = go.Bar(

    x=Season_boundaries['season'],

    y=Season_boundaries['6"s'],

    name='Run by 6"s',opacity=0.7)





data = [trace1, trace2, trace3]

layout = go.Layout(title="Run Distribution per year",barmode='stack',xaxis = dict(tickmode='linear',title="Year"),

                                    yaxis = dict(title= "Run Distribution"))



fig = go.Figure(data=data, layout=layout)

iplot(fig)
high_scores=deliveries.groupby(['match_id', 'inning','batting_team','bowling_team'])['total_runs'].sum().reset_index() 

high_scores=high_scores[high_scores['total_runs']>=200]

high_scores.nlargest(10,'total_runs')
high_scores=high_scores[high_scores.batting_team != 'GL']

high_scores=high_scores[high_scores.bowling_team != 'RPS']

high_scores=high_scores[high_scores.bowling_team != 'GL']

high_scores=high_scores[high_scores.bowling_team != 'PW']
high_scores=high_scores.groupby(['inning','batting_team']).count().reset_index()

high_scores.drop(["bowling_team","total_runs"],axis=1,inplace=True)

high_scores.rename(columns={"match_id":"total_times"},inplace=True)



high_scores_1 = high_scores[high_scores['inning']==1]

high_scores_2 = high_scores[high_scores['inning']==2]
high_scores_1.sort_values(by = 'total_times',ascending=False).head(2)
trace1 = go.Bar(x=high_scores_1['batting_team'],y=high_scores_1['total_times'],name='Ist Innings')

trace2 = go.Bar(x=high_scores_2['batting_team'],y=high_scores_2['total_times'],name='IInd Innings')



fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('At Ist Innings','At IInd Innings'))



fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 1, 2)



iplot(fig)
high_scores=deliveries.groupby(['match_id', 'inning','batting_team','bowling_team'])['total_runs'].sum().reset_index()

high_scores1=high_scores[high_scores['inning']==1]

high_scores2=high_scores[high_scores['inning']==2]

high_scores1=high_scores1.merge(high_scores2[['match_id','inning', 'total_runs']], on='match_id')

high_scores1.rename(columns={'inning_x':'inning_1','inning_y':'inning_2','total_runs_x':'inning1_runs','total_runs_y':'inning2_runs'},inplace=True)

high_scores1=high_scores1[high_scores1['inning1_runs']>=200]

high_scores1['is_score_chased']=1

high_scores1['is_score_chased'] = np.where(high_scores1['inning1_runs']<=high_scores1['inning2_runs'], 'yes', 'no')
slices=high_scores1['is_score_chased'].value_counts().reset_index().is_score_chased

list(slices)

labels=['No','Yes']

slices
trace0 = go.Pie(labels=labels, values=slices,

              hoverinfo='label+value')



layout=go.Layout(title='200 score chased ?')

fig = go.Figure(data=[trace0], layout=layout)

iplot(fig)
agg = matches[['id','season', 'winner', 'toss_winner', 'toss_decision', 'team1']].merge(deliveries, left_on = 'id', right_on = 'match_id', how = 'left').drop('id', axis = 1)

batsman_grp = agg.groupby(["season","match_id", "inning", "batting_team", "batsman"])

batsmen = batsman_grp["batsman_runs"].sum().reset_index()

runs_scored = batsmen.groupby(['season','batting_team', 'batsman'])['batsman_runs'].agg(['sum','mean']).reset_index()

runs_scored['mean']=round(runs_scored['mean'])
agg_battingteam = agg.groupby(['season','match_id', 'inning', 'batting_team', 'bowling_team','winner'])['total_runs'].sum().reset_index()

winner = agg_battingteam[agg_battingteam['batting_team'] == agg_battingteam['winner']]#agg_batting = agg_battingteam.groupby(['season', 'inning', 'team1','winner'])['total_runs'].sum().reset_index()

winner_batting_first = winner[winner['inning'] == 1]

winner_batting_second = winner[winner['inning'] == 2]



winner_runs_batting_first = winner_batting_first.groupby(['season', 'winner'])['total_runs'].mean().reset_index().round()

winner_runs_batting_second = winner_batting_second.groupby(['season', 'winner'])['total_runs'].mean().reset_index().round()



winner_runs = winner_runs_batting_first.merge(winner_runs_batting_second, on = ['season','winner'])

winner_runs.columns = ['season', 'winner', 'batting_first', 'batting_second']
total_win=matches.groupby(['season','winner']).count()[['id']].reset_index()

winner_runs["wins"]= total_win['id']
winner_runs.sort_values(by = ['season'],inplace=True)



csk= winner_runs[winner_runs['winner'] == 'CSK']

rr= winner_runs[winner_runs['winner'] == 'RR']

srh= winner_runs[winner_runs['winner'] == 'SRH']

kkr= winner_runs[winner_runs['winner'] == 'KKR']

mi= winner_runs[winner_runs['winner'] == 'MI']

rcb= winner_runs[winner_runs['winner'] == 'RCB']

kxip= winner_runs[winner_runs['winner'] == 'KXIP']

dd= winner_runs[winner_runs['winner'] == 'DC']
trace1 = go.Scatter(x=csk['season'],y = csk['batting_first'],name='Batting First')

trace2 = go.Scatter(x=csk['season'],y = csk['batting_second'],name='Batting Second')

trace3 = go.Scatter(x=rr['season'],y = rr['batting_first'],name='Batting First')

trace4 = go.Scatter(x=rr['season'],y = rr['batting_second'],name='Batting Second')

trace5 = go.Scatter(x=srh['season'],y = srh['batting_first'],name='Batting First')

trace6 = go.Scatter(x=srh['season'],y = srh['batting_second'],name='Batting Second')

trace7 = go.Scatter(x=kkr['season'],y = kkr['batting_first'],name='Batting First')

trace8 = go.Scatter(x=kkr['season'],y = kkr['batting_second'],name='Batting Second')

trace9 = go.Scatter(x=rcb['season'],y = rcb['batting_first'],name='Batting First')

trace10 = go.Scatter(x=rcb['season'],y = rcb['batting_second'],name='Batting Second')

trace11 = go.Scatter(x=kxip['season'],y = kxip['batting_first'],name='Batting First')

trace12 = go.Scatter(x=kxip['season'],y = kxip['batting_second'],name='Batting Second')

trace13 = go.Scatter(x=mi['season'],y = mi['batting_first'],name='Batting First')

trace14 = go.Scatter(x=mi['season'],y = mi['batting_second'],name='Batting Second')

trace15 = go.Scatter(x=dd['season'],y = dd['batting_first'],name='Batting First')

trace16 = go.Scatter(x=dd['season'],y = dd['batting_second'],name='Batting Second')



fig = tools.make_subplots(rows=4, cols=2, subplot_titles=('CSK', 'RR','SRH', 'KKR','RCB', 'KXIP','MI', 'DC'))



fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 1, 1)

fig.append_trace(trace3, 1, 2)

fig.append_trace(trace4, 1, 2)

fig.append_trace(trace5, 2, 1)

fig.append_trace(trace6, 2, 1)

fig.append_trace(trace7, 2, 2)

fig.append_trace(trace8, 2, 2)

fig.append_trace(trace9, 3, 1)

fig.append_trace(trace10, 3, 1)

fig.append_trace(trace11, 3, 2)

fig.append_trace(trace12, 3, 2)

fig.append_trace(trace13, 4, 1)

fig.append_trace(trace14, 4, 1)

fig.append_trace(trace15, 4, 2)

fig.append_trace(trace16, 4, 2)



fig['layout'].update(title='Batting first vs Batting Second of Teams',showlegend=False)

iplot(fig)
runs_per_over = deliveries.pivot_table(index=['over'],columns='batting_team',values='total_runs',aggfunc=sum)

runs_per_over.reset_index(inplace=True)

runs_per_over.drop(['KTK','PW','RPS','GL'],axis=1,inplace=True)
trace1 = go.Scatter(x=runs_per_over['over'],y = runs_per_over['CSK'],name='CSK',marker= dict(color= "blue",size=12))

trace2 = go.Scatter(x=runs_per_over['over'],y = runs_per_over['DC'],name='DC')

trace3 = go.Scatter(x=runs_per_over['over'],y = runs_per_over['KKR'],name='KKR')

trace4 = go.Scatter(x=runs_per_over['over'],y = runs_per_over['KXIP'],name='KXIP')

trace5 = go.Scatter(x=runs_per_over['over'],y = runs_per_over['MI'],name='MI')

trace6 = go.Scatter(x=runs_per_over['over'],y = runs_per_over['RCB'],name='RCB')

trace7 = go.Scatter(x=runs_per_over['over'],y = runs_per_over['RR'],name='RR')

trace8 = go.Scatter(x=runs_per_over['over'],y = runs_per_over['SRH'],name='SRH')



data = [trace1,trace2,trace3,trace4,trace5,trace6,trace7,trace8]



layout = go.Layout(title='Average Run in Each Over',xaxis = dict(tickmode='linear',title="Over"),

                                    yaxis = dict(title= "Runs"))



fig = go.Figure(data=data,layout=layout)

iplot(fig)
season=matches[['id','season','winner']]

complete_data=deliveries.merge(season,how='inner',left_on='match_id',right_on='id')
powerplay_data=complete_data[complete_data['over']<=6]



inn1 = powerplay_data[ powerplay_data['inning']==1].groupby('match_id')['total_runs'].agg(['sum']).reset_index()

inn2 = powerplay_data[ powerplay_data['inning']==2].groupby('match_id')['total_runs'].agg(['sum']).reset_index()
inn1.reset_index(inplace=True)

inn1.drop(["match_id"],axis=1,inplace=True)



inn2.reset_index(inplace=True)

inn2.drop(["match_id"],axis=1,inplace=True)
fig = {"data" : [{"x" : inn1["index"],"y" : inn1["sum"],"marker" : {"color" : "blue","size": 2},

                  "line": {"width" : 1.5},"type" : "scatter","mode" : "lines" },

        

                 {"x" : inn2["index"],"y" : inn2["sum"],"marker" : {"color" : "brown","size": 2},

                  "type" : "scatter","line": {"width" : 1.5},"mode" : "lines",

                  "xaxis" : "x2","yaxis" : "y2",}],

       

        "layout" : {"title": "Inning 1 vs Inning 2 in Powerplay Overs",

                    "xaxis2" : {"domain" : [0, 1],"anchor" : "y2",

                    "showticklabels" : False},

                    "yaxis2" : {"domain" : [.55, 1],"anchor" : "x2","title": "Inn2 Powerplay"},

                    "margin" : {"b" : 111},

                    "xaxis" : {"domain" : [0, 1],"title": "Matches"},

                    "yaxis" : {"domain" :[0, .45], "title": "Inn1 Poweplay"}}}



iplot(fig)
pi1=powerplay_data[ powerplay_data['inning']==1].groupby(['season','match_id'])['total_runs'].agg(['sum'])

pi1=pi1.reset_index().groupby('season')['sum'].mean()

pi1=pi1.to_frame().reset_index()



pi2=powerplay_data[ powerplay_data['inning']==2].groupby(['season','match_id'])['total_runs'].agg(['sum'])

pi2=pi2.reset_index().groupby('season')['sum'].mean()

pi2=pi2.to_frame().reset_index()
trace1 = go.Bar(x=pi1.season,y=pi1["sum"],

                name='Inning 1',opacity=0.4)



trace2 = go.Bar(x=pi2.season,y=pi2["sum"],name='Inning 2',

                marker=dict(color='red'),opacity=0.4)



data = [trace1, trace2]

layout = go.Layout(title='Powerplay Average runs per Year',

                   xaxis=dict(title='Year',tickmode='linear'),

                   yaxis=dict(title='Run'),bargap=0.2,bargroupgap=0.1)



fig = go.Figure(data=data, layout=layout)

iplot(fig)
powerplay_dismissals=powerplay_data.dropna(subset=['dismissal_kind']).groupby(['season','match_id','inning'])['dismissal_kind'].agg(['count']).reset_index().groupby('season')['count'].max()

powerplay_dismissals=powerplay_dismissals.reset_index()



powerplay_dismissals_first=powerplay_data[ powerplay_data['inning']==1].dropna(subset=['dismissal_kind']).groupby(['season','match_id','inning'])['dismissal_kind'].agg(['count']).reset_index().groupby('season')['count'].mean()

powerplay_dismissals_first=powerplay_dismissals_first.reset_index()



powerplay_dismissals_second=powerplay_data[ powerplay_data['inning']==2].dropna(subset=['dismissal_kind']).groupby(['season','match_id','inning'])['dismissal_kind'].agg(['count']).reset_index().groupby('season')['count'].mean()

powerplay_dismissals_second=powerplay_dismissals_second.reset_index()
trace1 = go.Bar(x=powerplay_dismissals.season,y=powerplay_dismissals["count"],

                name='Max',opacity=0.4)



trace2 = go.Bar(x=powerplay_dismissals_first.season,y=powerplay_dismissals_first["count"],name='Inning 1',

                marker=dict(color='red'),opacity=0.4)



trace3 = go.Bar(x=powerplay_dismissals_second.season,y=powerplay_dismissals_second["count"],name='Inning 2',

                marker=dict(color='lime'),opacity=0.4)



data = [trace1, trace2, trace3]

layout = go.Layout(title='Powerplay Average Dismissals per Year',

                   xaxis=dict(title='Year',tickmode='linear'),

                   yaxis=dict(title='Run'),bargap=0.2,bargroupgap=0.1)



fig = go.Figure(data=data, layout=layout)

iplot(fig)
df_strike_rate = deliveries.groupby(['batsman']).agg({'ball':'count','batsman_runs':'mean'}).sort_values(by='batsman_runs',ascending=False)

df_strike_rate.rename(columns ={'batsman_runs' : 'strike rate'}, inplace=True)

df_runs_per_match = deliveries.groupby(['batsman','match_id']).agg({'batsman_runs':'sum'})

df_total_runs = df_runs_per_match.groupby(['batsman']).agg({'sum' ,'mean','count'})

df_total_runs.rename(columns ={'sum' : 'batsman run','count' : 'match count','mean' :'average score'}, inplace=True)

df_total_runs.columns = df_total_runs.columns.droplevel()

df_sixes = deliveries[['batsman','batsman_runs']][deliveries.batsman_runs==6].groupby(['batsman']).agg({'batsman_runs':'count'})

df_four = deliveries[['batsman','batsman_runs']][deliveries.batsman_runs==4].groupby(['batsman']).agg({'batsman_runs':'count'})

df_batsman_stat = pd.merge(pd.merge(pd.merge(df_strike_rate,df_total_runs, left_index=True, right_index=True),

                                    df_sixes, left_index=True, right_index=True),df_four, left_index=True, right_index=True)
df_batsman_stat.rename(columns = {'ball' : 'ball', 'strike rate':'strike_rate','batsman run' : 'batsman_run',

                                  'match count' : 'match_count','average score' : 'average_score' ,'batsman_runs_x' :'six',

                                  'batsman_runs_y':'four'},inplace=True)

df_batsman_stat['strike_rate'] = df_batsman_stat['strike_rate']*100

df_batsman_stat.sort_values(by='batsman_run',ascending=False,inplace=True)

#df_batsman_stat.sort_values(by='batsman_run',ascending=False)

df_batsman_stat.reset_index(inplace=True)
average_score=df_batsman_stat.sort_values(by='average_score',ascending=False)

average_score=average_score[average_score['match_count']>50].head(10)



strike_rate=df_batsman_stat.sort_values(by='strike_rate',ascending=False)

strike_rate=strike_rate[strike_rate['match_count']>50].head(10)
trace1 = go.Bar(x=average_score['batsman'],y=average_score['average_score'],

                name='Average Score',marker=dict(color='gold'),opacity=0.6,showlegend=False)



trace2 = go.Bar(x=strike_rate['batsman'],y=strike_rate['strike_rate'],

                name='Strike Rate',marker=dict(color='brown'),opacity=0.6,showlegend=False)



fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('Highest Average Score','Highest Strike Rate'))



fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 1, 2)



fig['layout']['xaxis1'].update(title='Player')

fig['layout']['xaxis2'].update(title='Player')



iplot(fig)
top_df_batsman=df_batsman_stat.head(20)

top_df_batsman.head(5)
data=[{"y": top_df_batsman.match_count,

        "x": top_df_batsman.batsman,

        "mode":"markers",

        "marker":{"color":top_df_batsman.six,"size" :top_df_batsman.average_score,'showscale': True},

        "text":top_df_batsman.batsman }]



iplot(data)
toppers=deliveries.groupby(['batsman','batsman_runs'])['total_runs'].count().reset_index()

toppers=toppers.pivot('batsman','batsman_runs','total_runs')

toppers.reset_index(inplace=True)
top_6 = toppers.sort_values(6,ascending=False).head(10)

top_4 = toppers.sort_values(4,ascending=False).head(10)

top_2 = toppers.sort_values(2,ascending=False).head(10)

top_1 = toppers.sort_values(1,ascending=False).head(10)
trace1 = go.Scatter(x=top_6.batsman,y =top_6[6],name='6"s',marker =dict(color= "blue",size = 9),line=dict(width=2,dash='dash'))

trace2 = go.Scatter(x=top_4.batsman,y = top_4[4],name='4"s',marker =dict(color= "orange",size = 9),line=dict(width=2,dash='longdash'))

trace3 = go.Scatter(x=top_2.batsman,y = top_2[2],name='2"s',marker =dict(color= "green",size = 9),line=dict(width=2,dash='dashdot'))

trace4 = go.Scatter(x=top_1.batsman,y = top_1[1],name='1"s',marker =dict(color= "red",size = 9),line=dict(width=2,dash='longdashdot'))



fig = tools.make_subplots(rows=4, cols=1, subplot_titles=('Top 6"s Scorer','Top 4"s Scorer',

                                                          'Top 2"s Scorer','Top 1"s Scorer'))



fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 2, 1)

fig.append_trace(trace3, 3, 1)

fig.append_trace(trace4, 4, 1)



fig['layout'].update(title='Top Scorer in each Category',showlegend=False)

iplot(fig)
orange=matches[['id','season']]

orange=orange.merge(deliveries,left_on='id',right_on='match_id',how='left')

orange=orange.groupby(['season','batsman'])['batsman_runs'].sum().reset_index()

orange=orange.sort_values('batsman_runs',ascending=0)

orange=orange.drop_duplicates(subset=["season"],keep="first")

#orange.sort_values(by='season')
data = [go.Bar(x=orange['season'].values,y=orange['batsman_runs'].values,

                name='Total Matches',text=orange['batsman'].values,

                marker=dict(color='rgb(255,140,0)',

                            line=dict(color='rgb(8,48,107)',width=1.5,)),opacity=0.7)]



layout = go.Layout(title='Orange-Cap Holders',xaxis = dict(tickmode='linear',title="Year"),

                   yaxis=dict(title='Runs'))

fig = go.Figure(data=data, layout=layout)

iplot(fig)
bowlers=deliveries.groupby('bowler').sum().reset_index()

bowl=deliveries['bowler'].value_counts().reset_index()

bowlers=bowlers.merge(bowl,left_on='bowler',right_on='index',how='left')

bowlers=bowlers[['bowler_x','total_runs','bowler_y']]

bowlers.rename({'bowler_x':'bowler','total_runs':'runs_given','bowler_y':'balls'},axis=1,inplace=True)

bowlers['overs']=(bowlers['balls']//6)
dismissal_kinds = ["bowled", "caught", "lbw", "stumped", "caught and bowled", "hit wicket"]  

ct=deliveries[deliveries["dismissal_kind"].isin(dismissal_kinds)]

ct=ct['bowler'].value_counts().reset_index()

bowlers=bowlers.merge(ct,left_on='bowler',right_on='index',how='left').dropna()

bowlers=bowlers[['bowler_x','runs_given','overs','bowler_y']]

bowlers.rename({'bowler_x':'bowler','bowler_y':'wickets'},axis=1,inplace=True)

bowlers['economy']=(bowlers['runs_given']/bowlers['overs'])
bowlers_top=bowlers.sort_values(by='runs_given',ascending=False)

bowlers_top=bowlers_top.head(20)
trace = go.Scatter(y = bowlers_top['wickets'],x = bowlers_top['bowler'],mode='markers',

                   marker=dict(size= bowlers_top['wickets'].values,

                               color = bowlers_top['economy'].values,

                               colorscale='Viridis',

                               showscale=True,

                               colorbar = dict(title = 'Economy')),

                   text = bowlers['overs'].values)



data = [(trace)]



layout= go.Layout(autosize= True,

                  title= 'Top 20 Wicket Taking Bowlers',

                  hovermode= 'closest',

                  xaxis=dict(showgrid=False,zeroline=False,

                             showline=False),

                  yaxis=dict(title= 'Wickets Taken',ticklen= 5,

                             gridwidth= 2,showgrid=False,

                             zeroline=False,showline=False),

                  showlegend= False)



fig = go.Figure(data=data, layout=layout)

iplot(fig)
dismissal_kinds = ["bowled", "caught", "lbw", "stumped", "caught and bowled", "hit wicket"]  #since run-out is not creditted to the bowler

purple=deliveries[deliveries["dismissal_kind"].isin(dismissal_kinds)]

purple=purple.merge(matches,left_on='match_id',right_on='id',how='outer')

purple=purple.groupby(['season','bowler'])['dismissal_kind'].count().reset_index()

purple=purple.sort_values('dismissal_kind',ascending=False)

purple=purple.drop_duplicates('season',keep='first').sort_values(by='season')

purple.rename({'dismissal_kind':'count_wickets'},axis=1,inplace=True)
trace1 = go.Bar(x=purple['season'].values,y=purple['count_wickets'].values,

                name='Total Matches',text=purple['bowler'].values,

                marker=dict(color='rgb(75,0,130)',

                            line=dict(color='rgb(108,148,107)',width=1.5,)),

                opacity=0.7)



layout = go.Layout(title='Purple-Cap Holders',xaxis = dict(tickmode='linear',title="Year"),

                   yaxis=dict(title='Wickets'))



data=[trace1]

fig = go.Figure(data=data, layout=layout)

iplot(fig)
season_winner=matches.drop_duplicates(subset=['season'], keep='last')[['season','winner']].reset_index(drop=True)

season_winner = season_winner['winner'].value_counts()



season_winner = season_winner.to_frame()

season_winner.reset_index(inplace=True)

season_winner.rename(columns={'index':'team'},inplace=True)
season_winner
trace0 = go.Pie(labels=season_winner['team'], values=season_winner['winner'],

              hoverinfo='label+value+name',name="Winner")



layout=go.Layout(title='Winner of IPL season')

fig = go.Figure(data=[trace0], layout=layout)

iplot(fig)
finals=matches.drop_duplicates(subset=['season'],keep='last')

finals=finals[['id','season','city','team1','team2','toss_winner','toss_decision','winner']]

most_finals=pd.concat([finals['team1'],finals['team2']]).value_counts().reset_index()

most_finals.rename({'index':'team',0:'count'},axis=1,inplace=True)

xyz=finals['winner'].value_counts().reset_index()
most_finals=most_finals.merge(xyz,left_on='team',right_on='index',how='outer')

most_finals=most_finals.replace(np.NaN,0)

most_finals.drop('index',axis=1,inplace=True)

most_finals.set_index('team',inplace=True)

most_finals.rename({'count':'finals_played','winner':'won_count'},inplace=True,axis=1)

most_finals.reset_index(inplace=True)
trace1 = go.Bar(x=most_finals.team,y=most_finals.finals_played,

                name='Total Matches',opacity=0.4)



trace2 = go.Bar(x=most_finals.team,y=most_finals.won_count,

                name='Matches Won',marker=dict(color='red'),opacity=0.4)



data = [trace1, trace2]



layout = go.Layout(title='Match Played vs Wins In Finals',xaxis=dict(title='Team'),

                   yaxis=dict(title='Count'),bargap=0.2,bargroupgap=0.1)



fig = go.Figure(data=data, layout=layout)

iplot(fig)
df=finals[finals['toss_winner']==finals['winner']]

slices=[len(df),(len(finals)-len(df))]

labels=['yes','no']
trace0 = go.Pie(labels=labels, values=slices,

              hoverinfo='label+value+name',name="Winner")



layout=go.Layout(title='Winner of IPL season')

fig = go.Figure(data=[trace0], layout=layout)

iplot(fig)