import numpy as np 

import pandas as pd



import plotly.plotly as py

from plotly import tools

from plotly.offline import init_notebook_mode,iplot

init_notebook_mode(connected=False)

import plotly.figure_factory as ff

import plotly.graph_objs as go



import warnings

warnings.filterwarnings("ignore")
deliveries = pd.read_csv('../input/deliveries.csv')

matches = pd.read_csv('../input/matches.csv')
s_man_of_match = (matches.groupby(matches.player_of_match).player_of_match.count().

                  sort_values(ascending=False).head(15))



df_man_of_match =(s_man_of_match.to_frame().rename

                  (columns = {"player_of_match": "times"}).reset_index())



cen = deliveries.groupby(['batsman','match_id']).agg({'batsman_runs':'sum'})

cen = cen[cen['batsman_runs']>=100]

cen = cen.groupby(['batsman']).agg({'count'})

cen.columns = cen.columns.droplevel()

cen = cen.sort_values(by='count',ascending=False).reset_index()



half_cen = deliveries.groupby(['batsman','match_id']).agg({'batsman_runs':'sum'})

half_cen = half_cen[half_cen['batsman_runs']>=50]

half_cen = half_cen[half_cen['batsman_runs']<100]

half_cen = half_cen.groupby(['batsman']).agg({'count'})

half_cen.columns = half_cen.columns.droplevel()

half_cen = half_cen.sort_values(by='count',ascending=False).reset_index()



df_big = pd.merge(cen,half_cen, on='batsman',how='right')

df_big = df_big.fillna(0)



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



df_batsman_stat.rename(columns = {'ball' : 'Ball', 'strike rate':'Strike Rate','batsman run' : 'Batsman Run','match count' : 'Match Count',

                                  'average score' : 'Average score' ,'batsman_runs_x' :'Six','batsman_runs_y':'Four'},inplace=True)

df_batsman_stat['Strike Rate'] = df_batsman_stat['Strike Rate']*100

df_batsman_stat = df_batsman_stat.sort_values(by='Batsman Run',ascending=False).reset_index()



batsman_stats = pd.merge(df_batsman_stat,df_big, on='batsman',how='left').fillna(0)

batsman_stats.rename(columns = {'count_x' : '100s', 'count_y' : '50s'},inplace=True)
condition_catch = (deliveries.dismissal_kind == 'caught')

condition_run= (deliveries.dismissal_kind == 'run out')

condition_stump= (deliveries.dismissal_kind == 'stumped')

condition_caught_bowled = (deliveries.dismissal_kind == 'caught and bowled')



s_catch = deliveries.loc[condition_catch,:].groupby(deliveries.fielder).dismissal_kind.count().sort_values(ascending=False)

s_run = deliveries.loc[condition_run,:].groupby(deliveries.fielder).dismissal_kind.count().sort_values(ascending=False)

s_stump = deliveries.loc[condition_stump,:].groupby(deliveries.fielder).dismissal_kind.count().sort_values(ascending=False)

s_caught_bowled = deliveries.loc[condition_caught_bowled,:].groupby(deliveries.bowler).dismissal_kind.count().sort_values(ascending=False)



df_catch= s_catch.to_frame().reset_index().rename(columns ={'dismissal_kind' : 'catch'})

df_run= s_run.to_frame().reset_index().rename(columns ={'dismissal_kind' : 'run_out'})

df_stump= s_stump.to_frame().reset_index().rename(columns ={'dismissal_kind' : 'stump'})

df_caught_bowled = s_caught_bowled.to_frame().reset_index().rename(columns ={'dismissal_kind' : 'caught and bowled'})                                                                                                                           

                                                                                                                           

df_field = pd.merge(pd.merge(df_catch,df_run,on='fielder', how='outer'),df_stump,on='fielder',how='outer')

field_stats = df_field[~df_field['fielder'].str.contains("(sub)")].reset_index().drop(['index'],axis=1).fillna(0)
condition = ((deliveries.dismissal_kind.notnull()) &(deliveries.dismissal_kind != 'run out')&

            (deliveries.dismissal_kind != 'retired hurt' )&(deliveries.dismissal_kind != 'hit wicket') 

            &(deliveries.dismissal_kind != 'obstructing the field')&(deliveries.dismissal_kind != 'caught and bowled'))

        

df_bowlers = deliveries.loc[condition,:].groupby(deliveries.bowler).dismissal_kind.count().sort_values(ascending=False).reset_index()

df_bowlers = pd.merge(df_bowlers,df_caught_bowled , on='bowler',how='left').fillna(0)



high=deliveries.groupby(['match_id', 'bowler']).agg({'total_runs':'sum'}).reset_index()



over_count=deliveries.groupby(['match_id', 'bowler','over']).agg({'total_runs':'sum'}).reset_index()

overs = over_count.groupby(['match_id','bowler']).agg({'over':'count'}).reset_index()

overs = overs[overs['over']>=2]



bowlers = pd.merge(high,overs,on=['match_id', 'bowler'], how='right')

bowlers['economy'] = bowlers['total_runs']/bowlers['over']

bowlers['eco_range'] = pd.cut(bowlers['economy'], [0, 4, 5, 6, 9, 10, 11, 30], labels=['below4', '4-5', '5-6', '6-9','9-10','10-11','above11'])



bowlers = pd.concat([bowlers,pd.get_dummies(bowlers['eco_range'], prefix='eco')],axis=1)

economy_rates=bowlers.groupby(['bowler']).agg({'eco_below4':'sum','eco_4-5':'sum','eco_5-6':'sum','eco_6-9':'sum','eco_9-10':'sum','eco_10-11':'sum','eco_above11':'sum'}).reset_index()



maiden_over = over_count[over_count['total_runs']==0]

maidens = maiden_over['bowler'].value_counts().to_frame().reset_index().rename({'index':'bowler','bowler':'maiden_overs'},axis=1)



hauls=deliveries.groupby(['match_id', 'bowler']).agg({'player_dismissed':'count'}).reset_index()

hauls = hauls[hauls['player_dismissed']>=4]

hauls['haul'] = pd.cut(hauls['player_dismissed'], [0,4,8], labels=['4', '5'])

hauls = pd.concat([hauls,pd.get_dummies(hauls['haul'], prefix='haul')],axis=1)

hauls.drop(['player_dismissed','haul'],inplace=True,axis=1)

hauls=hauls.groupby(['bowler']).agg({'haul_4':'sum','haul_5':'sum'}).reset_index()



bowlers_stats = pd.merge(pd.merge(pd.merge(economy_rates,maidens,on='bowler', how='left'),df_bowlers,on='bowler',how='left'),hauls,on='bowler',how='right').fillna(0)

bowlers_stats.rename(columns ={'dismissal_kind' : 'wickets'},inplace=True)
data = [go.Bar(x=df_man_of_match['player_of_match'], 

               y=df_man_of_match["times"], 

               marker=dict(color='#EB89B5'),opacity=0.75)]



layout = go.Layout(title='Man of the Matches ',

                   xaxis=dict(title='Player',tickmode='linear'),

                   yaxis=dict(title='Count'),bargap=0.2)



fig = go.Figure(data=data, layout=layout)

iplot(fig)
centuries = batsman_stats.sort_values(by='100s').tail(15)

half_centuries = batsman_stats.sort_values(by='50s').tail(15)
fig = {"data" : [{"x" : centuries["batsman"],"y" : centuries["100s"],

                  "name" : "100s","marker" : {"color" : "lightblue","size": 12},

                  "line": {"width" : 3},"type" : "scatter","mode" : "lines+markers" ,

                  "xaxis" : "x1","yaxis" : "y1"},

        

                 {"x" : half_centuries["batsman"],"y" : half_centuries["50s"],

                  "name" : "50s","marker" : {"color" : "brown","size": 12},

                  "type" : "scatter","line": {"width" : 3},"mode" : "lines+markers",

                  "xaxis" : "x2","yaxis" : "y2"}],

       

        "layout" : {"title": "Total centuries and half-centuries by top batsman",

                    "xaxis2" : {"domain" : [0, 1],"anchor" : "y2",

                    "showticklabels" : True},"margin" : {"b" : 111},

                    "yaxis2" : {"domain" : [.55, 1],"anchor" : "x2","title": "50s"},                    

                    "xaxis" : {"domain" : [0, 1],"tickmode":'linear',"title": "Batsman"},

                    "yaxis" : {"domain" :[0, .45], "anchor" : "x2","title": "100s"}}}



iplot(fig)
cen = batsman_stats[['100s','50s','batsman']]

cen['points'] = (cen['100s']*8) + (cen['50s']*4)

cen.sort_values(by='points',inplace=True,ascending=False)
trace = go.Table(

    domain=dict(x=[0, 0.55],

                y=[0, 1.0]),

    header=dict(values=["Batsman","Points","100s","50s"],

                fill = dict(color = '#119DFF'),

                font = dict(color = 'white', size = 14),

                align = ['center'],

               height = 30),

    cells=dict(values=[cen['batsman'].head(10), cen['points'].head(10), cen['100s'].head(10), cen['50s'].head(10)],

               fill = dict(color = ['#25FEFD', 'white']),

               align = ['center']))



trace1 = go.Bar(x=cen['batsman'].head(10),

                y=cen["points"].head(10),

                xaxis='x1',

                yaxis='y1',

                marker=dict(color='brown'),opacity=0.60)



layout = dict(

    width=830,

    height=415,

    autosize=False,

    title='Batsman with highest points by centuries and half centuries',

    margin = dict(t=100),

    showlegend=False,   

    xaxis1=dict(**dict(domain=[0.65, 1], anchor='y1', showticklabels=True)),

    yaxis1=dict(**dict(domain=[0, 1.0], anchor='x1', hoverformat='.2f')),  

)



fig1 = dict(data=[trace, trace1], layout=layout)

iplot(fig1)
fours = batsman_stats.sort_values(by='Four').tail(15)

sixes = batsman_stats.sort_values(by='Six').tail(15)

runs = batsman_stats.sort_values(by='Batsman Run').tail(15)
trace1 = go.Scatter(x=sixes.batsman,y =sixes.Six,name='6"s',marker =dict(color= "blue",size = 9),line=dict(width=2,dash='dash'),showlegend=True)

trace2 = go.Scatter(x=fours.batsman,y = fours.Four,name='4"s',marker =dict(color= "green",size = 9),line=dict(width=2,dash='longdash'))

trace3 = go.Scatter(x=runs.batsman,y = runs['Batsman Run'],name='2"s',marker =dict(color= "red",size = 9),line=dict(width=2,dash='dashdot'))



fig = tools.make_subplots(rows=3, cols=1, subplot_titles=('Top 6"s Scorer','Top 4"s Scorer',"Highest total runs"), print_grid=False)



fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 2, 1)

fig.append_trace(trace3, 3, 1)



fig['layout'].update(height=700, width=820,title='Top Scorer in Boundaries and Total Runs',showlegend=False)

iplot(fig)
runs = batsman_stats[['Six','Four','Batsman Run','batsman']]

runs['point'] = (runs['Six']*1) + (runs['Four']*0.5) + (runs['Batsman Run']*0.5)

runs.sort_values(by='point',inplace=True,ascending=False)
trace = go.Table(

    domain=dict(x=[0, 0.55],

                y=[0, 1.0]),

    header=dict(values=["Batsman","Points","Sixes","Fours"],

                fill = dict(color='#d562be'),

                font = dict(color=['rgb(45, 45, 45)'] * 5, size=14),

                align = ['center'],

               height = 30),

    cells=dict(values=[runs['batsman'].head(10), runs['point'].head(10), runs['Six'].head(10), runs['Four'].head(10)],

               fill = dict(color=['rgb(235, 193, 238)', 'rgba(228, 222, 249, 0.65)']),

               align = ['center']))



trace1 = go.Bar(x=runs['batsman'].head(10),

                y=runs["point"].head(10),

                xaxis='x1',

                yaxis='y1',

                marker=dict(color='gold'),opacity=0.60)



layout = dict(

    width=830,

    height=415,

    autosize=False,

    title='Batsman with highest points by runs and boundaries',

    margin = dict(t=100),

    showlegend=False,   

    xaxis1=dict(**dict(domain=[0.61, 1], anchor='y1', showticklabels=True)),

    yaxis1=dict(**dict(domain=[0, 1.0], anchor='x1', hoverformat='.2f')),  

)



fig1 = dict(data=[trace, trace1], layout=layout)

iplot(fig1)
final = pd.merge(cen,runs,on='batsman', how='inner')

final['total_points']=final['points']+final['point']

final['max'] = final['100s']+final['50s']



final.sort_values(by='total_points',ascending=False,inplace=True)

best_batsman = final[['batsman','total_points']]
final['Batsman Run'] = (final['Batsman Run'])/(final['Batsman Run'].max()/100)

final['Six'] = (final['Six'])/(final['Six'].max()/100)

final['Four'] = (final['Four'])/(final['Four'].max()/100)

final['max'] = (final['max'])/(final['max'].max()/100)

final['total_points'] = (final['total_points'])/(final['total_points'].max()/100)
x = final[final["batsman"] == "V Kohli"]

y = final[final["batsman"] == "CH Gayle"]

z = final[final["batsman"] == "S Dhawan"]



data = [go.Scatterpolar(

  r = [x['Four'].values[0],x['Six'].values[0],x['Batsman Run'].values[0],x['max'].values[0],x['total_points'].values[0]],

  theta = ['Four','Six','Runs','Centuries','Points'],

  fill = 'toself', opacity = 0.8,

  name = "V Kohli"),

        

    go.Scatterpolar(

  r = [y['Four'].values[0],y['Six'].values[0],y['Batsman Run'].values[0],y['max'].values[0],y['total_points'].values[0]],

  theta = ['Four','Six','Runs','Centuries','Points'],

  fill = 'toself',subplot = "polar2",

    name = "CH Gayle"),

       

    go.Scatterpolar(

  r = [z['Four'].values[0],z['Six'].values[0],z['Batsman Run'].values[0],z['max'].values[0],z['total_points'].values[0]],

  theta = ['Four','Six','Runs','Centuries','Points'],

  fill = 'toself',subplot = "polar3",

    name = "S Dhawan")]



layout = go.Layout(title = "Comparison Between V Kohli, CH Gayle, S Dhawan",

                   

                   polar = dict(radialaxis = dict(visible = True,range = [0, 100]),

                   domain = dict(x = [0, 0.25],y = [0, 1])),

                  

                   polar2 = dict(radialaxis = dict(visible = True,range = [0, 100]),

                   domain = dict(x = [0.35, 0.65],y = [0, 1])),

                  

                   polar3 = dict(radialaxis = dict(visible = True,range = [0, 100]),

                   domain = dict(x = [0.75, 1.0],y = [0, 1])),)



fig = go.Figure(data=data, layout=layout)

iplot(fig)
trace1 = go.Bar(x=field_stats.fielder.head(15),y=field_stats.catch,

                name='Caught',opacity=0.4)



trace2 = go.Bar(x=field_stats.fielder.head(15),y=field_stats.run_out,name='Run out',

                marker=dict(color='red'),opacity=0.4)



trace3 = go.Bar(x=field_stats.fielder.head(15),y=field_stats.stump,name='Stump out',

                marker=dict(color='lime'),opacity=0.4)



data = [trace1, trace2, trace3]

layout = go.Layout(title='Best fielders',

                   xaxis=dict(title='Player',tickmode='linear'),

                   yaxis=dict(title='Dismissals'),bargap=0.2,bargroupgap=0.1)



fig = go.Figure(data=data, layout=layout)

iplot(fig)
field = field_stats[['fielder','stump','catch','run_out']]





field1 = field[(field['stump'] > 0)]

field2 = field[~(field['stump'] > 0)]



field1['points'] = (field1['catch']*4) + (field1['stump']*6) + (field1['run_out']*2)

field2['points'] = (field2['catch']*4) + (field2['stump']*6) + (field2['run_out']*6)
field = pd.concat([field1, field2])

field.sort_values(by='points',ascending=False,inplace=True)



field1.sort_values(by='points',ascending=False,inplace=True)

field2.sort_values(by='points',ascending=False,inplace=True)



best_fielder = field[['fielder','points']]
trace = go.Table(

    domain=dict(x=[0, 0.65],

                y=[0, 1.0]),

    header=dict(values=["Fielder","Stump","Catch","Run out","Points"],

                fill = dict(color = 'grey'),

                font = dict(color = 'white', size = 14),

                align = ['center'],

               height = 30),

    cells=dict(values=[field['fielder'].head(10),field['stump'].head(10),field['catch'].head(10),field['run_out'].head(10),field['points'].head(10)],

               fill = dict(color = ['lightgrey', 'white']),

               align = ['center']))



trace1 = go.Bar(x=field['fielder'].head(10),

                y=field["points"].head(10),

                xaxis='x1',

                yaxis='y1',

                marker=dict(color='hotpink'),opacity=0.60)



layout = dict(

    width=850,

    height=440,

    autosize=False,

    title='Fielder with maximum Points',

    showlegend=False,   

    xaxis1=dict(**dict(domain=[0.7, 1], anchor='y1', showticklabels=True)),

    yaxis1=dict(**dict(domain=[0, 1.0], anchor='x1', hoverformat='.2f')),  

)



fig1 = dict(data=[trace, trace1], layout=layout)

iplot(fig1)
trace0 = go.Scatter(

    x=field1['points'].head(5),

    y=field1['fielder'],

    name = 'Wicketkeeper',

    mode='markers',

    marker=dict(

        color='rgba(156, 165, 196, 0.95)',

        line=dict(color='rgba(156, 165, 196, 1.0)',width=1),

        symbol='circle',

        size=16,

    ))

trace1 = go.Scatter(

    x=field2['points'].head(5),

    y=field2['fielder'],

    name='Fielder',

    mode='markers',

    marker=dict(

        color='rgba(204, 204, 204, 0.95)',

        line=dict(color='rgba(217, 217, 217, 1.0)',width=1),

        symbol='circle',

        size=16,

    ))



data = [trace0,trace1]

layout = go.Layout(

    title="Ten best Fielders and Wicketkeepers for Fantasy League ",

    xaxis=dict(

        showgrid=False,

        showline=True,

        linecolor='rgb(102, 102, 102)',

        titlefont=dict(color='rgb(204, 204, 204)'),

        tickfont=dict(color='rgb(102, 102, 102)',),

        showticklabels=True,

        ticks='outside',

        tickcolor='rgb(102, 102, 102)',

    ),

    margin=dict(l=140,r=40,b=50,t=80),

    legend=dict(

        font=dict(size=10,),

        yanchor='middle',

        xanchor='right',

    ),

    hovermode='closest',

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
haul5 = bowlers_stats.sort_values(by='haul_5',ascending=False).head(10)

haul4 = bowlers_stats.sort_values(by='haul_4',ascending=False).head(10)
trace1 = go.Scatter(x=haul5['bowler'],y=haul5['haul_5'],name='5 Wickets Haul',marker =dict(color= "gold",size = 13),line=dict(width=3,dash='longdashdot'))

trace2 = go.Scatter(x=haul4['bowler'],y=haul4['haul_4'],name='4 Wickets Haul',marker =dict(color= "lightgrey",size = 13),line=dict(width=3))



fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('Five Wickets','Four Wickets'), print_grid=False)



fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 1, 2)



iplot(fig)
wicket = bowlers_stats.sort_values(by='wickets',ascending=False).head(10)

caught_bowled = bowlers_stats.sort_values(by='caught and bowled',ascending=False).head(10)
dismissals = bowlers_stats[['bowler','wickets','caught and bowled']]

dismissals['dismissals'] = dismissals['wickets']+dismissals['caught and bowled']



dismissals['points'] = (dismissals['wickets']*10) + (dismissals['caught and bowled']*14)

dismissals.sort_values(by='points',ascending=False,inplace=True)
trace = go.Table(

    domain=dict(x=[0, 0.52],

                y=[0, 1.0]),

    header=dict(values=["Bowler","Dismissals","Points"],

                fill = dict(color = 'red'),

                font = dict(color = 'white', size = 14),

                align = ['center'],

               height = 30),

    cells=dict(values=[dismissals['bowler'].head(10),dismissals['dismissals'].head(10),dismissals['points'].head(10)],

               fill = dict(color = ['lightsalmon', 'white']),

               align = ['center']))



trace1 = go.Bar(x=dismissals['bowler'].head(10),

                y=dismissals["points"].head(10),

                xaxis='x1',

                yaxis='y1',

                marker=dict(color='lightblue'),opacity=0.60)



layout = dict(

    width=830,

    height=410,

    autosize=False,

    title='Bowlers with maximum dismissal points',

    showlegend=False,   

    xaxis1=dict(**dict(domain=[0.58, 1], anchor='y1', showticklabels=True)),

    yaxis1=dict(**dict(domain=[0, 1.0], anchor='x1', hoverformat='.2f')),  

)



fig1 = dict(data=[trace, trace1], layout=layout)

iplot(fig1)
e1 = bowlers_stats.sort_values(by='eco_below4',ascending=False).head(10)

e2 = bowlers_stats.sort_values(by='eco_4-5',ascending=False).head(10)

e3 = bowlers_stats.sort_values(by='eco_5-6',ascending=False).head(10)

e4 = bowlers_stats.sort_values(by='eco_6-9',ascending=False).head(10)

e5 = bowlers_stats.sort_values(by='eco_9-10',ascending=False).head(10)

e6 = bowlers_stats.sort_values(by='eco_10-11',ascending=False).head(10)

e7 = bowlers_stats.sort_values(by='eco_above11',ascending=False).head(10)

m = bowlers_stats.sort_values(by='maiden_overs',ascending=False).head(10)
trace1 = go.Scatter(x=e1['bowler'],y = e1['eco_below4'],name='below 4')

trace2 = go.Scatter(x=e2['bowler'],y = e2['eco_4-5'],name='between 4-5')

trace3 = go.Scatter(x=e3['bowler'],y = e3['eco_5-6'],name='between 5-6')

trace4 = go.Scatter(x=e4['bowler'],y = e4['eco_6-9'],name='between 6-9')

trace5 = go.Scatter(x=e5['bowler'],y = e5['eco_9-10'],name='between 9-10')

trace6 = go.Scatter(x=e6['bowler'],y = e6['eco_10-11'],name='between 10-11')

trace7 = go.Scatter(x=e7['bowler'],y = e7['eco_above11'],name='above 11')

trace8 = go.Scatter(x=m['bowler'],y = m['maiden_overs'],name='Maiden overs')



fig = tools.make_subplots(rows=4, cols=2,print_grid=False,

                          subplot_titles=('Economy below 4','Economy between 4-5','Economy between 5-6',

                                          'Economy between 6-9','Economy between 9-10','Economy between 10-11',

                                          'Economy above 11','Maiden Overs'))



fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 1, 2)

fig.append_trace(trace3, 2, 1)

fig.append_trace(trace4, 2, 2)

fig.append_trace(trace5, 3, 1)

fig.append_trace(trace6, 3, 2)

fig.append_trace(trace7, 4, 1)

fig.append_trace(trace8, 4, 2)





fig['layout'].update(height=950, width=850,title='Economy and maiden Overs analysis',showlegend=False)

iplot(fig)
eco = bowlers_stats[['bowler','maiden_overs','eco_below4','eco_4-5','eco_5-6','eco_9-10','eco_10-11','eco_above11']]



eco['points'] = ((eco['eco_below4']*3)+(eco['eco_4-5']*2)+(eco['eco_5-6']*1)+

                 (eco['eco_9-10']*(-1))+(eco['eco_10-11']*(-2))+(eco['eco_above11']*(-3))+(eco['maiden_overs']*4))



eco.sort_values(by='points',ascending=False,inplace=True)
trace = go.Table(

    domain=dict(x=[0, 0.52],

                y=[0, 1.0]),

    header=dict(values=["Bowler","Maiden Overs","Points"],

                fill = dict(color = 'green'),

                font = dict(color = 'white', size = 14),

                align = ['center'],

               height = 30),

    cells=dict(values=[eco['bowler'].head(10),eco['maiden_overs'].head(10),eco['points'].head(10)],

               fill = dict(color = ['lightgreen', 'white']),

               align = ['center']))



trace1 = go.Bar(x=eco['bowler'].head(10),

                y=eco["points"].head(10),

                xaxis='x1',

                yaxis='y1',

                marker=dict(color='gray'),opacity=0.60,name='bowler')



layout = dict(

    width=830,

    height=410,

    autosize=False,

    title='Bowlers with maximum economy and maiden points',

    showlegend=False,   

    xaxis1=dict(**dict(domain=[0.56, 1], anchor='y1', showticklabels=True)),

    yaxis1=dict(**dict(domain=[0, 1.0], anchor='x1', hoverformat='.2f')),  

)



fig1 = dict(data=[trace, trace1], layout=layout)

iplot(fig1)
final = bowlers_stats

final['points_x'] = ((final['eco_below4']*3)+(final['eco_4-5']*2)+(final['eco_5-6']*1)+(final['eco_9-10']*(-1))+

                   (final['eco_10-11']*(-2))+(final['eco_above11']*(-3))+(final['maiden_overs']*4))



final['points_y'] = (final['wickets']*10) + (final['caught and bowled']*14)

final['points_z'] = (final['haul_4']*4) + (final['haul_5']*8)



final['points'] = final['points_x']+final['points_y']+final['points_z']

final['dismissals'] = final['wickets']+final['caught and bowled']



final.sort_values(by='points',ascending=False,inplace=True)

final_bowl = final.head(10)



best_bowler = final[['bowler','points']]
trace = go.Scatter(y = final_bowl['points'],x = final_bowl['bowler'],mode='markers',

                   marker=dict(size= final_bowl['dismissals'].values,

                               color = final_bowl['maiden_overs'].values,

                               colorscale='Viridis',

                               showscale=True,

                               colorbar = dict(title = 'Economy')),

                   text = final_bowl['dismissals'].values)



data = [(trace)]



layout= go.Layout(autosize= True,

                  title= 'Top Bowlers with maximum points',

                  hovermode= 'closest',

                  xaxis=dict(showgrid=False,zeroline=False,

                             showline=False),

                  yaxis=dict(title= 'Best Bowlers',ticklen= 5,

                             gridwidth= 2,showgrid=False,

                             zeroline=False,showline=False),

                  showlegend= False)



fig = go.Figure(data=data, layout=layout)

iplot(fig)

final['points_x'] = (final['points_x'])/(final['points_x'].max()/100)

final['points_y'] = (final['points_y'])/(final['points_y'].max()/100)

final['points_z'] = (final['points_z'])/(final['points_z'].max()/100)

final['points'] = (final['points'])/(final['points'].max()/100)
x = final[final["bowler"] == "Harbhajan Singh"]

y = final[final["bowler"] == "SP Narine"]

z = final[final["bowler"] == "R Ashwin"]

w = final[final["bowler"] == "B Kumar"]



data = [go.Scatterpolar(

  r = [x['points_x'].values[0],x['points_y'].values[0],x['points_z'].values[0],x['points'].values[0]],

  theta = ['Economy rating','Wicket Haul rating','Wickets rating','Total rating'],

  fill = 'toself', opacity = 0.8,

  name = "Harbhajan Singh"),

        

    go.Scatterpolar(

  r = [y['points_x'].values[0],y['points_y'].values[0],y['points_z'].values[0],y['points'].values[0]],

  theta = ['Economy rating','Wicket Haul rating','Wickets rating','Total rating'],

  fill = 'toself',subplot = "polar2",

    name = "SP Narine"),

       

    go.Scatterpolar(

  r = [z['points_x'].values[0],z['points_y'].values[0],z['points_z'].values[0],z['points'].values[0]],

  theta = ['Economy rating','Wicket Haul rating','Wickets rating','Total rating'],

  fill = 'toself',subplot = "polar3",

    name = "R Ashwin"),

       

    go.Scatterpolar(

  r = [w['points_x'].values[0],w['points_y'].values[0],w['points_z'].values[0],w['points'].values[0]],

  theta = ['Economy rating','Wicket Haul rating','Wickets rating','Total rating'],

  fill = 'toself',subplot = "polar4",

    name = "B Kumar")]



layout = go.Layout(title = "Comparison Between Harbhajan Singh, SP Narine, R Ashwin, B Kumar",

                   

                   polar = dict(radialaxis = dict(visible = True,range = [0, 100]),

                   domain = dict(x = [0, 0.40],y = [0, 0.40])),

                  

                   polar2 = dict(radialaxis = dict(visible = True,range = [0, 100]),

                   domain = dict(x = [0.60, 1],y = [0, 0.40])),

                  

                   polar3 = dict(radialaxis = dict(visible = True,range = [0, 100]),

                   domain = dict(x = [0, 0.40],y = [0.60, 1])),

                  

                   polar4 = dict(radialaxis = dict(visible = True,range = [0, 100]),

                   domain = dict(x = [0.60, 1.0],y = [0.60, 1])))



fig = go.Figure(data=data, layout=layout)

iplot(fig)
best_batsman = best_batsman.rename(columns={"batsman": "player"})

best_bowler = best_bowler.rename(columns={"bowler": "player"})

best_fielder = best_fielder.rename(columns={"fielder": "player"})
best_player = pd.merge(pd.merge(best_batsman,best_bowler,on='player',how='outer'),best_fielder,on='player',how='outer')



best_player = best_player.fillna(0)

best_player['points'] = best_player['total_points']+best_player['points_x']+best_player['points_y']

best_player.sort_values(by='points',ascending=False,inplace=True)

best_player=best_player.reset_index().drop(['index'],axis=1)



best_player = best_player.head(20)
trace1 = go.Bar(

    x=best_player['player'],

    y=best_player['total_points'],

    name='Batting points',opacity=0.8,

    marker=dict(color='lightblue'))



trace2 = go.Bar(

    x=best_player['player'],

    y=best_player['points_x'],

    name='Bowling points',opacity=0.7,

    marker=dict(color='gold'))



trace3 = go.Bar(

    x=best_player['player'],

    y=best_player['points_y'],

    name='Fielding points',opacity=0.7,

    marker=dict(color='lightgreen'))





data = [trace1, trace2, trace3]

layout = go.Layout(title="Points Distribution of Top Players",barmode='stack',xaxis = dict(tickmode='linear'),

                                    yaxis = dict(title= "Points Distribution"))



fig = go.Figure(data=data, layout=layout)

iplot(fig)