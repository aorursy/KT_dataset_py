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
import numpy as np 
import pandas as pd
import datetime
import seaborn as sns 
import matplotlib.pyplot as plt
# Plotly Libraris
import plotly.express as px
import plotly.graph_objects as go


import warnings
warnings.filterwarnings("ignore")
Football_Results=pd.read_csv("/kaggle/input/womens-international-football-results/results.csv")
df=Football_Results.copy()

df.head()
df.info()
df.describe().T
df.isnull().values.any()
df.isnull().sum()
df['date']=pd.to_datetime(df['date'])
df['year']=pd.to_datetime(df['date']).dt.year
df['month']=pd.to_datetime(df['date']).dt.month
df['month_name']=df['date'].dt.strftime('%B')
df['total_score']=df['home_score']+df['away_score']
df['match_result']=np.where((df['home_score']>df['away_score']),'Home Team Win',
                np.where(df['home_score']==df['away_score'],'Draw',
np.where((df['home_score']<df['away_score']),'Away Team Win',"Not Specified")))
df.head()
df.info()
# Pie with custom colors

df['match_result']=np.where((df['home_score']>df['away_score']),'Home Team Win',np.where(df['home_score']==df['away_score'],'Draw',
np.where((df['home_score']<df['away_score']),'Away Team Win',"Not Specified")))

df_result=df.groupby('match_result')['city'].count().reset_index()

colors=['lightcyan','cyan',"darkcyan"]
fig = go.Figure([go.Pie(labels=df_result['match_result'], values=df_result['city'])])
fig.update_traces(hoverinfo='label+percent', textinfo='percent+value', textfont_size=15,
                 marker=dict(colors=colors, line=dict(color='#000000', width=2)))
fig.update_layout(title="Match Result",title_x=0.5)
fig.show()
# Bar Chart - Gradient & Text Position

df_month=df['month_name'].value_counts().reset_index().rename(columns={'index':'month_name','month_name':'Count'})

#Sort
custom_dict ={"January":0,"February":1,"March":2, "April":3,"May":4,"June":5,"July":6,"August":7,"September":8,"October":9,"November":10,"December":11}
df_month['month_name'] = pd.Categorical(df_month['month_name'], categories=sorted(custom_dict, key=custom_dict.get), ordered=True)
df_month=df_month.sort_values('month_name').reset_index(drop=True)

fig = go.Figure(go.Bar(
    x=df_month['month_name'],y=df_month['Count'],
    marker={'color': df_month['Count'], 
    'colorscale': 'Viridis'},  
    text=df_month['Count'],
    textposition = "outside",
))
fig.update_layout(title_text='(1969-2020) Monthly  Match Counter Report',xaxis_title="Month",yaxis_title="Count",title_x=0.5)
fig.show()
# Bar Chart - Gradient & Text Position

df_month_score=df.groupby('month_name')['total_score'].sum().reset_index()
df_month_score.columns = ['month_name', 'Total_Score']

custom_dict ={"January":0,"February":1,"March":2, "April":3,"May":4,"June":5,"July":6,"August":7,"September":8,"October":9,"November":10,"December":11}
df_month_score['month_name'] = pd.Categorical(df_month_score['month_name'], categories=sorted(custom_dict, key=custom_dict.get), ordered=True)
df_month_score=df_month_score.sort_values('month_name').reset_index(drop=True)

fig = go.Figure(go.Bar(
    x=df_month_score['month_name'],y=df_month_score['Total_Score'],
    marker={'color': df_month_score['Total_Score'], 
    'colorscale': 'Viridis'},  
    text=df_month_score['Total_Score'],
    textposition = "outside",
))
fig.update_layout(title_text='(1969-2020) Monthly Total Score Report',xaxis_title="Month",yaxis_title="Count",title_x=0.5)
fig.show()
#  Bubble Plot with Color gradient

df_tournament=df['tournament'].value_counts().to_frame().reset_index().rename(columns={'index':'Tournament','tournament':'Count'})[0:6]


fig = go.Figure(data=[go.Scatter(
    x=df_tournament['Tournament'], y=df_tournament['Count'],
    mode='markers',
    marker=dict(
        color=df_tournament['Count'],
        size=df_tournament['Count']*0.05,
        showscale=True
    ))])

fig.update_layout(title='(1969-2020) Top Six Tournament Match Count',
                  xaxis_title="Tournament Name",
                  yaxis_title=" Match Count",
                  title_x=0.5)
fig.show()
# Horizontal Bar Chart

df_city=df['city'].value_counts().to_frame().reset_index().rename(columns={'index':'City','city':'Count'}).sort_values('Count',ascending="False")
df_city=df_city[945:960]

fig = go.Figure(go.Bar(y=df_city['City'], x=df_city['Count'], # Need to revert x and y axis
                      orientation="h")) # default orentation value is "v" - vertical ,we need to change it as orientation="h"
fig.update_layout(title_text='Top 15 Citys Match Was Played',
                  xaxis_title="Count ",
                  yaxis_title="Citys",
                  title_x=0.5)
fig.show()
# Bar Chart - Gradient & Text Position

df_country=df['country'].value_counts().to_frame().reset_index().rename(columns={'index':'Country','country':'Count'})[0:15]

fig = go.Figure(go.Bar(
    x=df_country['Country'],y=df_country['Count'],
    marker={'color': df_country['Count'], 
    'colorscale': 'Viridis'},  
    text=df_country['Count'],
    textposition = "outside",
))
fig.update_layout(title_text='Top 15 Country Match Was Played',xaxis_title="Country",yaxis_title="Count",title_x=0.5)
fig.show()
# Bar Chart - Gradient & Text Position

df_score=df['total_score'].value_counts().to_frame().reset_index().rename(columns={'index':'Total_Score','total_score':'Count'}).sort_values('Total_Score',ascending="False")

fig = go.Figure(go.Bar(
    x=df_score['Total_Score'],y=df_score['Count'],
    marker={'color': df_score['Count'], 
    'colorscale': 'Viridis'},  
    text=df_score['Count'],
    textposition = "outside",
))
fig.update_layout(title_text=' Distribution of Goals ',
                  xaxis_title="Goals",
                  yaxis_title="Count",
                  title_x=0.5)
fig.show()
# Tables - Cell Colo

df_max=df[df['total_score']==26][['date','home_team','away_team','home_score','away_score','tournament','total_score']]
df_max['date']="2006-11-10"

tab_df=df_max

colors=['lightblue','lightpink','lightgreen','yellow','lightseagreen']
    
fig = go.Figure(data=[go.Table(header=dict(values=['Date', 'Home Team','Away Team','Home Score','Away Score','Tournament'],
                                          line_color='white', fill_color='gray',
                                  align='center',font=dict(color='white', size=12)
                                          ),
                               
                 cells=dict( values=[tab_df['date'], tab_df['home_team'],tab_df['away_team'],tab_df['home_score'],tab_df['away_score'],tab_df['tournament']],
                           line_color=colors, fill_color=colors,
                           align='center', font=dict(color='black', size=11))
                              )])
                      
fig.show()
# Bar Chart - Gradient & Text Position

df_tournament=df.groupby(['tournament'])['total_score'].agg('sum').to_frame().reset_index().rename(columns={'tournament':'Tournament','total_score':'Total Score'})
df_tournament=df_tournament.sort_values(by='Total Score', ascending=False)[0:20]

fig = go.Figure(go.Bar(
    x=df_tournament['Tournament'],y=df_tournament['Total Score'],
    marker={'color': df_tournament['Total Score'], 
    'colorscale': 'Viridis'},  
    text=df_tournament['Total Score'],
    textposition = "outside",
))
fig.update_layout(title_text='Top 20 Tournament Total Score',
                  xaxis_title="Tournament",
                  yaxis_title="Total Score",
                  title_x=0.5)
fig.show()
# Bar Chart - Gradient & Text Position

df_tournament=df.groupby(['tournament'])['total_score'].agg('mean').to_frame().reset_index().rename(columns={'tournament':'Tournament','total_score':'Total Score'})
df_tournament=df_tournament.sort_values(by='Total Score', ascending=False)[0:20]

fig = go.Figure(go.Bar(
    x=df_tournament['Tournament'],y=df_tournament['Total Score'],
    marker={'color': df_tournament['Total Score'], 
    'colorscale': 'Viridis'},  
    #text=df_tournament['Total Score'],
    textposition = "outside",
))
fig.update_layout(title_text='Top 20 Tournament Mean Score',
                  xaxis_title="Tournament",
                  yaxis_title="Mean Score",
                  title_x=0.5)
fig.show()
# Multiple Bullet

df_Worl_2019=df[df['tournament']=='FIFA World Cup']
df_Worl_2019=df_Worl_2019[df_Worl_2019['year']==2019]

mach_palyed=df_Worl_2019['country'].count()

total_score=df_Worl_2019['total_score'].sum()

mean_score=df_Worl_2019['total_score'].mean()

max_score=df_Worl_2019['total_score'].max()

fig = go.Figure()

fig.add_trace(go.Indicator(
    mode = "number+gauge", value =  max_score,
    domain = {'x': [0.25, 1], 'y': [0.2, 0.3]},
    title = {'text': "Max Score",'font':{'color': 'black','size':17}},
     number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None, 20]},
        'bar': {'color': "blue"}}))

fig.add_trace(go.Indicator(
    mode = "number+gauge", value = total_score,
    domain = {'x': [0.25, 1], 'y': [0.4, 0.5]},
    title = {'text': "Total Score",'font':{'color': 'black','size':17}},
    number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None,200]},
        'bar': {'color': "blue"}}))

fig.add_trace(go.Indicator(
    mode = "number+gauge", value = mean_score,
    domain = {'x': [0.25, 1], 'y': [0.6, 0.7]},
    title = {'text' :"Mean Score",'font':{'color': 'black','size':17}},
     number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None,10]},
        'bar': {'color': "blue"}}
))
fig.add_trace(go.Indicator(
    mode = "number+gauge", value = mach_palyed,
    domain = {'x': [0.25, 1], 'y': [0.8, 0.9]},
    title = {'text' :"Mach Played",'font':{'color': 'black','size':17}},
     number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None,75]},
        'bar': {'color': "blue"}}
))
fig.update_layout(title="FIFA World Cup 2019 France ",title_x=0.5)
fig.show()

df_Worl_2019_home_score=df_Worl_2019.groupby(by =['home_team'])['home_score'].sum().to_frame().reset_index().rename(columns={'home_team':'home_team','home_score':'Score1'})
df_Worl_2019_home_score=df_Worl_2019_home_score.sort_values(by='home_team', ascending=False)
df_Worl_2019_home_score.columns = ['Team', 'Score']

df_Worl_2019_away_score=df_Worl_2019.groupby(by =['away_team'])['away_score'].sum().to_frame().reset_index()
df_Worl_2019_away_score=df_Worl_2019_away_score.sort_values(by='away_team', ascending=False)
df_Worl_2019_away_score.columns = ['Team', 'Score']

df_World_Cup2019=pd.concat([df_Worl_2019_away_score,df_Worl_2019_home_score],ignore_index=True)

total_score_World_Cup_2019=df_World_Cup2019.groupby(by =['Team'])['Score'].sum().to_frame().reset_index()
total_score_World_Cup_2019.columns = ['Team','Total_Score']
total_score_World_Cup_2019=total_score_World_Cup_2019.sort_values(by='Total_Score', ascending=False)

fig = go.Figure(go.Bar(
    x=total_score_World_Cup_2019['Team'],y=total_score_World_Cup_2019['Total_Score'],
    marker={'color': total_score_World_Cup_2019['Total_Score'], 
    'colorscale': 'Viridis'},  
    text=total_score_World_Cup_2019['Total_Score'],
    textposition = "outside",
))
fig.update_layout(title_text='Score FIFA World Cup 2019 France',xaxis_title="Country",yaxis_title="Number of Score",title_x=0.5)
fig.show()
# Tables - Cell Colo

#FIFA World Cup 2019 France  Final Macth

df_Worl_2019=df[df['tournament']=='FIFA World Cup']
df_Worl_2019=df_Worl_2019[df_Worl_2019['year']==2019]

df_final=df_Worl_2019[df_Worl_2019['date']=='2019-07-07']

tab_df=df_final

colors=['lightblue','lightpink','lightgreen','yellow','lightseagreen']
    
fig = go.Figure(data=[go.Table(header=dict(values=['Date', 'Home Team','Away Team','Home Score','Away Score','Tournament'],
                                          line_color='white', fill_color='gray',
                                  align='center',font=dict(color='white', size=12)
                                          ),
                               
                 cells=dict( values=[tab_df['date'], tab_df['home_team'],tab_df['away_team'],tab_df['home_score'],tab_df['away_score'],tab_df['tournament']],
                           line_color=colors, fill_color=colors,
                           align='center', font=dict(color='black', size=11))
                              )])
                      
fig.show()
# Multiple Bullet

df_UEFA_Euro_2017=df[df['tournament']=='UEFA Euro']
df_UEFA_Euro_2017=df_UEFA_Euro_2017[df_UEFA_Euro_2017['year']==2017]

mach_palyed=df_UEFA_Euro_2017['country'].count()

total_score=df_UEFA_Euro_2017['total_score'].sum()

mean_score=df_UEFA_Euro_2017['total_score'].mean()

max_score=df_UEFA_Euro_2017['total_score'].max()

fig = go.Figure()

fig.add_trace(go.Indicator(
    mode = "number+gauge", value =  max_score,
    domain = {'x': [0.25, 1], 'y': [0.2, 0.3]},
    title = {'text': "Max Score",'font':{'color': 'black','size':17}},
     number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None, 20]},
        'bar': {'color': "blue"}}))

fig.add_trace(go.Indicator(
    mode = "number+gauge", value = total_score,
    domain = {'x': [0.25, 1], 'y': [0.4, 0.5]},
    title = {'text': "Total Score",'font':{'color': 'black','size':17}},
    number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None,100]},
        'bar': {'color': "blue"}}))

fig.add_trace(go.Indicator(
    mode = "number+gauge", value = mean_score,
    domain = {'x': [0.25, 1], 'y': [0.6, 0.7]},
    title = {'text' :"Mean Score",'font':{'color': 'black','size':17}},
     number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None,10]},
        'bar': {'color': "blue"}}
))
fig.add_trace(go.Indicator(
    mode = "number+gauge", value = mach_palyed,
    domain = {'x': [0.25, 1], 'y': [0.8, 0.9]},
    title = {'text' :"Mach Played",'font':{'color': 'black','size':17}},
     number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None,50]},
        'bar': {'color': "blue"}}
))
fig.update_layout(title="UEFA Euro Cup 2017 Netherlands ",title_x=0.5)
fig.show()
df_UEFA_Euro_2017_home_score=df_UEFA_Euro_2017.groupby(by =['home_team'])['home_score'].sum().to_frame().reset_index().rename(columns={'home_team':'home_team','home_score':'Score1'})
df_UEFA_Euro_2017_home_score=df_UEFA_Euro_2017_home_score.sort_values(by='home_team', ascending=False)
df_UEFA_Euro_2017_home_score.columns = ['Team', 'Score']

df_UEFA_Euro_2017_away_score=df_UEFA_Euro_2017.groupby(by =['away_team'])['away_score'].sum().to_frame().reset_index()
df_UEFA_Euro_2017_away_score=df_UEFA_Euro_2017_away_score.sort_values(by='away_team', ascending=False)
df_UEFA_Euro_2017_away_score.columns = ['Team', 'Score']

df_UEFA_Euro_Cup_2017=pd.concat([df_UEFA_Euro_2017_home_score,df_UEFA_Euro_2017_away_score],ignore_index=True)

total_score_UEFA_Euro_Cup_2017=df_UEFA_Euro_Cup_2017.groupby(by =['Team'])['Score'].sum().to_frame().reset_index()
total_score_UEFA_Euro_Cup_2017.columns = ['Team','Total_Score']
total_score_UEFA_Euro_Cup_2017=total_score_UEFA_Euro_Cup_2017.sort_values(by='Total_Score', ascending=False)

fig = go.Figure(go.Bar(
    x=total_score_UEFA_Euro_Cup_2017['Team'],y=total_score_UEFA_Euro_Cup_2017['Total_Score'],
    marker={'color': total_score_UEFA_Euro_Cup_2017['Total_Score'], 
    'colorscale': 'Viridis'},  
    text=total_score_UEFA_Euro_Cup_2017['Total_Score'],
    textposition = "outside",
))
fig.update_layout(title_text='Score UEFA Euro Cup 2017 Netherlands',xaxis_title="Country",yaxis_title="Number of Score",title_x=0.5)
fig.show()
# Tables - Cell Colo

#UEFA Euro 2017 Netherlands Final Macth

df_UEFA_Euro_2017=df[df['tournament']=='UEFA Euro']
df_UEFA_Euro_2017=df_UEFA_Euro_2017[df_UEFA_Euro_2017['year']==2017]

df_final=df_UEFA_Euro_2017[df_UEFA_Euro_2017['date']=='2017-08-06']

tab_df=df_final

colors=['lightblue','lightpink','lightgreen','yellow','lightseagreen']
    
fig = go.Figure(data=[go.Table(header=dict(values=['Date', 'Home Team','Away Team','Home Score','Away Score','Tournament'],
                                          line_color='white', fill_color='gray',
                                  align='center',font=dict(color='white', size=12)
                                          ),
                               
                 cells=dict( values=[tab_df['date'], tab_df['home_team'],tab_df['away_team'],tab_df['home_score'],tab_df['away_score'],tab_df['tournament']],
                           line_color=colors, fill_color=colors,
                           align='center', font=dict(color='black', size=11))
                              )])
                      
fig.show()
# Gauge Indicator

df_United_States_home=df[df['home_team']=='United States']
df_United_States_away=df[df['away_team']=='United States']

df_United_States_home_score=df_United_States_home.groupby(by =['home_team'])['home_score'].sum().to_frame().reset_index()
df_United_States_home_score.columns = ['Team', 'Total_Score']

df_United_States_away_score=df_United_States_away.groupby(by =['away_team'])['away_score'].sum().to_frame().reset_index()
df_United_States_away_score.columns = ['Team', 'Total_Score']

df_United_States_Total_Score=pd.concat([df_United_States_home_score,df_United_States_away_score],ignore_index=True)

df_United_States_Total_Score2=df_United_States_Total_Score.groupby(by =['Team'])['Total_Score'].sum().to_frame().reset_index()
df_United_States_Total_Score2.columns = ['Team','Total_Score']

Total_Score=df_United_States_Total_Score2['Total_Score'].max()

fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    gauge = {
       'axis': {'range': [None, 750]}},
    value = Total_Score,
    title = {'text': "United States Total Score"},
    domain = {'x': [0, 1], 'y': [0, 1]}
))
fig.show()
# Multiple Bullet Gauge  

df_United_States_home=df[df['home_team']=='United States']

df_United_States_away=df[df['away_team']=='United States']


mach_palyed1=df_United_States_home['country'].count()

mach_palyed2=df_United_States_away['country'].count()

total_mach_palyed=mach_palyed1+mach_palyed2


home_score=df_United_States_home['home_score'].sum()

away_score=df_United_States_away['away_score'].sum()

total_score=home_score+away_score

home_score_other=df_United_States_home['away_score'].sum()

away_score_other=df_United_States_away['home_score'].sum()

concede_a_goal=home_score_other+away_score_other

mean_score=total_score/total_mach_palyed


max_score_home=df_United_States_home['home_score'].max()


max_score_away=df_United_States_away['away_score'].max()


max_score=max_score_home

fig = go.Figure()

fig.add_trace(go.Indicator(
    mode = "number+gauge", value =  max_score,
    domain = {'x': [0.25, 1], 'y': [0.1, 0.2]},
    title = {'text': "Max Score",'font':{'color': 'black','size':17}},
     number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None, 20]},
        'bar': {'color': "blue"}}))

fig.add_trace(go.Indicator(
    mode = "number+gauge", value = total_score,
    domain = {'x': [0.25, 1], 'y': [0.3, 0.4]},
    title = {'text': "Total Score",'font':{'color': 'black','size':17}},
    number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None,750]},
        'bar': {'color': "cyan"}}))

fig.add_trace(go.Indicator(
    mode = "number+gauge", value = mean_score,
    domain = {'x': [0.25, 1], 'y': [0.5, 0.6]},
    title = {'text' :"Mean Score",'font':{'color': 'black','size':17}},
     number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None,10]},
        'bar': {'color': "darkblue"}}
))
fig.add_trace(go.Indicator(
    mode = "number+gauge", value = total_mach_palyed,
    domain = {'x': [0.25, 1], 'y': [0.7, 0.8]},
    title = {'text' :"Mach Played",'font':{'color': 'black','size':17}},
     number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None,250]},
        'bar': {'color': "darkcyan"}}
))
fig.add_trace(go.Indicator(
    mode = "number+gauge", value = concede_a_goal,
    domain = {'x': [0.25, 1], 'y': [0.9,1]},
    title = {'text' :"Concede a Goal",'font':{'color': 'black','size':17}},
     number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None,200]},
        'bar': {'color': "red"}}
))
fig.update_layout(title=" United States ",title_x=0.5)
fig.show()
# Multiple Bullet Gauge  

df_US_home=df[df['home_team']=='United States']

mach_count=df[df['home_team']=='United States']["date"].count()

mach_home_win=df_US_home[df_US_home['match_result']=='Home Team Win']["date"].count()

mach_home_lose=df_US_home[df_US_home['match_result']=='Away Team Win']["date"].count()

mach_home_draw=df_US_home[df_US_home['match_result']=='Draw']["date"].count()

home_score=df_US_home['home_score'].sum()

away_team_score=df_US_home['away_score'].sum()

fig = go.Figure()

fig.add_trace(go.Indicator(
    mode = "number+gauge", value =  mach_home_win,
    domain = {'x': [0.25, 1], 'y': [0.9, 1]},
    title = {'text': "Home Win",'font':{'color': 'black','size':17}},
     number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None, 150]},
        'bar': {'color': "green"}}))

fig.add_trace(go.Indicator(
    mode = "number+gauge", value = mach_home_lose,
    domain = {'x': [0.25, 1], 'y': [0.5, 0.6]},
    title = {'text': "Home Lose",'font':{'color': 'black','size':17}},
    number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None,20]},
        'bar': {'color': "red"}}))

fig.add_trace(go.Indicator(
    mode = "number+gauge", value = mach_home_draw,
    domain = {'x': [0.25, 1], 'y': [0.7, 0.8]},
    title = {'text' :"Draw",'font':{'color': 'black','size':17}},
     number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None,50]},
        'bar': {'color': "darkblue"}}
))
fig.add_trace(go.Indicator(
    mode = "number+gauge", value =  home_score,
    domain = {'x': [0.25, 1], 'y': [0.3, 0.4]},
    title = {'text': "Home Score",'font':{'color': 'black','size':17}},
     number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None, 500]},
        'bar': {'color': "cyan"}}))
fig.add_trace(go.Indicator(
    mode = "number+gauge", value =  away_team_score,
    domain = {'x': [0.25, 1], 'y': [0.1, 0.2]},
    title = {'text': "Away Score",'font':{'color': 'black','size':17}},
     number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None, 100]},
        'bar': {'color': "darkcyan"}}))

fig.update_layout(title=" Home Match United States ",title_x=0.5)
fig.show()
# Multiple Bullet Gauge  

df_US_away=df[df['away_team']=='United States']

away_score=df_US_away['away_score'].sum()

mach_count=df[df['away_team']=='United States']["date"].count()


mach_home_win=df_US_away[df_US_away['match_result']=='Home Team Win']["date"].count()


mach_home_lose=df_US_away[df_US_away['match_result']=='Away Team Win']["date"].count()


mach_home_draw=df_US_away[df_US_away['match_result']=='Draw']["date"].count()


home_score=df_US_away['home_score'].sum()

away_team_score=df_US_away['away_score'].sum()

fig = go.Figure()

fig.add_trace(go.Indicator(
    mode = "number+gauge", value =  mach_home_win,
    domain = {'x': [0.25, 1], 'y': [0.9, 1]},
    title = {'text': "Home Win",'font':{'color': 'black','size':17}},
     number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None, 30]},
        'bar': {'color': "red"}}))

fig.add_trace(go.Indicator(
    mode = "number+gauge", value = mach_home_lose,
    domain = {'x': [0.25, 1], 'y': [0.5, 0.6]},
    title = {'text': "Home Lose",'font':{'color': 'black','size':17}},
    number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None,100]},
        'bar': {'color': "green"}}))

fig.add_trace(go.Indicator(
    mode = "number+gauge", value = mach_home_draw,
    domain = {'x': [0.25, 1], 'y': [0.7, 0.8]},
    title = {'text' :"Draw",'font':{'color': 'black','size':17}},
     number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None,50]},
        'bar': {'color': "darkblue"}}
))
fig.add_trace(go.Indicator(
    mode = "number+gauge", value =  home_score,
    domain = {'x': [0.25, 1], 'y': [0.3, 0.4]},
    title = {'text': "Home Score",'font':{'color': 'black','size':17}},
     number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None, 100]},
        'bar': {'color': "cyan"}}))
fig.add_trace(go.Indicator(
    mode = "number+gauge", value =  away_team_score,
    domain = {'x': [0.25, 1], 'y': [0.1, 0.2]},
    title = {'text': "Away Score",'font':{'color': 'black','size':17}},
     number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None, 200]},
        'bar': {'color': "darkcyan"}}))

fig.update_layout(title=" Away Match United States ",title_x=0.5)
fig.show()