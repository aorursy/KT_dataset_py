import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
import seaborn as sns
match_df = pd.read_csv('../input/WorldCupMatches.csv', encoding='Latin-1')
cup_df = pd.read_csv('../input/WorldCups.csv', encoding='Latin-1')
match_df.head()
match_df = match_df.dropna()
match_df['Year'] = match_df['Year'].astype(int)
match_df['date'] = match_df['Datetime'].str.split('-').str[0]
match_df['Stadium'] = match_df['Stadium'].str.split().str[0]
match_df = match_df.replace('Germany FR','Germany')
match_df = match_df.drop_duplicates(subset='MatchID', keep='first')
cup_df.head()
cup_df = cup_df.replace('Germany FR','Germany')
cup_df["Attendance"] = cup_df['Attendance'].str.replace('.','').astype(int)
cup_df['Year'] = cup_df['Year'].astype(str)
cup_df.head(3)
winner_count_df = cup_df['Winner'].value_counts().reset_index()
winner_count_df.columns = ['country','count']
winner_count_df
winner_year_df = cup_df.groupby('Winner')['Year'].apply(' '.join).reset_index()
# TypeError: sequence item 0: expected str instance, int found
winner_year_df.columns = ['country','year_str']
winner_year_df
winner_df = pd.merge(winner_year_df,winner_count_df,on='country')
winner_df
prize_list = ['Winner','Runners-Up','Third']
prize_df_list = []
for prize in prize_list:
    prize_count = cup_df[prize].value_counts().reset_index()
    prize_count.columns = ['country','{}_count'.format(prize)]
    prize_year = cup_df.groupby(prize)['Year'].apply(' '.join).reset_index()
    prize_year.columns = ['country', '{}_year_str'.format(prize)]
    prize_df = pd.merge(prize_year,prize_count,on='country')
    prize_df_list.append(prize_df)
# all_df = pd.merge(prize_df_list,on='country',how='outer')
# TypeError: merge() missing 1 required positional argument: 'right'

all_df = prize_df_list[0].merge(prize_df_list[1],on='country',how='outer').merge(prize_df_list[2],on='country',how='outer')
all_df = all_df.sort_values(by=['Winner_count','Runners-Up_count','Third_count'], ascending=False)
all_df
# can not write text on bar plot, year_str doesn't make sense
# all_df.plot(kind='barh',x='country',figsize=(12,24),colormap=['gold','silver','brown'])
all_df.plot(kind='bar',x='country',y=['Winner_count','Runners-Up_count','Third_count'],figsize=(18,6),color =['gold','silver','brown'],
           linewidth=0.7, edgecolor='w',fontsize=15,width=0.8, align='center')
# width: bar/bin width
# color=['red','blue','#d88c03']
# plt.grid(True)
plt.xlabel('Countries')
plt.ylabel('Number of podium')
plt.title('Number of podium by country')
# can not write text on!!!!!
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
data= []

# each x:country has the ammount of y:prize divided by name/type:prize
# easy to understand to see the data
for prize in prize_list:
    country = all_df['country']
    count = all_df['{}_count'.format(prize)]
    data.append(
        go.Bar(
            x=country,
            y=count,
            name = prize,
        )
    )

layout = go.Layout(
    barmode = "stack", 
    title = "Number of podium by country",
#     showlegend = False
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='pyplot-fifa')
data
match_df.head(3)
home_goal_df = match_df.groupby('Home Team Name')['Home Team Goals'].sum().reset_index()
home_goal_df.columns = ['country','home_goal']
home_goal_df.head(3)
away_goal_df = match_df.groupby('Away Team Name')['Away Team Goals'].sum().reset_index()
away_goal_df.columns = ['country','away_goal']
away_goal_df.head(3)
country_goal_df = pd.merge(home_goal_df,away_goal_df,on='country')
country_goal_df['total_goal'] = country_goal_df['home_goal'] + country_goal_df['away_goal']
country_goal_df = country_goal_df.sort_values(by='total_goal', ascending=False)
country_goal_df.head()
# plt.figure(figsize=(10,6))
# sns.barplot(data=country_goal_df[:10],x='country',y='total_goal')
country_goal_df[:10].plot(kind='bar',y='total_goal',x='country')
plt.xlabel('Countries')
plt.ylabel('Number of goals')
plt.title('Top 10 of Number of goals by country')
plt.show()
cup_df.head(3)
plot_list = cup_df.columns.tolist()[-4:]
plot_list
plt.figure(figsize=(22,16))
for i,plot in enumerate(plot_list):
    plt.subplot('22{}'.format(i+1))
    ax = sns.barplot(data=cup_df,x='Year',y=plot, palette='Blues')
    ax.set_title('{} per cup'.format(plot),fontsize=16)
# can only show the last plot?????
# NO!!! data_df.plot() can not be used, sns.barplot() ok!
plt.subplots_adjust(wspace = 0.2, hspace = 0.4,top = 0.9)
plt.show()
match_df.head(3)
home_cup_goal = match_df.groupby(['Year','Home Team Name'])['Home Team Goals'].sum()
home_cup_goal
away_cup_goal = match_df.groupby(['Year', 'Away Team Name'])['Away Team Goals'].sum()
away_cup_goal
# country_cup_goal_df = pd.merge(home_cup_goal, away_cup_goal)
# ValueError: can not merge DataFrame with instance of type <class 'pandas.core.series.Series'>
country_cup_goal_df = pd.concat([home_cup_goal, away_cup_goal],axis=1)
country_cup_goal_df.head(15)
country_cup_goal_df = country_cup_goal_df.reset_index()
country_cup_goal_df.columns = ['year','country','home_goal','away_goal']
country_cup_goal_df.head(15)
country_cup_goal_df['total_goal'] = country_cup_goal_df['home_goal'] + country_cup_goal_df['away_goal']
country_cup_goal_df.head(3)
new_df = country_cup_goal_df.copy()
new_df = new_df.sort_values(by=['year','total_goal'], ascending=[True,False])
new_df = new_df.groupby('year').head(5)
new_df.head(10)
# new_df = pd.DataFrame.from_dict(new_df.to_dict())
# new_df = new_df.sort_values(by=['year','total_goal'], ascending=[True,False])
# new_df = new_df.groupby('year').head(5)

# no need to transform to dict and re-transform to df
country_list = new_df['country'].value_counts().index.tolist()
data = []
for country in country_list:
    year = new_df[new_df['country']==country]['year']
    goal = new_df[new_df['country']==country]['total_goal']
    data.append(
        go.Bar(
            x=year,
            y=goal,
            name=country
        )
    )
layout = go.Layout(
    barmode = "stack", 
    title = "Top 5 teams which scored the most goals",
    showlegend = False
)    
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='pyplot-fifa')
