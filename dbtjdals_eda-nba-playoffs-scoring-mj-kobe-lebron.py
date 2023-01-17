#import libraries



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import plotly.tools as tls

import plotly.offline as py

from plotly.offline import init_notebook_mode, iplot, plot

import plotly.graph_objs as go

import plotly.express as px

init_notebook_mode(connected=True)
#show visualizations

%matplotlib inline
#read each player's data

mj_playoffs = pd.read_csv('../input/mj playoffs.csv')

kobe_playoffs = pd.read_csv('../input/kobe playoffs.csv')

lebron_playoffs = pd.read_csv('../input/lebron playoffs.csv')
#preview Michael Jordan data

mj_playoffs.head()
#preview Kobe Bryant data

kobe_playoffs.head()
#preview LeBron James data

lebron_playoffs.head()
#rename date column for all tables

mj_playoffs.rename(columns = {'1985 Playoffs':'Date'}, inplace = True)

kobe_playoffs.rename(columns = {'1997 Playoffs':'Date'}, inplace = True)

lebron_playoffs.rename(columns = {'2006 Playoffs':'Date'}, inplace = True)



#assign player column to each player table

mj_playoffs['Player'] = 'Michael Jordan'

kobe_playoffs['Player'] = 'Kobe Bryant'

lebron_playoffs['Player'] = 'Lebron James'



#concatenate player tables

playoffs_mj_kb_lj = pd.concat([mj_playoffs,kobe_playoffs,lebron_playoffs])

#Rename the unnamed home/away column

playoffs_mj_kb_lj.rename(columns={'Unnamed: 5':'HomeAway'},inplace=True)



#replace null values with 'home' and @ values with 'away'

playoffs_mj_kb_lj['HomeAway'].fillna(value='Home',inplace=True)



# replace '@' values with 'away'

playoffs_mj_kb_lj['HomeAway'].replace('@','Away',inplace = True)



#Rename the unnamed win/loss column

playoffs_mj_kb_lj.rename(columns={'Unnamed: 8':'WinLossPlusMinus'},inplace=True)

#Remove inactive games

playoffs_mj_kb_lj= playoffs_mj_kb_lj[playoffs_mj_kb_lj.WinLossPlusMinus != 'Inactive']
#preview concatenated table

playoffs_mj_kb_lj.head()
#basic information

playoffs_mj_kb_lj.info()
#check all null values

playoffs_mj_kb_lj.isna().sum()
#number of playoff games played

playoffs_mj_kb_lj['Player'].value_counts()
#set style for visualization

sns.set(style='white')



#set plot size

plt.figure(figsize=(6,4))



#plot

sns.countplot(x='Player',

              data=playoffs_mj_kb_lj,

              order= playoffs_mj_kb_lj['Player'].value_counts().index,

             palette=['b','y','r']).set_title('Career NBA Playoff Games Played')
#compare playoff points per game averages bar plot

sns.barplot(x='Player',y='PTS',data=playoffs_mj_kb_lj,palette=['r','y','b']).set_title('Playoff Points Per Game')
#career playoff scoring games distribution swarm plot



plt.figure(figsize=(5,5))

sns.swarmplot(x='Player',y='PTS',data=playoffs_mj_kb_lj,palette=['r','y','b']).set_title('Playoff Scoring Games')
#career playoff scoring games distribution box plot



plt.figure(figsize=(5,5))

sns.boxplot(x='Player',y='PTS',data=playoffs_mj_kb_lj,palette=['r','y','b'],linewidth=1).set_title('Playoff Scoring Games')
#career playoff scoring games distribution violin plot



plt.figure(figsize=(5,5))

sns.violinplot(x='Player',y='PTS',data=playoffs_mj_kb_lj,palette=['r','y','b'],linewidth=1).set_title('Playoff Scoring Games')
#career playoff scoring games distribution plot

g = sns.FacetGrid(data=playoffs_mj_kb_lj,col='Player')

g.map(sns.distplot,'PTS',bins=30,kde=False)
#replace eastern & western conference series names with just playoff round numbers

playoffs_mj_kb_lj.replace('EC1','R1',inplace = True)

playoffs_mj_kb_lj.replace('ECS','R2',inplace = True)

playoffs_mj_kb_lj.replace('ECF','R3',inplace = True)

playoffs_mj_kb_lj.replace('WC1','R1',inplace = True)

playoffs_mj_kb_lj.replace('WCS','R2',inplace = True)

playoffs_mj_kb_lj.replace('WCF','R3',inplace = True)



playoffs_mj_kb_lj.replace('FIN','R4',inplace = True)
#pivot points per game average by series

playoffs_by_series = playoffs_mj_kb_lj.pivot_table(values='PTS', index=['Series'], columns=['Player'], aggfunc=np.mean)



#reorder rows

playoffs_by_series= playoffs_by_series.reindex(['R1','R2','R3','R4'])



#reorder columns

columnsTitles = ['Michael Jordan', 'Kobe Bryant', 'Lebron James']

playoffs_by_series = playoffs_by_series.reindex(columns=columnsTitles)



playoffs_by_series
sns.lineplot(data=playoffs_by_series,palette=['r','y','b'],

             dashes=False,markers=True).set_title('Playoffs Average Scoring by Series')



plt.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0.)
#convert date column to date time format

playoffs_mj_kb_lj['Date'] = pd.to_datetime(playoffs_mj_kb_lj['Date'])



#create year column

playoffs_mj_kb_lj['Year'] = playoffs_mj_kb_lj['Date'].apply(lambda time: time.year)
#create a pivot table of average playoff scoring by year

playoffs_years= playoffs_mj_kb_lj.pivot_table(values='PTS', index=['Year'], columns=['Player'], aggfunc=np.mean)

columnsTitles = ['Michael Jordan', 'Kobe Bryant', 'Lebron James']

playoffs_years = playoffs_years.reindex(columns=columnsTitles)

playoffs_years
sns.lineplot(data=playoffs_years,palette=['r','y','b'],

             dashes=False,markers=True).set_title('Playoffs Average Scoring by Year')



plt.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0.)
#plot points scored by year

px.scatter(playoffs_mj_kb_lj,x='Date',y='PTS',color='Player',color_discrete_sequence=['red','gold','blue'])
#visualize a linear model plot for playoff scoring by year

sns.lmplot(x='Year',y='PTS',data=playoffs_mj_kb_lj,col='Player')
#filter and create a dataframe of NBA Finals data only

finals_mj_kb_lj = playoffs_mj_kb_lj[playoffs_mj_kb_lj['Series'] == 'R4']
#number of finals games played

finals_mj_kb_lj['Player'].value_counts()
#compare Finals points per game averages



plt.figure(figsize=(5,5))

sns.barplot(x='Player',y='PTS',data=finals_mj_kb_lj,palette=['r','y','b']).set_title('Finals Points Per Game')
#career finals scoring game distribution



plt.figure(figsize=(5,5))

sns.stripplot(x='Player',y='PTS',data=finals_mj_kb_lj,palette=['r','y','b']).set_title('Finals Scoring Games')
#career finals scoring game distribution



plt.figure(figsize=(5,5))

sns.boxplot(x='Player',y='PTS',data=finals_mj_kb_lj,palette=['r','y','b'],linewidth=1).set_title('Finals Scoring Games')
#career finals scoring game distribution



plt.figure(figsize=(5,5))

sns.violinplot(x='Player',y='PTS',data=finals_mj_kb_lj,palette=['r','y','b'],linewidth=1).set_title('Finals Scoring Games')
#career finals scoring game distribution

g = sns.FacetGrid(data=finals_mj_kb_lj,col='Player')

g.map(sns.distplot,'PTS',kde=False)
px.scatter(finals_mj_kb_lj,x='Date',y='PTS',color='Player',color_discrete_sequence=['red','gold','blue'])
#visualize NBA Finals scoring by year

g= sns.lmplot(x='Year',y='PTS',data=finals_mj_kb_lj,col='Player')

g = (g.set_axis_labels("Year", "PTS").set(ylim=(0, 70)))
#create 2 point field goal column

playoffs_mj_kb_lj['2P']= playoffs_mj_kb_lj['FG']-playoffs_mj_kb_lj['3P']
#create column for points scored off 2 pointers; count of 2pt times 2

playoffs_mj_kb_lj['2PT'] = playoffs_mj_kb_lj['2P']*2



#create column for points scored off 3 pointers; count of 3pt times 3 

playoffs_mj_kb_lj['3PT'] = playoffs_mj_kb_lj['3P']*3
#create pivot table

scoring_methods = pd.pivot_table(data=playoffs_mj_kb_lj,index='Player',values=['2PT','3PT','FT'],aggfunc='mean')



#reorder rows

scoring_methods= scoring_methods.reindex(['Michael Jordan','Kobe Bryant','Lebron James'])



scoring_methods
#store values as series

Points_2Pointers = scoring_methods['2PT']

Points_3Pointers = scoring_methods['3PT']

Points_FT = scoring_methods['FT']
#create stacked bar chart; average Playoff Points by Scoring Method

fig = go.Figure(data=[

    go.Bar(name='2PT', x=['Michael Jordan','Kobe Bryant','Lebron James'], y=Points_2Pointers),

    go.Bar(name='3PT', x=['Michael Jordan','Kobe Bryant','Lebron James'], y=Points_3Pointers),

    go.Bar(name='FT', x=['Michael Jordan','Kobe Bryant','Lebron James'], y=Points_FT)])



# Change the bar mode

fig.update_layout(barmode='stack',title='Average Playoff Scoring by Method')

fig.show()
#Michael Jordan points scored vs. assists

sns.jointplot(x='AST',y='PTS',data=mj_playoffs,color='r',kind='reg')
#Kobe Bryant points scored vs. assists

sns.jointplot(x='AST',y='PTS',data=kobe_playoffs,color='y',kind='reg')
#LeBron James points scored vs. assists

sns.jointplot(x='AST',y='PTS',data=lebron_playoffs,color='b',kind='reg')
#check the minutes played column

playoffs_mj_kb_lj['MP'].head()
#separate the minutes played value rounded down and store to a new column as integers

playoffs_mj_kb_lj['Minutes Played']=playoffs_mj_kb_lj['MP'].apply(lambda x: x.split(':')[0]).astype(int)
#pivot minutes per playoff game for each player

playoffs_mj_kb_lj.pivot_table(values='Minutes Played',index='Player',aggfunc=['min','mean','max'])
#create rough interactive scatter plot

px.scatter(playoffs_mj_kb_lj,x='Minutes Played',y='PTS',color='Player',color_discrete_sequence=['red','gold','blue'],hover_name='Date')
#plot relationship between points scored and minutes played

g= sns.lmplot(x='Minutes Played',y='PTS',data=playoffs_mj_kb_lj,col='Player')

g = (g.set_axis_labels("Minutes Played", "PTS").set(ylim=(0, 70)))
#create points per minute column

playoffs_mj_kb_lj['PTS per Min']=(playoffs_mj_kb_lj['PTS']/playoffs_mj_kb_lj['Minutes Played'])
#pivot average points per minute for each player

playoffs_mj_kb_lj.pivot_table(values='PTS per Min',index='Player',aggfunc=['min','mean','max'])
px.scatter(playoffs_mj_kb_lj,x='Minutes Played',y='PTS per Min',color='Player',color_discrete_sequence=['red','gold','blue'],hover_name='Date')
#plot relationship between points per minute and minutes played

g= sns.lmplot(x='Minutes Played',y='PTS per Min',data=playoffs_mj_kb_lj,col='Player')

g = (g.set_axis_labels("Minutes Played", "PTS per Min").set(ylim=(0, 2.5)))
#Find out what the win/loss column looks like in the raw data table

winloss = playoffs_mj_kb_lj['WinLossPlusMinus']

winloss.tail()
#return only W or L and store to a new column

wl = []

for x in winloss:

    y = x[0]

    wl += y

playoffs_mj_kb_lj['WinLoss']= wl
#new table

playoffs_mj_kb_lj.tail()
#Points per game average in Wins vs. Losses

playoffs_mj_kb_lj.pivot_table(values='PTS', index=['Player'], columns=['WinLoss'], aggfunc=np.mean)
#carry over this change to each of the player tables

mj_playoffs = playoffs_mj_kb_lj[playoffs_mj_kb_lj['Player'] == 'Michael Jordan']

kobe_playoffs = playoffs_mj_kb_lj[playoffs_mj_kb_lj['Player'] == 'Kobe Bryant']

lebron_playoffs = playoffs_mj_kb_lj[playoffs_mj_kb_lj['Player'] == 'Lebron James']
#Player scoring distribution during Wins vs. Losses

g = sns.FacetGrid(data=playoffs_mj_kb_lj,col='WinLoss',row='Player')

g.map(sns.distplot,'PTS')
#Playoff scoring distribution Win vs. Loss

sns.stripplot(x='Player',y='PTS',data=playoffs_mj_kb_lj, jitter=True,hue='WinLoss',palette=['r','g'],dodge=True).set_title('Playoff Scoring Distribution in Wins vs. Losses')



plt.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0.)
#Points per game average in home vs. away

playoffs_mj_kb_lj.pivot_table(values='PTS', index=['Player'], columns=['HomeAway'], aggfunc=np.mean)
#Playoff scoring distribution home vs. away

sns.stripplot(x='Player',y='PTS',data=playoffs_mj_kb_lj, jitter=True,hue='HomeAway',palette=['k','c'],dodge=True).set_title('Playoff Scoring Distribution in Home vs. Away')



plt.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0.)
#pivot average points per minute for each player

playoffs_mj_kb_lj.pivot_table(values='FG%',index='Player',aggfunc=['min','mean','max'])
#plot field goal % vs points scored

px.scatter(playoffs_mj_kb_lj,x='FG%',y='PTS',color='Player',color_discrete_sequence=['red','gold','blue'],hover_name='Date')
#plot relationship between points scored and field goal %

g= sns.lmplot(x='FG%',y='PTS',data=playoffs_mj_kb_lj,col='Player')

g = (g.set_axis_labels("FG%", "PTS").set(ylim=(0, 70)))
playoffs_mj_kb_lj['True Shooting %'] = playoffs_mj_kb_lj['PTS']/(2*(playoffs_mj_kb_lj['FGA']+0.44*playoffs_mj_kb_lj['FTA']))
#pivot average points per minute for each player

playoffs_mj_kb_lj.pivot_table(values='True Shooting %',index='Player',aggfunc=['min','mean','max'])
#plot true shooting % vs points scored

px.scatter(playoffs_mj_kb_lj,x='True Shooting %',y='PTS',color='Player',color_discrete_sequence=['red','gold','blue'],hover_name='Date')
#plot relationship between points scored and field goal %

g= sns.lmplot(x='True Shooting %',y='PTS',data=playoffs_mj_kb_lj,col='Player')

g = (g.set_axis_labels("True Shooting %", "PTS").set(ylim=(0, 70)))