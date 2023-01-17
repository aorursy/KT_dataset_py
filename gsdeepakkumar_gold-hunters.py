import numpy as np #Linear algebra
import pandas as pd ## data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import warnings 
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
warnings.filterwarnings('ignore')
games=pd.read_csv('../input/athlete_events.csv')
noc=pd.read_csv('../input/noc_regions.csv')
## Glimpse of the data:
games.head()
noc.head()
print(games.isnull().any())
team=games.groupby(['Year'])['Team'].nunique().reset_index()
team.rename({'Team':'Team_Count'},inplace=True,axis=1)
team.head()
trace = go.Scatter(
                x=team['Year'],
                y=team['Team_Count'],
                name = "Team Participation in Olympics",
                line = dict(color = '#17BECF'),
                opacity = 0.8,
                mode="lines+markers"
                )

data = [trace]

layout = dict(
    title = "Team Participation in Olympics(Both Summer and Winter )",
)

fig = dict(data=data, layout=layout)
py.iplot(fig, filename = "Team Participation in Olympics")
plt.figure(figsize=(8,8))
plt.subplot(311)
ax=sns.distplot(games['Age'].dropna(),color='blue',kde=True)
ax.set_xlabel('Age')
ax.set_ylabel('Density')
ax.set_title('Age Distribution of Sportspersons',fontsize=16,fontweight=200)
plt.subplot(312)
ax1=sns.distplot(games['Height'].dropna(),color='Red',kde=True)
ax1.set_xlabel('Height')
ax1.set_ylabel('Density')
ax1.set_title('Height Distribution of Sportspersons',fontsize=16,fontweight=200)
plt.subplot(313)
ax2=sns.distplot(games['Weight'].dropna(),color='green',kde=True)
ax2.set_xlabel('Weight')
ax2.set_ylabel('Density')
ax2.set_title('Weight Distribution of Sportspersons',fontsize=16,fontweight=200)

plt.subplots_adjust(wspace = 0.2, hspace = 0.6,top = 0.9)
games.drop(['ID','Year'],axis=1).describe()
Year=games.groupby('City').apply(lambda x:x['Year'].unique()).to_frame().reset_index()
Year.columns=['City','Years']
Year['Count']=[len(c) for c in Year['Years']]
Year.sort_values('Count',ascending=False)
sports=games.groupby('Year').Sport.nunique().to_frame().reset_index()
sports.columns=['Year','Count of Sport']

trace1 = go.Scatter(
                x=sports['Year'],
                y=sports['Count of Sport'],
                name = "Sports in Olympics",
                line = dict(color = '#17BECD'),
                opacity = 0.8,
                mode="lines+markers"
                )

data1 = [trace1]

layout1 = dict(
    title = "Representation of Sports (Count) in Olympics(Summer and Winter)",
)

fig = dict(data=data1, layout=layout1)
py.iplot(fig, filename = "Sport Count")
sports=games.groupby(['Year','Season']).Sport.nunique().to_frame().reset_index()
plt.figure(figsize=(10,10))
ax=sns.pointplot(x=sports['Year'],y=sports['Sport'],hue=sports['Season'],dodge=True)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set_xlabel('Year',fontsize=10)
ax.set_ylabel('Count',fontsize=10)
ax.set_title('Sports in Olympics',fontsize=16)
### Sport in Summer Olympics:
summer_olympic=games[games['Season']=='Summer']
Sport_Count=summer_olympic.groupby('Sport').apply(lambda x:x['Year'].unique()).to_frame().reset_index()
Sport_Count.columns=['Sport','Years']
Sport_Count['Count']=[len(c) for c in Sport_Count['Years']]

Sport_Count['Years']=pd.Series(Sport_Count['Years'])
Sport_Count['Years']=Sport_Count['Years'].apply(lambda x:sorted(x))  ### Sort Year in ascending order inside the Year column.
Sport_Count.sort_values('Count',ascending=False,inplace=True)
Sport_Count
### Sport in Winter Olympics:
Winter_olympic=games[games['Season']=='Winter']
Winter_Count=Winter_olympic.groupby('Sport').apply(lambda x:x['Year'].unique()).to_frame().reset_index()
Winter_Count.columns=['Sport','Years']
Winter_Count['Count']=[len(c) for c in Winter_Count['Years']]

#Winter_Count['Years']=pd.Series(Winter_Count['Years'])
Winter_Count['Years']=Winter_Count['Years'].apply(lambda x:sorted(x))
Winter_Count.sort_values('Count',ascending=False)
game_noc=pd.merge(games,noc,how='left',on='NOC')
game_noc.drop_duplicates(inplace=True,keep=False)
medal=game_noc.groupby(['region','Medal'])['Medal'].count()
medal=medal.unstack(level=-1,fill_value=0).reset_index()
medal.head()
medal['Total']=medal['Bronze']+medal['Gold']+medal['Silver']
total_games=game_noc.groupby('region')['Sport'].nunique().to_frame().reset_index()
total_games.rename({'Sport':'TotalGames'},inplace=True,axis=1)
#total_games.head()
medal=pd.merge(medal,total_games,how='left',on='region')
medal.sort_values('Total',ascending=False,inplace=True)

medal=medal[['region','TotalGames','Gold','Silver','Bronze','Total']]  ### Reordering the columns
medal.head(10)
plt.figure(figsize=(10,10))
ax=sns.barplot(medal['region'].head(10),medal['Total'].head(10),palette=sns.color_palette('Set1',10))
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set_xlabel('Region',fontsize=10)
ax.set_ylabel('Total Medals',fontsize=10)
ax.set_title('Total Medal Count of Top 10 Countries in Olympics')
USA_Gold=game_noc[(game_noc['region']=='USA') & (game_noc['Medal']=='Gold')]
champ=USA_Gold.groupby('Sport').size().to_frame().reset_index()
champ.columns=['Sport','Count']
champ.sort_values(by='Count',ascending=False,inplace=True)
plt.figure(figsize=(10,10))
ax=sns.barplot(champ['Sport'].head(10),champ['Count'].head(10),palette=sns.color_palette('viridis_r',10))
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set_xlabel('Sport',fontsize=10)
ax.set_ylabel('Count',fontsize=10)
ax.set_title('Games where USA has won Gold maximum times')
champ=USA_Gold.groupby(['Name','Sport']).size().to_frame().reset_index()

champ.columns=['Name','Sport','Golds']
#champ.head()
champ.sort_values(by='Golds',ascending=False,inplace=True)
champ.head(10)
Phelps=game_noc[(game_noc['Name']=='Michael Fred Phelps, II' ) & (game_noc['Medal']=='Gold')]
print("Swimming Event where Phelps has won Gold\n",Phelps['Event'].unique)

game_noc[game_noc['Name']=='Michael Fred Phelps, II'].Year.nunique()
gold=game_noc.loc[game_noc['Medal']=='Gold'].groupby(['Name','Sport','region']).size().to_frame().reset_index()
gold.columns=['Name','Sport','region','count']
gold.sort_values('count',ascending=False,inplace=True)
gold.head(10)
### Considering only sports that were played from inception so that we have a good comparion with lot of data !!!!.
sport_box=game_noc[game_noc['Sport'].isin(Sport_Count.Sport[:10])]
plt.figure(figsize=(10,8))
plt.subplot(311)
ax=sns.boxplot(x='Sport',y='Age',data=sport_box,palette=sns.color_palette(palette='viridis_r'))
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set_xlabel('Sport',fontsize=10)
ax.set_ylabel('Age',fontsize=10)
ax.set_title('Age distribution across sports',fontsize=16)
plt.subplot(312)
ax=sns.boxplot(x='Sport',y='Height',data=sport_box,palette=sns.color_palette(palette='viridis_r'))
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set_xlabel('Sport',fontsize=10)
ax.set_ylabel('Height',fontsize=10)
ax.set_title('Height distribution across sports',fontsize=16)
plt.subplot(313)
ax=sns.boxplot(x='Sport',y='Weight',data=sport_box,palette=sns.color_palette(palette='viridis_r'))
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set_xlabel('Sport',fontsize=10)
ax.set_ylabel('Weight',fontsize=10)
ax.set_title('Weight distribution across sports',fontsize=16)
plt.subplots_adjust(wspace = 1, hspace = 1,top = 1.3)