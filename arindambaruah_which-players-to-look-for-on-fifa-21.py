import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px

import plotly.graph_objects as go

import plotly

plotly.offline.init_notebook_mode (connected = True)
pd.set_option('display.max_columns',None)

df=pd.read_csv('../input/fifa-20-complete-player-dataset/players_20.csv')

df.head()
df.isna().any().value_counts()
plt.figure(figsize=(10,8))

sns.heatmap(df.isnull(),cmap='viridis')
df.info()
df.fillna(0,inplace=True)
df['Count']=1

df_foot=df.groupby('preferred_foot')['Count'].sum().reset_index()



fig=px.pie(df_foot,values='Count',names='preferred_foot',hole=0.4)

fig.update_layout(title='Distribution of preferred foot of FIFA players',title_x=0.5,

                  annotations=[dict(text='Foot',font_size=20, showarrow=False,height=800,width=700)])

fig.update_traces(textfont_size=15,textinfo='percent+label')

fig.show()
plt.figure(figsize=(10,8))

sns.kdeplot(df['age'],shade=True)

plt.title('Age distribution of players',size=25)

plt.axvline(df['age'].mean(),color='red')

plt.xlabel('Mean age:{0:.0f}'.format(df['age'].mean()),size=20)
df_nation=df.groupby('nationality')['Count'].sum().reset_index().sort_values(by='Count',ascending=False).head(10)
sns.catplot('nationality','Count',data=df_nation,kind='bar',height=8,aspect=2)

plt.xticks(size=15,rotation=45)

plt.xlabel('Nationality',size=20)

plt.yticks(size=15)

plt.title('Number of professional footballers of each nation',size=25)
df_val=df.groupby('club')['value_eur'].sum().reset_index().sort_values(by='value_eur',ascending=False).head(10)
sns.catplot('club','value_eur',data=df_val,kind='bar',height=8,aspect=2,palette='coolwarm_r')

plt.xticks(size=15,rotation=45)

plt.xlabel('Club name',size=20)

plt.yticks(size=15)

plt.title('Squad value',size=25)

plt.ylabel('Squad value (X 100 Million Euros)',size=15)
df_pot=df.groupby('club')['potential'].sum().reset_index().sort_values(by='potential',ascending=False).head(10)
sns.catplot('club','potential',data=df_pot,kind='bar',height=8,aspect=2,palette='winter')

plt.xticks(size=15,rotation=45)

plt.xlabel('Club name',size=20)

plt.yticks(size=15)

plt.title('Potential of the squads',size=25)

plt.ylabel('Squad potential)',size=15)
plt.figure(figsize=(10,8))

sns.distplot(df['value_eur'])

plt.title('Player value distribution',size=25)

plt.xlabel('Player value in Euros (X 100 million)',size=15)
sns.set()

plt.figure(figsize=(10,8))

sns.kdeplot(df['wage_eur'],shade=True,color='green')

plt.title('Player wages distribution',size=25)

plt.xlabel('Player wage in Euros',size=15)
df_top10=df.sort_values(by='overall',ascending=False).head(10)
sns.catplot('short_name','overall',data=df_top10,kind='point',height=8,aspect=2)

plt.xlabel('\n Name of players',size=20)



plt.ylabel('Overall',size=15)

plt.title('Top 10 players worldwide',size=25)

plt.xticks(size=15)
df_young=df[df['age']<24]

df_young_top10=df_young.sort_values(by='overall',ascending=False).head(10)
sns.catplot('short_name','overall',data=df_young_top10,kind='point',height=8,aspect=2,color='seagreen')

plt.xlabel('\n Name of players',size=20)



plt.ylabel('Overall',size=15)

plt.title('Top 10 young players worldwide',size=25)

plt.xticks(size=15)
df_gk=df[df['team_position']=='GK']

df_gk_top10=df_gk.sort_values(by='overall',ascending=False).head(10)
sns.catplot('short_name','overall',data=df_gk_top10,kind='point',height=8,aspect=2,color='indianred')

plt.xlabel('\n Name of players',size=20)



plt.ylabel('Overall',size=15)

plt.title('Top 10 Goalkeepers worldwide',size=25)

plt.xticks(size=15)
df_fk_top10=df.sort_values(by='skill_fk_accuracy',ascending=False).head(10)

sns.catplot('short_name','skill_fk_accuracy',data=df_fk_top10,kind='point',height=8,aspect=2,color='green')

plt.xlabel('\n Name of players',size=20)



plt.ylabel('Overall',size=15)

plt.title('Top 10 FK specialists worldwide',size=25)

plt.xticks(size=15,rotation=45)
df_def_top10=df.sort_values(by='defending',ascending=False).head(10)

sns.catplot('short_name','defending',data=df_def_top10,kind='point',height=8,aspect=2,color='blue')

plt.xlabel('\n Name of players',size=20)



plt.ylabel('Overall',size=15)

plt.title('Top 10 defenders worldwide',size=25)

plt.xticks(size=15,rotation=45)
df_pac_top10=df.sort_values(by='pace',ascending=False).head(10)

sns.catplot('short_name','pace',data=df_pac_top10,kind='point',height=8,aspect=2,color='black')

plt.xlabel('\n Name of players',size=20)



plt.ylabel('Overall pace',size=15)

plt.title('Top 10 defenders worldwide',size=25)

plt.xticks(size=15,rotation=45)
df_mean_age=df.groupby('club')['age'].mean().reset_index().sort_values(by='age').round(0)
df_mean_age=df_mean_age.loc[[505,224,400,350,389,226,462,626,63,92],:].sort_values(by='age',ascending=False)
sns.catplot('club','age',data=df_mean_age,kind='bar',height=8,aspect=2,palette='summer')

plt.xticks(size=15,rotation=45)

plt.xlabel('Club name',size=20)

plt.ylabel('Average age',size=20)

plt.title('Average squad age of clubs',size=25)
df_top_pos=df.iloc[df.groupby(df['team_position'])['overall'].idxmax()]

df_top_player_pos=df_top_pos.loc[:,['short_name','overall','team_position']]

df_top_player_pos.drop(327,axis=0,inplace=True) #Bad value

df_top_player_pos.sort_values(by='overall',ascending=False).reset_index(drop=True)
df_workrate=df.groupby('work_rate')['Count'].sum().reset_index().sort_values(by='Count')



fig=px.pie(df_workrate,values='Count',names='work_rate',hole=0.4)

fig.update_layout(title='Distribution of work rate of FIFA players',title_x=0.5,

                  annotations=[dict(text='Work rate',font_size=20, showarrow=False,height=800,width=700)])

fig.update_traces(textfont_size=15,textinfo='percent')

fig.show()