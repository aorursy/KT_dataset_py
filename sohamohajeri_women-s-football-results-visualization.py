import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import cufflinks as cf

cf.go_offline()

import plotly.express as px

import plotly.io as pio

pio.renderers.default='notebook'
df=pd.read_csv('../input/womens-international-football-results/results.csv')
df.head()
df.shape
df.info()
df.isnull().sum()
type(df['date'].iloc[0])
df['date']=pd.to_datetime(df['date'])
type(df['date'].iloc[0])
df['year']=df['date'].apply(lambda x: x.year)
df.head()
home=df.groupby('home_team').sum()['home_score'].sort_values(ascending=False).head(15)
away=df.groupby('away_team').sum()['away_score'].sort_values(ascending=False).head(15)
plt.figure(figsize=(18,6))

plt.subplot(1,2,1)

plt.bar(x=list(home.index), height=list(home.values),color='deepskyblue')

plt.xticks(rotation=90)

plt.xlabel('Country',  fontsize=13)

plt.ylabel('Home Score',  fontsize=13)

plt.title('The Top 15 Teams With The Highest Home Scores', fontsize=16)

plt.subplot(1,2,2)

plt.bar(x=list(away.index), height=list(away.values) ,color='deeppink')

plt.xticks(rotation=90)

plt.xlabel('Country', fontsize=13)

plt.ylabel('Away Score',  fontsize=13)

plt.title('The Top 15 Teams With The Highest Away Scores', fontsize=16)

plt.show()
df_home_away=pd.DataFrame({'total_home_scores':home ,'total_away_scores':away})
df_home_away.isnull().sum()
df_home_away.fillna(0, inplace=True)
df_home_away['total_scores']= df_home_away['total_home_scores'] + df_home_away['total_away_scores']
df_home_away_sorted=df_home_away.sort_values(by='total_scores', ascending=False).head(5)
df_home_away_sorted
plt.figure(figsize=(8,6))

plt.bar(x=list(df_home_away_sorted.index), height=df_home_away_sorted['total_scores'],color='slateblue')

plt.xticks(rotation=90)

plt.xlabel('Country', fontsize=13)

plt.ylabel('Total Score', fontsize=13)

plt.title('The Best 5 Teams Of All Time', fontsize=16)

plt.show()
country=[]

total_score=[]



for x in df['year'].unique():

    home=df[df['year']==x].groupby('home_team').sum()['home_score']

    away=df[df['year']==x].groupby('away_team').sum()['away_score']

    new=pd.DataFrame({'home_team':home ,'away_team':away})

    new.fillna(0, inplace=True)

    new['total']= new['home_team'] + new['away_team']

    total_sorted=new['total'].sort_values(ascending=False).head(1)

    country=country+list(total_sorted.index)

    total_score=total_score+list(total_sorted)
df_year= pd.DataFrame({'country':country,'total_score':total_score,'year':df['year'].unique()})
df_year.head()
fig=px.scatter_3d(df_year,x='country', y='year',z='total_score')

fig.update_layout(

    title={

        'text': 'Countries That Gained The Highest Total Score In Each Year',

        'y':0.92,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'})

fig.show()
df_year[df_year['total_score']==df_year['total_score'].max()]
number=[]

for x in df['year'].unique():

    number.append(df[df['year']==x]['tournament'].nunique())
plt.figure(figsize=(12,6))

plt.bar(x=df['year'].unique(), height=number,color='orchid')

plt.xlabel('Year', fontsize=13)

plt.ylabel('Number of Tournaments', fontsize=13)

plt.xticks(ticks=df['year'].unique(),rotation=90)

plt.title('Number of Tournaments Per Year', fontsize=16)

plt.show()
tournament=[]

for x in df['tournament'].unique():

    tournament.append(df[df['tournament']==x]['year'].nunique())
plt.figure(figsize=(12,6))

plt.bar(x=df['tournament'].unique(),height=tournament, color='r')

plt.xticks(rotation=90)

plt.title('Number of Tournaments In Total', fontsize=16)

plt.xlabel('Tournaments', fontsize=13)

plt.ylabel('Number', fontsize=13)

plt.show()
plt.figure(figsize=(17,6))





plt.subplot(1,2,1)

sns.distplot(df['home_score'],bins=15,color='lime',hist_kws=dict(edgecolor='black'))

plt.xlabel('Home Score', fontsize=13)

plt.ylabel('Density', fontsize=13)

plt.title('Distribuition Of Home Scores', fontsize=16)



plt.subplot(1,2,2)

sns.distplot(df['away_score'],bins=15,color='magenta',hist_kws=dict(edgecolor='black'))

plt.xlabel('Away Score', fontsize=13)

plt.title('Distribuition Of Away Scores', fontsize=16)



plt.show()
plt.figure(figsize=(10,6))

sns.kdeplot(df['home_score'], color='r')

sns.kdeplot(df['away_score'], color='b')

plt.xlabel('Score', fontsize=13)

plt.ylabel('Density', fontsize=13)

plt.legend(['Home Score', 'Away Score'], fontsize=12)

plt.title('Comparison Of Home And Away Scores', fontsize=16)

plt.show()
number=[]

for x in df['country'].unique():

    number.append(df[df['country']==x]['year'].nunique())
df_country=pd.DataFrame({'Host Country':df['country'].unique(),'Number of Hosting':number}).sort_values(by='Number of Hosting',ascending=False).head(15)
plt.figure(figsize=(12,6))

plt.bar(x=df_country['Host Country'], height=df_country['Number of Hosting'], color='turquoise')

plt.xlabel('Country', fontsize=13)

plt.ylabel('Number of Hosting', fontsize=13)

plt.xticks(rotation=90)

plt.title('Top 15 Countries With The Highest Number Of Hosting', fontsize=16)

plt.show()
only_host=[]

for x in df['country'].unique():

        df_con=df[df['country']==x]

        only_host.append(df_con[(df_con['home_team']!=x) & (df_con['away_team']!=x)]['date'].count())
df_host=pd.DataFrame({'Host Country':df['country'].unique(), 'Number of Only Hosting':only_host}).sort_values(by='Number of Only Hosting',ascending=False).head(15)
plt.figure(figsize=(12,6))

plt.bar(x=df_host['Host Country'], height=df_host['Number of Only Hosting'],color='purple')

plt.xlabel('Country', fontsize=13)

plt.ylabel('Number of Hosting', fontsize=13)

plt.xticks(rotation=90)

plt.title('Top 15 Countries That only Hosted Matches', fontsize=16)

plt.show()
win=[]

lose=[]

equal=[]

for x in df['country'].unique():

    df_con=df[df['country']==x]

    df_home=df_con[df_con['home_team']==x]

    win.append(df_con[df_con['home_score']>df_con['away_score']]['date'].count())

    lose.append(df_con[df_con['home_score']<df_con['away_score']]['date'].count())

    equal.append(df_con[df_con['home_score']==df_con['away_score']]['date'].count())
df_chance=pd.DataFrame({'Host Country':df['country'].unique(), 'Number of Winning':win, 'Number of Loosing':lose, 'Number of Equal':equal})
df_merge=pd.merge(df_chance,df_country, on='Host Country').sort_values(by='Number of Hosting', ascending=False)
df_merge.head()
plt.figure(figsize=(12,6))

plt.bar(x=df_merge['Host Country'], height=df_merge['Number of Winning'], color='royalblue')

plt.bar(x=df_merge['Host Country'], height=df_merge['Number of Loosing'], color='greenyellow')

plt.legend(['Winning','Loosing'])

plt.xlabel('Host Country', fontsize=13)

plt.ylabel('Number', fontsize=13)

plt.xticks(rotation=90)

plt.legend(['Winning', 'Loosing'], fontsize=12)

plt.title('Number Of Winning And Loosing Of The 15 Top Host Countries', fontsize=16)

plt.show()
fig=px.scatter_3d(data_frame=df,x='home_score', y='away_score',z='away_team',color='home_team',hover_name='year')

fig.update_layout(

    title={

        'text': 'Home Teams And Away Teams Versus Home Scores And Away Scores In Each Year',

        'y':0.92,

        'x':0.45,

        'xanchor': 'center',

        'yanchor': 'top'})

fig.show()