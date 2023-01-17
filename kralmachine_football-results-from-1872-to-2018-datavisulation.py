import numpy as np 

import pandas as pd 

import seaborn as sns

import plotly.plotly as py

from plotly.offline import init_notebook_mode,iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

from wordcloud import WordCloud

import matplotlib.pyplot as plt

import os

print(os.listdir("../input"))

data=pd.read_csv('../input/results.csv')
data.head()
data.tail()
data.sample(5)
data.sample(frac=0.1)
data.sample()
data.dtypes
data.corr()
sns.heatmap(data.corr(),annot=True)

plt.show()
data.columns
for i,col in enumerate(data.columns):

    print((i+1),'-',col)
data.info()

data.shape
data[['home_score']].describe()
data.describe()
data.columns
away_team=pd.DataFrame(data.groupby('away_team')['home_score'].count().index)

away_score=pd.DataFrame(data.groupby('away_team')['home_score'].count().values,columns=['Score'])

away_score_team=pd.concat([away_team,away_score],axis=1)
plt.figure(figsize=(10,10))

away_score_team=away_score_team.sort_values(by='Score',ascending=False)

sns.barplot(x=away_score_team.away_team[:50],y=away_score_team.Score[:50])

plt.xticks(rotation=90)

plt.show()
data.isnull().values.any()
data.isnull().sum()
data.head(1)
data['tournament'].value_counts()
sns.barplot(x=data['tournament'].value_counts().index[:20],y=data['tournament'].value_counts().values[:20])

plt.xticks(rotation=90)

plt.show()
data['tournament'].unique()
sayisi=len(data['tournament'].unique())

sayisi
tournament=data['tournament'].value_counts()

names=tournament.index

values=tournament.values
plt.figure(figsize=(10,10))

sns.barplot(x=names[:10],y=values[:10])

plt.xticks(rotation=90)

plt.ylabel('Values')

plt.xlabel('Tournament')

plt.title('Tournament vs Values How Play in the World')

plt.show()
data.shape
data.info()
sns.pairplot(data.iloc[:,3:5])

plt.show()
sns.violinplot(x=data['home_score'][:50],y=data['away_score'][:50])

plt.show()
sns.violinplot(x="home_score", y="away_score", hue="neutral",data=data)

sns.despine(left=True)

plt.show()
sns.scatterplot(y="home_score", x="away_score",

                hue="neutral",data=data)

plt.show()
sns.scatterplot(x="home_score", y="away_score",

                hue="neutral",data=data)

plt.show()
sns.heatmap(data.corr(),annot=True)

plt.show()
matches = data.astype({'date':'datetime64[ns]'})

tournament = matches['tournament'].value_counts()

tournament = tournament[:15]



plt.figure(figsize = (15,10))

ax = sns.barplot(y=tournament.index, x=tournament.values, orient='h')

ax.set_ylabel('Tournament', size=16)

ax.set_xlabel('Number of tournament', size=16)

ax.set_title("TOP 15 TYPE OF MATCH TOURNAMENTS", fontsize=18)

plt.show()
data.head()
home_name_index=data.home_team.value_counts()

home_name_index=home_name_index.head(10)



plt.figure(figsize=(10,10))

ax=sns.barplot(x=home_name_index.index,y=home_name_index.values,palette=sns.cubehelix_palette(len(home_name_index.index)))

plt.xlabel('Home Team Name')

plt.ylabel('Frequency')

plt.xticks(rotation=90)

plt.title('Most common 100 Home Team Name')

plt.show()
data.head()
tournament_liste=data.tournament.unique()

home_score_liste=[]

away_score_liste=[]

for tour in tournament_liste:

    home_score_liste.append(sum(data[data['tournament']==tour].home_score))

    away_score_liste.append(sum(data[data['tournament']==tour].away_score))



trace1=go.Bar(x=tournament_liste,y=home_score_liste,name='Home Score',marker = dict(color = 'rgba(255, 50, 70, 0.2)',line=dict(color='rgb(0,0,0)',width=1.5)),text='')

trace2=go.Bar(x=tournament_liste,y=away_score_liste,name='Away Score',marker = dict(color = 'rgba(0, 0, 0, 0.3)',line=dict(color='rgb(0,0,0)',width=1.5)),text ='') 

data2=[trace1,trace2]

layout=go.Layout(barmode='group')

fig=go.Figure(data=data2,layout=layout)

iplot(fig)
trace1={'x':tournament_liste,'y':home_score_liste,'name':'Home Score','type':'bar'};

trace2={'x':tournament_liste,'y':away_score_liste,'name':'Away Score','type':'bar'};

data3=[trace1,trace2]

layout = {

  'xaxis': {'title': 'Tournaments'},

  'barmode': 'relative',

  'title': 'Home Score and Away Score'

};

fig = go.Figure(data = data3, layout = layout)

iplot(fig)

data.head()
veri=data[data['tournament']=='FIFA World Cup'].tail(56)

veri
veri=data[data['tournament']=='FIFA World Cup'].tail(56)



len(veri['home_team'].unique())

allteam=veri['home_team'].unique()

allteam
away_scores_allteam=[]

home_scores_allteam=[]

for team in allteam:

    toplam=sum(veri[veri['home_team']==team].away_score)

    away_scores_allteam.append(toplam)

    home_scores_allteam.append(sum(veri[veri['home_team']==team].home_score))

    toplam=0



away_scores_allteam

home_scores_allteam

allteam



all_team=pd.DataFrame([allteam,home_scores_allteam,away_scores_allteam])

                       

all_team
f,ax1=plt.subplots(figsize=(20,10))

sns.pointplot(x=allteam,y=home_scores_allteam,data=veri,color='lime',alpha=0.8)

sns.pointplot(x=allteam,y=away_scores_allteam,data=veri,color='red',alpha=0.8)

plt.text(30,0.5,'FIFA CUP 2018 HOME SCORE',color='red',fontsize = 10,style = 'italic')

plt.text(30,0.5,'FIFA CUP 2018 AWAY SCORE',color='lime',fontsize = 10,style = 'italic')

plt.xlabel('TEAM',fontsize = 15,color='blue')

plt.xticks(rotation=90)

plt.ylabel('SCORES',fontsize = 15,color='blue')

plt.title('AWAY SCORES VS HOME SCORES',fontsize = 20,color='blue')

plt.grid()
sns.barplot(x=allteam,y=home_scores_allteam)

plt.title('FIFA CUP 2018 Home Scores',color='b',fontsize=15)

plt.xticks(rotation=90)

plt.show()
sns.barplot(x=allteam,y=away_scores_allteam)

plt.title('FIFA CUP 2018 Away Scores',color='b',fontsize=15)

plt.xticks(rotation=90)

plt.show()
data.head()
data['neutral'].unique()

data['neutral'].value_counts()

sns.countplot(data['neutral'])

plt.xlabel('Neutral Value')

plt.title('Neutral Value vs Frequency',color = 'blue',fontsize = 15)

plt.show()
data.head()
sns.countplot(veri.city)

plt.xlabel('City')

plt.xticks(rotation=90)

plt.title('City List',color='blue',fontsize=15)

plt.show()
veri.head()
sns.countplot(veri.city,hue=veri.neutral)

plt.xticks(rotation=90)

plt.title('City for Hue Neutral')

plt.show()
tournaments=data.tournament.value_counts()

alltournaments=tournaments[:20]



fig = {

  "data": [

    {

      "values": alltournaments.values,

      "labels": alltournaments.index,

      "domain": {"x": [0, .5]},

      "name": "Tournaments Count",

      "hoverinfo":"label+percent+name",

      "hole": .3,

      "type": "pie"

    },],

  "layout": {

        "title":"Tournaments Rate",

        "annotations": [

            { "font": { "size": 20},

              "showarrow": False,

              "text": "Number of Tour Counts",

                "x": 0.1,

                "y": 1.1

            },

        ]

    }

}

iplot(fig)
data.head()

import plotly.figure_factory as ff

dataframe = data[data.tournament == 'FIFA World Cup']

data2015 = dataframe.loc[:,["tournament","home_score", "away_score","id"]]

data2015

fig = ff.create_scatterplotmatrix(data2015, diag='box', index='id',colormap='Portland',colormap_type='cat',height=700, width=700)

iplot(fig)

data.head()
x=data[data['home_team']=='England'].groupby('away_team')['home_score'].count()

away_team=pd.DataFrame(x.index)

score=pd.DataFrame(x.values,columns=['scores'])
away_team_score=pd.concat([away_team,score],axis=1)

away_team_score.sort_values(by='scores',ascending=False,inplace=True)
sns.barplot(x='away_team',y='scores',data=away_team_score[:20])

plt.xticks(rotation=90)

plt.show()
data.head()
max(data.date)
filter_2018=data[data['date'].str.contains("2018")]

filter_2018=filter_2018.sort_values(by="away_score",ascending=False)

filter_2018
min(data.date)
filter_1872=data[data['date'].str.contains("1872")]

filter_1872=filter_1872.sort_values(by="away_score",ascending=False)

filter_1872
data.corr()
data.corr()
sns.heatmap(data.corr(),annot=True,fmt='.1f')

plt.show()
ax = sns.heatmap(data.corr(), cmap="YlGnBu",annot=True)

plt.show()
corr = np.corrcoef(np.random.randn(10, 200))

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True

with sns.axes_style("white"):

    ax = sns.heatmap(corr, mask=mask, vmax=.3, square=True)

plt.show()