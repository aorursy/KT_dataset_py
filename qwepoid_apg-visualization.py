import pandas as pd

import plotly.offline as py

import plotly.graph_objs as go

import matplotlib.pyplot as plt

import seaborn as sns

py.init_notebook_mode(connected=True)

import plotly.tools as tls
train = pd.read_csv('../input/preprocessedData.csv')
country2019wc = ['Afghanistan','Australia','Bangladesh','England','India','New Zealand','Pakistan','South Africa','Sri Lanka','West Indies']

trainwc2019 = train[train.Team.isin(country2019wc)]

trainwc2019 = trainwc2019[trainwc2019.Opposition.isin(country2019wc)]

trainwc2019.shape[0]
totalMatches=pd.concat([train['Team'],train['Opposition']])

totalMatches=totalMatches.value_counts().reset_index()

totalMatches.columns=['Team','Total Matches']

totalMatches.head(11)

# totalMatches.columns=['Team','Total Matches']
totalMatches['wins']=train['Winner'].value_counts().reset_index()['Winner']

totalMatches.set_index('Team',inplace=True)

totalMatches.head(11)
trace1 = go.Bar(

    x=totalMatches.index,

    y=totalMatches['Total Matches'].head(11),

    name='Total Matches'

)



trace2 = go.Bar(

    x=totalMatches.index,

    y=totalMatches['wins'].head(11),

    name='Wins'

)

data = [trace1, trace2]

layout = go.Layout(

    barmode='stack'

)

fig = go.Figure(data = data, layout =layout)

py.iplot(fig, filename='stacked-bar')
match_succes_rate = (totalMatches['wins']/totalMatches['Total Matches'])*100

print(match_succes_rate.head(11))


def annot_plot(ax,w,h):                                    # function to add data to plot

    ax.spines['top'].set_visible(False)

    ax.spines['right'].set_visible(False)

    for p in ax.patches:

        ax.annotate('{0:.1f}'.format(p.get_height()), (p.get_x()+w, p.get_height()+h))



data = match_succes_rate.head(11).sort_values(ascending = False)

plt.figure(figsize=(7,3))

ax = sns.barplot(x = data.index, y = data, palette="Set2");

plt.ylabel('Succes rate of each team')

plt.xticks(rotation=80)

annot_plot(ax,0.08,1)
#performance in england

trainEngland = train[train['Match Country'].isin(['England'])]

trainEngland.head()

#TODO: add a check to include only those countries wh have played 10 or more matches
trainEngland = trainEngland.reset_index()

trainEngland = trainEngland.drop(['index'],axis=1)

trainEngland.head()
totalMatchesInEngland=pd.concat([trainEngland['Team'],trainEngland['Opposition']])

totalMatchesInEngland=totalMatchesInEngland.value_counts().reset_index()

totalMatchesInEngland.columns=['Team','Total Matches']
totalMatchesInEngland['wins']=trainEngland['Winner'].value_counts().reset_index()['Winner']

totalMatchesInEngland.set_index('Team',inplace=True)

totalMatchesInEngland.head(10)
trace1 = go.Bar(

    x=totalMatchesInEngland.index,

    y=totalMatchesInEngland['Total Matches'].head(8),

    name='Total Matches'

)



trace2 = go.Bar(

    x=totalMatchesInEngland.index,

    y=totalMatchesInEngland['wins'].head(8),

    name='Wins'

)

data = [trace1, trace2]

layout = go.Layout(

    barmode='stack'

)

fig = go.Figure(data = data, layout =layout)

py.iplot(fig, filename='stacked-bar')
match_succes_rate = (totalMatchesInEngland['wins'].head(8)/totalMatchesInEngland['Total Matches'].head(8))*100

print(match_succes_rate.head(8))
def annot_plot(ax,w,h):                                    # function to add data to plot

    ax.spines['top'].set_visible(False)

    ax.spines['right'].set_visible(False)

    for p in ax.patches:

        ax.annotate('{0:.1f}'.format(p.get_height()), (p.get_x()+w, p.get_height()+h))



data = match_succes_rate.sort_values(ascending = False)

plt.figure(figsize=(7,3))

ax = sns.barplot(x = data.index, y = data, palette="Set2");

plt.ylabel('Succes rate of each team')

plt.xticks(rotation=80)

annot_plot(ax,0.08,1)
yearsSince2016 = ['2016','2017','2018','2019']

trainSince2016 = train[train.year.isin(yearsSince2016)]

trainSince2016.shape[0]
totalMatches=pd.concat([trainSince2016['Team'],trainSince2016['Opposition']])

totalMatches=totalMatches.value_counts().reset_index()

totalMatches.columns=['Team','Total Matches']
totalMatches['wins']=trainSince2016['Winner'].value_counts().reset_index()['Winner']

totalMatches.set_index('Team',inplace=True)

totalMatches.head(11)
trace1 = go.Bar(

    x=totalMatches.index,

    y=totalMatches['Total Matches'].head(11),

    name='Total Matches'

)



trace2 = go.Bar(

    x=totalMatches.index,

    y=totalMatches['wins'].head(11),

    name='Wins'

)

data = [trace1, trace2]

layout = go.Layout(

    barmode='stack'

)

fig = go.Figure(data = data, layout =layout)

py.iplot(fig, filename='stacked-bar')
match_succes_rate = (totalMatches['wins']/totalMatches['Total Matches'])*100

print(match_succes_rate.head(11))



def annot_plot(ax,w,h):                                    # function to add data to plot

    ax.spines['top'].set_visible(False)

    ax.spines['right'].set_visible(False)

    for p in ax.patches:

        ax.annotate('{0:.1f}'.format(p.get_height()), (p.get_x()+w, p.get_height()+h))



data = match_succes_rate.head(11).sort_values(ascending = False)

plt.figure(figsize=(7,3))

ax = sns.barplot(x = data.index, y = data, palette="Set2");

plt.ylabel('Succes rate of each team')

plt.xticks(rotation=80)

annot_plot(ax,0.08,1)
trainEngland = trainEngland[trainEngland.year.isin(yearsSince2016)]

totalMatchesInEngland=pd.concat([trainEngland['Team'],trainEngland['Opposition']])

totalMatchesInEngland=totalMatchesInEngland.value_counts().reset_index()

totalMatchesInEngland.columns=['Team','Total Matches']
totalMatchesInEngland['wins']=trainEngland['Winner'].value_counts().reset_index()['Winner']

totalMatchesInEngland.set_index('Team',inplace=True)

totalMatchesInEngland.head(8)
trace1 = go.Bar(

    x=totalMatchesInEngland.index,

    y=totalMatchesInEngland['Total Matches'].head(8),

    name='Total Matches'

)



trace2 = go.Bar(

    x=totalMatchesInEngland.index,

    y=totalMatchesInEngland['wins'].head(8),

    name='Wins'

)

data = [trace1, trace2]

layout = go.Layout(

    barmode='stack'

)

fig = go.Figure(data = data, layout =layout)

py.iplot(fig, filename='stacked-bar')
match_succes_rate = (totalMatchesInEngland['wins']/totalMatchesInEngland['Total Matches'])*100

print(match_succes_rate.head(8))



def annot_plot(ax,w,h):                                    # function to add data to plot

    ax.spines['top'].set_visible(False)

    ax.spines['right'].set_visible(False)

    for p in ax.patches:

        ax.annotate('{0:.1f}'.format(p.get_height()), (p.get_x()+w, p.get_height()+h))



data = match_succes_rate.head(8).sort_values(ascending = False)

plt.figure(figsize=(7,3))

ax = sns.barplot(x = data.index, y = data, palette="Set2");

plt.ylabel('Succes rate of each team')

plt.xticks(rotation=80)

annot_plot(ax,0.08,1)
yearsSince2018 = ['2018','2019']

trainSince2018 = train[train.year.isin(yearsSince2018)]

trainSince2018.shape[0]
totalMatches=pd.concat([trainSince2018['Team'],trainSince2018['Opposition']])

totalMatches=totalMatches.value_counts().reset_index()

totalMatches.columns=['Team','Total Matches']
totalMatches['wins']=trainSince2018['Winner'].value_counts().reset_index()['Winner']

totalMatches.set_index('Team',inplace=True)

totalMatches.head(11)
trace1 = go.Bar(

    x=totalMatches.index,

    y=totalMatches['Total Matches'].head(11),

    name='Total Matches'

)



trace2 = go.Bar(

    x=totalMatches.index,

    y=totalMatches['wins'].head(11),

    name='Wins'

)

data = [trace1, trace2]

layout = go.Layout(

    barmode='stack'

)

fig = go.Figure(data = data, layout =layout)

py.iplot(fig, filename='stacked-bar')
match_succes_rate = (totalMatches['wins']/totalMatches['Total Matches'])*100

print(match_succes_rate.head(11))



def annot_plot(ax,w,h):                                    # function to add data to plot

    ax.spines['top'].set_visible(False)

    ax.spines['right'].set_visible(False)

    for p in ax.patches:

        ax.annotate('{0:.1f}'.format(p.get_height()), (p.get_x()+w, p.get_height()+h))



data = match_succes_rate.head(11).sort_values(ascending = False)

plt.figure(figsize=(7,3))

ax = sns.barplot(x = data.index, y = data, palette="Set2");

plt.ylabel('Succes rate of each team')

plt.xticks(rotation=80)

annot_plot(ax,0.08,1)