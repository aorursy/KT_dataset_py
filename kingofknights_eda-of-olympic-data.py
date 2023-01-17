import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.offline as off
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly

%matplotlib inline
sns.set_style("white")
off.init_notebook_mode(connected=True)

region = pd.read_csv('../input/noc_regions.csv')
athlete = pd.read_csv('../input/athlete_events.csv', index_col='ID')

region.drop('notes', axis=1, inplace=True)

athlete.Medal.fillna('Not Winner', inplace=True)
athlete.Height.fillna(athlete.Height.mean(), inplace=True)
athlete.Weight.fillna(athlete.Weight.mean(), inplace=True)
athlete.Age.fillna(athlete.Age.mean(), inplace=True)
athlete.head()
plt.figure(figsize=(19,10))
plt.subplot(211)
ax = sns.boxplot(x='Year', y='Age', hue='Sex', data=athlete)
ax.set_title('Variation in Age in Sex through the olympics history.')
plt.subplot(212)
ax = sns.pointplot(x='Year', y='Age',hue='Sex', data=athlete)
plt.figure(figsize=(19,10))
plt.subplot(211)
ax = sns.boxplot(x='Year', y='Weight', hue='Sex', data=athlete)
ax.set_title('Variation in Weight in Sex through the olympics history.')
plt.subplot(212)
ax = sns.pointplot(x='Year', y='Weight',hue='Sex', data=athlete)
plt.figure(figsize=(19,10))
plt.subplot(211)
ax = sns.boxplot(x='Year', y='Height', hue='Sex', data=athlete)
ax.set_title('Variation in Height in Sex through the olympics history.')
plt.subplot(212)
ax = sns.pointplot(x='Year', y='Height',hue='Sex', data=athlete)
plt.figure(figsize=(19,8))
ax = sns.barplot(x='Year', y='NOC', hue='Sex', data=athlete.groupby(['Year', 'Sex'], as_index=False).count())
ax.set_title('Number of athlete participated through the olympics history.')
ax.set_ylabel('Count')
data = athlete.groupby(['Year','Season', 'Name'], as_index=False).count()
plt.figure(figsize=(19,4))
ax = sns.pointplot(x='Year', y='Name',hue='Season', data=data.groupby(['Year','Season'], as_index=False).count())
ax.set_title('Number of athlete participated through the olympics history.')
ax.set_ylabel('athlete')
data = athlete.groupby(['Year','Season', 'NOC'], as_index=False).count()
plt.figure(figsize=(19,4))
ax = sns.pointplot(x='Year', y='NOC',hue='Season', data=data.groupby(['Year','Season'], as_index=False).count())
ax.set_title('Number of nations participated through the olympics history.')
ax.set_ylabel('Nations')
data = athlete.groupby(['Year','Season', 'Sport'], as_index=False).count()
plt.figure(figsize=(19,4))
ax = sns.pointplot(x='Year', y='Sport',hue='Season', data=data.groupby(['Year','Season'], as_index=False).count())
ax.set_title('Number of Sport held through the olympics history.')
ax.set_ylabel('Sports')
data = athlete.groupby(['Year','Season', 'Event'], as_index=False).count()
plt.figure(figsize=(19,4))
ax = sns.pointplot(x='Year', y='Event',hue='Season', data=data.groupby(['Year','Season'], as_index=False).count())
ax.set_title('Number of Events held through the olympics history.')
ax.set_ylabel('Events')

plt.figure(figsize=(19,4))
ax = sns.pointplot(x='Year', y='Name',hue='Sex', data=athlete[athlete.Season == 'Summer'].groupby(['Year', 'Sex'], as_index=False).count())
ax.set_title('Number of participants olympics history.')
ax.set_ylabel('Count')
winners = athlete[athlete.Medal != 'Not Winner'].groupby('NOC', as_index=False).count()
winners = winners.sort_values(by='Medal', ascending=False)
winners = winners[winners.Medal != 0]

plt.figure(figsize=(19,24))
ax = sns.barplot(x='Medal', y='NOC', data=winners)
ax.set_title('Number of Events held through the olympics history.')
ax.set_ylabel('Country')

def plotmedalbyyearforcountriesbySeaborn(year):
    data = athlete[athlete.Year == year]
    data = data[data.Medal != 'Not Winner']
    data = data.groupby(['NOC', 'Medal'], as_index=False).count()
    data = data.sort_values(by='Name', ascending=False)
    plt.figure(figsize=(19,20))
    ax = sns.barplot(y='NOC', x='Name', hue='Medal', data=data)
    ax.set_title('Number of Medal won by each countries in year ' + str(year))
    ax.set_ylabel('Country')
def plotmedalbyyearforcountriesbtPlotly(year):
    data = athlete[athlete.Year == year]
    data = data[data.Medal != 'Not Winner']
    data = data.groupby(['NOC', 'Medal'], as_index=False).count()
    data = data.sort_values(by='Name', ascending=False)
    gold = go.Bar(y=data[data.Medal == 'Gold'].Name, x=data[data.Medal == 'Gold'].NOC, name='Gold')
    Silver = go.Bar(y=data[data.Medal == 'Silver'].Name, x=data[data.Medal == 'Silver'].NOC, name='Silver')
    Bronze = go.Bar(y=data[data.Medal == 'Bronze'].Name, x=data[data.Medal == 'Bronze'].NOC, name='Bronze')
    layout = go.Layout(barmode='stack', title='Number of medal won by countries in olympics of year ' + str(year), xaxis=dict(title='Countries'), yaxis=dict(title='Count'))
    figure = go.Figure(data=[gold, Silver, Bronze], layout=layout)
    off.iplot(figure)
years = athlete.Year.unique()
years = np.sort(years)

for year in years:
    plotmedalbyyearforcountriesbtPlotly(year)