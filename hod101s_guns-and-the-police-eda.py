!pip install calmap
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go

import plotly.figure_factory as ff

from plotly.subplots import make_subplots

import calmap

plt.rcParams['figure.figsize'] = 8, 5

plt.style.use("fivethirtyeight")

pd.options.plotting.backend = "plotly"

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
data = pd.read_csv('../input/data-police-shootings/fatal-police-shootings-data.csv')

data.head()
data.date = pd.to_datetime(data.date)
data.shape
data.info()
def nulls(df):

    for col in df.columns:

        nll = data[col].isnull().sum()

        print(f"{col} \t\t {round(nll/len(df)*100,2)}% Null")

nulls(data)
fig,ax = calmap.calendarplot(data.groupby(['date']).id.count(), monthticks=1, daylabels='MTWTFSS',cmap='YlGn',

                    linewidth=0, fig_kws=dict(figsize=(20,20)))

fig.show()
freq = data[['date','id']]

freq['year'] = freq.date.dt.year

freq['month'] = freq.date.dt.month

freq.head()
fig = go.Figure(data=[go.Bar(x=freq.groupby(['year']).agg('count')['id'].index, y=freq.groupby(['year']).agg('count')['id'].values,)])

fig.update_layout(title_text='Shootouts by year')

fig.show()
fig = go.Figure(data=[go.Bar(x=freq.groupby('month').agg('count')['id'].index, y=freq.groupby('month').agg('count')['id'].values,)])

fig.update_layout(title_text='Shootouts by Month')

fig.show()
fig = go.Figure(data=[go.Pie(labels=data.manner_of_death.value_counts().index, values=data.manner_of_death.value_counts().values,textinfo='label+percent')])

fig.update_layout(title='How were they killed?')

fig.show()
data.plot.hist(x="age")
fig = go.Figure([go.Bar(x=data.gender.value_counts().index, y=data.gender.value_counts().values)])

fig.update_layout(title="Number of Shootouts by Gender")

fig.show()
5176/238
print(f"{len(data.loc[data.armed=='unarmed'])/len(data)*100}% Cases were Unarmed")
armed=list(data['armed'].dropna().unique())

fig, (ax2) = plt.subplots(1,1,figsize=[17, 10])

wordcloud2 = WordCloud(width=1000,height=400).generate(" ".join(armed))

ax2.imshow(wordcloud2,interpolation='bilinear')

ax2.axis('off')

ax2.set_title('Most Used Arms',fontsize=20)
fig = go.Figure(data=[go.Pie(labels=data.flee.value_counts().index, values=data.flee.value_counts().values,textinfo='label+percent')])

fig.update_layout(title='Did they Flee?')

fig.show()
print(f'{len(data.loc[(data.flee=="Not fleeing") & (data.armed=="unarmed")])} Cases were Unarmed and Did not Flee. Yet were Killed.')
fig = go.Figure([go.Bar(x=data.threat_level.value_counts().index, y=data.threat_level.value_counts().values)])

fig.update_layout(title="Threat Level Assessment")

fig.show()
fig = go.Figure(data=[go.Pie(labels=data.race.value_counts().index, values=data.race.value_counts().values,textinfo='label+percent')])

fig.update_layout(title='Did they Flee?')

fig.show()
fig = go.Figure([go.Choropleth(

    locations=data.groupby(['state']).agg('count')['id'].index,

    z=data.groupby(['state']).agg('count')['id'].values.astype(float),

    locationmode='USA-states',

    colorscale='Reds',

    autocolorscale=False,

    text=data['state'], # hover text

    marker_line_color='white', # line markers between states

    showscale = True,

#     text=data.groupby(['state','race']).agg('count')['id'],

)])

fig.update_layout(geo_scope='usa',title='Shootouts across the States')

fig.show()
data.groupby(['state','race'])['id'].count().unstack('state').plot.bar()
fig = go.Figure(go.Bar(

    x= data.groupby('city').agg('count')['id'].sort_values(ascending=False)[:20].index, 

    y= data.groupby('city').agg('count')['id'].sort_values(ascending=False)[:20].values,  

    text=data.groupby('city').agg('count')['id'].sort_values(ascending=False)[:20].index,

    textposition='outside',

    marker_color=data.groupby('city').agg('count')['id'].sort_values(ascending=False)[:20].values

))

fig.update_layout(title='Shootout by City Stats')

fig.show()