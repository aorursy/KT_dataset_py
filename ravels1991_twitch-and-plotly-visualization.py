import pandas as pd

import numpy as np

import random

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.io as pio

import plotly.express as px

pio.templates.default = "plotly_dark"
df = pd.read_csv('../input/twitchdata/twitchdata-update.csv')

df.head()
#changing minutes to hours

df['Watch time(Minutes)'] = (df['Watch time(Minutes)'] / 60).astype(int)

df['Stream time(minutes)'] = (df['Stream time(minutes)'] / 60).astype(int)

df.rename(columns={'Watch time(Minutes)': 'Watch time(hours)', 'Stream time(minutes)': 'Stream time(hours)'}, inplace=True)

df.head()
languages = df['Language'].value_counts(ascending=True, normalize=True) * 100
fig1 = go.Figure()



fig1.add_trace(go.Bar(

     x= languages.index.to_list(),

     y= languages.to_list(),

     marker={'color': 'skyblue',

            'line':{'color':'black', 'width': 1}}))



fig1.update_layout(title='Percentage of Streamers by language',

                           xaxis=dict(title='Streamers language'),

                           yaxis=dict(title='Percentage'))

fig1.show()
colors = px.colors.sequential.Rainbow # create a list with all the colors of Rainbow palette
def plot_px(y, title, df):

    random.shuffle(colors) # shuffle the colors for giving diferente colors to our plot every time we call our funciton

    col = df[y].sort_values(ascending=False)[:25] # Select top 25 stremers from a select column

    fig = px.bar(df, x=df.iloc[col.index]['Channel'].to_list(),

                     y=col.value_counts().index.to_list(),

                     color=df.iloc[col.index]['Language'].to_list(),

                     color_discrete_sequence = colors,

                     title=title,

                     labels={'color':'Languages'})

    fig.update_layout(xaxis=dict(title='Streamers'),

                      yaxis=dict(title=y))

    return fig.show()
plot_px('Followers gained', 'Top 25 Streamers by Followers gained', df)
plot_px('Followers', 'Top 25 Streamers by Followers', df)
plot_px('Watch time(hours)', 'Top 25 Most Watched Streamers', df)
plot_px('Average viewers', 'Top 25 Streamers by Average viewers', df)
plot_px('Peak viewers', 'Top 25 Streamers by Peak viewers', df)
mature = pd.Series(np.where(df['Mature'] == 0, 'No', 'Yes')).value_counts()

partnered = pd.Series(np.where(df['Partnered'] == 0, 'No', 'Yes')).value_counts()
f1 = go.Bar(

     x= mature.index.to_list(),

     y= mature.to_list(),

     name='Twitch Mature',

     marker={'color': '#51ff0d'})

f2 = go.Bar(

     x= partnered.index.to_list(),

     y= partnered.to_list(),

     name='Twitch Partnered',

     marker={'color': 'skyblue'})

data = [f1, f2]

layout = go.Layout(barmode='group')

fig = go.Figure(data=data, layout=layout)

fig.update_layout(title='Streamers Matura and Partnered', yaxis=dict(title='Count'), autosize=False)

fig.show()
cols = ['Peak viewers', 'Followers', 'Followers gained', 'Views gained']

a = df.groupby('Language')[cols].mean()

top_6 = df['Language'].value_counts()[:6].index.to_list()
fig1 = go.Bar(

     x= list(a.loc[top_6[0]].index),

     y= list(a.loc[top_6[0]].values),

     name='English',

     marker={'color': '#ff0dff'})

fig2 = go.Bar(

     x= list(a.loc[top_6[1]].index),

     y= list(a.loc[top_6[1]].values),

     name='Korean',

     marker={'color': '#f5ff0d'})

fig3 = go.Bar(

     x= list(a.loc[top_6[2]].index),

     y= list(a.loc[top_6[2]].values),

     name='Russian',

     marker={'color': '#1ac1dd'})

fig4 = go.Bar(

     x= list(a.loc[top_6[3]].index),

     y= list(a.loc[top_6[3]].values),

     name='Spanish',

     marker={'color': '#3eff0d'})

fig5 = go.Bar(

     x= list(a.loc[top_6[4]].index),

     y= list(a.loc[top_6[4]].values),

     name='French',

     marker={'color': '#d0748b'})

fig6 = go.Bar(

     x= list(a.loc[top_6[5]].index),

     y= list(a.loc[top_6[5]].values),

     name='Portuguese',

     marker={'color': '#0d60ff'})
data = [fig1, fig2, fig3, fig4, fig5, fig6]

random.shuffle(data)

layout = go.Layout(barmode='group')

fig = go.Figure(data=data, layout=layout)

fig.update_layout(title='Average values by languages')

fig.show()