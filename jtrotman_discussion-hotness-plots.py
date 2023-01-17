import matplotlib.pyplot as plt

import plotly.express as px

import gc, os, sys, re, time

import pandas as pd

import numpy as np



dates = [

    'postDate',

    'lastCommentPostDate',

]
filepath = '../input/discussion-hotness-2020-06-28/discussion-hotness-2020-06-28.csv'
df = pd.read_csv(filepath, parse_dates=dates, index_col=0)

df.shape
df.authorType.value_counts()
latest = df.postDate.max() + pd.Timedelta(1, 'm')
df['Age'] = ((latest - df.postDate) / pd.Timedelta(1, 'd')).astype('float32')
# for plot.ly

df['postDateText'] = df.postDate.dt.strftime("%c")

df['lastCommentPostDateText'] = df.lastCommentPostDate.dt.strftime("%c")
df['VotesPerDay'] = df['votes'] / df['Age']

df['VotesPerComment'] = df['votes'] / df['commentCount']

df['CommentsPerDay'] = df['commentCount'] / df['Age']
df.describe(include='all').T
plt.rc("figure", figsize=(14, 9))

plt.rc("font", size=(14))



title = "Discussion Posts Hotness"

params = dict(title=title, alpha=0.4)
df.plot.scatter('Page', 'Age', **params)
df.query("Page<=80").plot.scatter('Page', 'Age', **params)
hover_data = [

    'Page', 'commentCount', 'id', 'lastCommenterType', 'lastCommenterName',

    'lastCommentPostDateText', 'medal', 'parentName', 'postDateText', 'votes',

    'author_displayName', 'author_tier', 'author_userUrl', 'Age',

    'VotesPerDay', 'VotesPerComment', 'CommentsPerDay'

]



color_discrete_map = {

    'gold': 'gold',

    'silver': 'silver',

    'bronze': 'chocolate',

    '': 'lightgreen'

}
fig = px.scatter(df.query("Page<=75").reset_index().fillna(""),

                 title=title,

                 x='Index',

                 y='Age',

                 hover_name='title',

                 hover_data=hover_data,

                 color='medal',

                 color_discrete_map=color_discrete_map)

fig.update_traces(marker=dict(size=8))

fig.update_layout(height=750, showlegend=False)
df.plot.scatter('id', 'Page', **params)

plt.axes().invert_yaxis()
df.plot.scatter('id', 'author_id', **params)
df.plot.scatter('Page', 'author_id', **params)
df.plot.scatter('id', 'commentCount', **params)

plt.yscale('symlog')
pdates = df.postDate.dt.strftime("%Y %m %d %a")
df.groupby(pdates).Page.agg(['count','min','mean','std','max']).round(1).tail(30).sort_index(ascending=False).style.background_gradient()
show = ["Page", "Age", "title", "medal", "votes", "commentCount"]
df[df.title.str.contains("Jigsaw")][show]
subset = df.query("parentName=='Jigsaw Multilingual Toxic Comment Classification'").copy().reset_index()
subset['Index'] = subset.Index.rank()

subset['id'] = subset.id.rank()
fig = px.scatter(subset.fillna(""),

                 title=title,

                 x='id',

                 y='Index',

                 hover_name='title',

                 hover_data=hover_data,

                 color='medal',

                 color_discrete_map=color_discrete_map)

fig.update_traces(marker=dict(size=8))

fig.update_layout(height=750, showlegend=False)