import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings



import plotly

import plotly.express as px

import plotly.graph_objs as go

import plotly.figure_factory as ff



warnings.simplefilter("ignore")

plt.style.use("classic")
# Read in the data

data = pd.read_csv("../input/womens-international-football-results/results.csv")

data.head()
data.isna().sum()
names = list(dict(data['home_team'].value_counts()).keys())[:15]

values = data['home_team'].value_counts().tolist()[:15]



fig = go.Bar(x = names,

            y = values,

            marker = dict(color = 'rgba(255, 0, 0, 0.5)',

                         line=dict(color='rgb(0,0,1)',width=1.5)),

            text = names)



layout = go.Layout()

fig = go.Figure(data = fig, layout = layout)

fig.update_layout(title_text='Top-15 Home Teams of Players')

fig.show()
names = list(dict(data['away_team'].value_counts()).keys())[:15]

values = data['away_team'].value_counts().tolist()[:15]



fig = go.Bar(x = names,

            y = values,

            marker = dict(color = 'rgba(0, 255, 0, 0.5)',

                         line=dict(color='rgb(0,0,50)',width=1.5)),

            text = names)



layout = go.Layout()

fig = go.Figure(data = fig, layout = layout)

fig.update_layout(title_text='Top-15 Away Teams of Players')

fig.show()
sns.distplot(data['home_score'], bins=15)

plt.xlabel("Home Score")

plt.ylabel("Density")

plt.title(f"Home Score Distribution [ \u03BC: {data['home_score'].mean():.2f} ]")

plt.show()
# Also the plotly figure

fig = ff.create_distplot(

    hist_data=[data['home_score'].tolist()],

    group_labels=['Home Score'],

    colors=['#ff00e1'],

    show_hist=False,

    show_rug=False,

)



fig.layout.update({'title':f"Home Score Distribution<br>[Average Score: {data['home_score'].mean():.2f} ]"})



fig.show()
sns.distplot(data['away_score'], bins=15, color='red')

plt.xlabel("Away Score")

plt.ylabel("Density")

plt.title(f"Away Score Distribution [ \u03BC: {data['away_score'].mean():.2f} ]")

plt.show()
# Also the plotly figure

fig = ff.create_distplot(

    hist_data=[data['away_score'].tolist()],

    group_labels=['Away Score'],

    colors=['#00BFFF'],

    show_hist=False,

    show_rug=False,

)



fig.layout.update({'title':f"Away Score Distribution<br>[Average Score: {data['away_score'].mean():.2f} ]"})



fig.show()
# Also the plotly figure

fig = ff.create_distplot(

    hist_data=[data['away_score'].tolist(), data['home_score'].tolist()],

    group_labels=['Away Score', 'Home Score'],

    colors=['#00008B', '#DC143C'],

    show_hist=False,

    show_rug=False,

)

total_avg = (data['away_score'].mean() + data['home_score'].mean()) / 2



fig.layout.update({'title':f"Complete Score Distribution<br>[Average Score: {total_avg:.2f} ]"})



fig.show()
data.head()
names = list(dict(data['tournament'].value_counts()).keys())

values = data['tournament'].value_counts().tolist()



fig = go.Bar(x = names,

             y = values,

             marker = dict(color = 'rgba(0, 0, 255, 0.5)',

                         line=dict(color='rgb(0,0,50)',width=1.5)),

             text = names)



layout = go.Layout()

fig = go.Figure(data = fig, layout = layout)

fig.update_layout(title_text='All Tournaments by number of players')

fig.show()