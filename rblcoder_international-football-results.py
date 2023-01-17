import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt
df_football = pd.read_csv('/kaggle/input/international-football-results-from-1872-to-2017/results.csv')
df_football['date'] = pd.to_datetime(df_football['date'])

df_football['year'] = df_football['date'].dt.year

print(df_football['year'].nunique())

_=df_football['year'].plot(kind='hist')
df_football.info()
df_football.describe()
df_football.sample(5)
_=plt.scatter(df_football['home_score'], df_football['away_score'])
df_football['home away'] = df_football['home_team'] + ' vs ' + df_football['away_team']
df_football['home away'].nunique()
df_football['home away'].value_counts()
home_away = ['Argentina vs Uruguay', 'Uruguay vs Argentina', 'Austria vs Hungary', 'Hungary vs Austria', 'Kenya vs Uganda '] 

df_football_filtered_home_away = df_football[df_football['home away'].isin(home_away)]
import plotly.express as px

#iris = px.data.iris()

fig = px.scatter(df_football_filtered_home_away, x="home_score", y="away_score", color="year", facet_row="home away", width=700, height=2000)

fig.show()
import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot





init_notebook_mode(connected=True)





fig = go.Figure()



fig.add_trace(go.Scatter(

    x=df_football_filtered_home_away['year'],

    y=df_football_filtered_home_away['home away'],

    hovertext=df_football_filtered['year'],

    name='home_score',

    marker=dict(

        color='rgba(50, 165, 196, 0.7)',

        line_color='rgba(156, 165, 196, 0.5)',

    )

))

fig.add_trace(go.Scatter(

    x=df_football_filtered_home_away['away_score'], y=df_football_filtered_home_away['home away'],

    hovertext=df_football_filtered['year'],

    name='away_score',

    marker=dict(

        color='rgba(204, 204, 204, 0.7)',

        line_color='rgba(217, 217, 217, 0.5)'

    )

))



fig.update_traces(mode='markers', marker=dict(line_width=1, symbol='circle', size=5))



fig.update_layout(

    title="Scores",

    xaxis=dict(

        showgrid=False,

        showline=True,

        linecolor='rgb(102, 102, 102)',

        tickfont_color='rgb(102, 102, 102)',

        showticklabels=True,

        dtick=10,

        ticks='outside',

        tickcolor='rgb(102, 102, 102)',

    ),

    margin=dict(l=140, r=40, b=50, t=80),

    legend=dict(

        font_size=10,

        yanchor='middle',

        xanchor='right',

    ),

    width=700,

    height=1200,

    paper_bgcolor='white',

    plot_bgcolor='white',

    hovermode='closest',

    xaxis_type="log",

    xaxis_rangeslider_visible=True

)



fig.show()
df_football.columns
df_football_filtered = df_football[df_football['year']>=2005]
df_football_filtered.info()
import plotly.express as px

iris = px.data.iris()

fig = px.scatter(df_football_filtered, x="home_score", y="away_score", color="year",

                  hover_data=['home_team', 'away_team'])

fig.show()
import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot





init_notebook_mode(connected=True)





fig = go.Figure()



fig.add_trace(go.Scatter(

    x=df_football_filtered['home_score'],

    y=df_football_filtered['home away'],

    hovertext=df_football_filtered['year'],

    name='home_score',

    marker=dict(

        color='rgba(50, 165, 196, 0.7)',

        line_color='rgba(156, 165, 196, 0.5)',

    )

))

fig.add_trace(go.Scatter(

    x=df_football_filtered['away_score'], y=df_football_filtered['home away'],

    hovertext=df_football_filtered['year'],

    name='away_score',

    marker=dict(

        color='rgba(204, 204, 204, 0.7)',

        line_color='rgba(217, 217, 217, 0.5)'

    )

))



fig.update_traces(mode='markers', marker=dict(line_width=1, symbol='circle', size=5))



fig.update_layout(

    title="Scores for years 2005 and above",

    xaxis=dict(

        showgrid=False,

        showline=True,

        linecolor='rgb(102, 102, 102)',

        tickfont_color='rgb(102, 102, 102)',

        showticklabels=True,

        dtick=10,

        ticks='outside',

        tickcolor='rgb(102, 102, 102)',

    ),

    margin=dict(l=140, r=40, b=50, t=80),

    legend=dict(

        font_size=10,

        yanchor='middle',

        xanchor='right',

    ),

    width=700,

    height=1200,

    paper_bgcolor='white',

    plot_bgcolor='white',

    hovermode='closest',

    xaxis_type="log",

    xaxis_rangeslider_visible=True

)



fig.show()