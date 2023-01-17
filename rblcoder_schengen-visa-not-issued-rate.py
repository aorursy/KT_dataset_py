import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import plotly.express as px

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df_2018 = pd.read_csv('/kaggle/input/schengen-visa-stats/2018-data-for-consulates.csv')

print(df_2018.info())

df_2018.dropna(subset=['Schengen State', 'Country where consulate is located'], inplace=True)

print(df_2018.sample(10))

print(df_2018['Schengen State'].value_counts())

print(df_2018[['Not issued rate for ATVs and uniform visas ']].describe())

df_2018['Not issued rate for ATVs and uniform visas '] = df_2018['Not issued rate for ATVs and uniform visas '].str.replace('%', '')

df_2018['Not issued rate for ATVs'] = df_2018['Not issued rate for ATVs'].str.replace('%', '')

df_2018['Not issued rate for uniform visas'] = df_2018['Not issued rate for uniform visas'].str.replace('%', '')

df_2018['Not issued rate for ATVs and uniform visas '] = df_2018['Not issued rate for ATVs and uniform visas '].astype(float) 

df_2018['Not issued rate for ATVs'] = df_2018['Not issued rate for ATVs'].astype(float) 

df_2018['Not issued rate for uniform visas'] = df_2018['Not issued rate for uniform visas'].astype(float) 

df_2018.loc[df_2018['Not issued rate for ATVs']>100,['Not issued rate for ATVs']] = 100
cols_int = ['Airport transit visas (ATVs) applied for ',

       ' ATVs issued (including multiple)', 'Multiple ATVs issued',

       'ATVs not issued ', 

       'Uniform visas applied for',

       'Total  uniform visas issued (including MEV) \n',

       'Multiple entry uniform visas (MEVs) issued',

       'Share of MEVs on total number of uniform visas issued',

       'Total LTVs issued', 'Uniform visas not issued',

       'Total ATVs and uniform visas applied for',

       'Total ATVs and uniform visas issued  (including multiple ATVs, MEVs and LTVs) ',

       'Total ATVs and uniform visas not issued']

df_2018[cols_int] = df_2018[cols_int].apply(pd.to_numeric, errors='coerce')

# df_2018[cols_int] = df_2018[cols_int].astype(int)

print(df_2018[['Not issued rate for ATVs and uniform visas ']].describe())

print(df_2018[['Not issued rate for ATVs']].describe())

print(df_2018[['Not issued rate for uniform visas']].describe())



# _=plt.scatter(df_2018[['Not issued rate for ATVs']], df_2018[['Not issued rate for uniform visas']])
import plotly.express as px

fig = px.scatter(df_2018, x="Not issued rate for ATVs", y="Not issued rate for uniform visas",

                hover_data=['Schengen State', 'Country where consulate is located', 'Consulate'])

fig.show()
!pip install pingouin

import pingouin as pg

expected, observed, stats = pg.chi2_independence(df_2018, x='Schengen State', y='Not issued rate for ATVs and uniform visas ')

print(stats)

expected, observed, stats = pg.chi2_independence(df_2018, x='Schengen State', y='Not issued rate for ATVs')

print(stats)

expected, observed, stats = pg.chi2_independence(df_2018, x='Schengen State', y='Not issued rate for uniform visas')

print(stats)
df_2017 = pd.read_csv('/kaggle/input/schengen-visa-stats/2017-data-for-consulates.csv')

df_2017.dropna(subset=['Schengen State', 'Country where consulate is located'], inplace=True)

df_2017['Not issued rate for ATVs and uniform visas '] = df_2017['Not issued rate for ATVs and uniform visas '].str.replace('%', '')

df_2017['Not issued rate for ATVs'] = df_2017['Not issued rate for ATVs'].str.replace('%', '')

df_2017['Not issued rate for uniform visas'] = df_2017['Not issued rate for uniform visas'].str.replace('%', '')

df_2017['Not issued rate for ATVs and uniform visas '] = df_2017['Not issued rate for ATVs and uniform visas '].astype(float) 

df_2017['Not issued rate for ATVs'] = df_2017['Not issued rate for ATVs'].astype(float) 

df_2017['Not issued rate for uniform visas'] = df_2017['Not issued rate for uniform visas'].astype(float) 


fig = px.scatter(df_2017, x="Not issued rate for ATVs", y="Not issued rate for uniform visas",

                hover_data=['Schengen State', 'Country where consulate is located', 'Consulate'])

fig.show()
df_2017['year'] = '2017'

df_2018['year'] = '2018'

df_concat = pd.concat([df_2017, df_2018])

df_concat.sample(5)

df_concat.dropna(subset=['Not issued rate for ATVs and uniform visas '], inplace=True)

xmin = df_concat['Not issued rate for ATVs'].min()

xmax = df_concat['Not issued rate for ATVs'].max()

ymin = np.maximum(df_concat['Not issued rate for uniform visas'].min(), 0.1)

ymax = df_concat['Not issued rate for uniform visas'].max()

df_concat['Not issued rate for uniform visas'].describe()
df_concat['Schengen State'].value_counts()[:5]

states_filter = df_concat['Schengen State'].value_counts()[:5].index
# https://plot.ly/python/plotly-express/

import plotly.express as px



fig = px.scatter(df_concat, 

                 x="Not issued rate for ATVs", y="Not issued rate for uniform visas", 

                 animation_frame="year",range_x=[xmin, xmax], range_y=[ymin, ymax],

                 size="Not issued rate for ATVs and uniform visas ", color="Schengen State",

           hover_name="Country where consulate is located", log_x=True, log_y=True, size_max=60,

                width=1000, height=1000)

fig.show()
# https://plot.ly/python/plotly-express/

# import plotly.express as px



# fig = px.scatter(df_concat[df_concat['Schengen State'].isin(states_filter[:2])], 

#                  x="Not issued rate for ATVs", y="Not issued rate for uniform visas", 

#                  animation_frame="year",range_x=[xmin, xmax], range_y=[ymin, ymax],

#                  facet_col="Schengen State",

#                  size="Not issued rate for ATVs and uniform visas ", color="Schengen State",

#            hover_name="Country where consulate is located", log_x=True, log_y=True, size_max=60)

# fig.show()
fig = px.scatter(df_concat[df_concat['Schengen State'].isin(states_filter)], 

                 x="Not issued rate for ATVs", y="Not issued rate for uniform visas", 

                 animation_frame="year",range_x=[xmin, xmax], range_y=[ymin, ymax],

                 facet_col="Schengen State",

                 size="Not issued rate for ATVs and uniform visas ", color="Schengen State",

           hover_name="Country where consulate is located", log_x=True, log_y=True, size_max=60,

                width=1200, height=800)

fig.show()


# fig = px.scatter(df_concat, x="Not issued rate for ATVs", y="Not issued rate for uniform visas", 

#                  color = 'year',,



#                 hover_data=['Schengen State', 'Country where consulate is located', 'Consulate'])

# fig.show()
df_concat['Schengen State Year'] = df_concat['Schengen State'] + ' ' + df_concat['year']
import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot





init_notebook_mode(connected=True)





fig = go.Figure()



fig.add_trace(go.Scatter(

    x=df_concat['Not issued rate for ATVs'],

    y=df_concat['Schengen State Year'],

    name='Not issued rate for ATVs',

    marker=dict(

        color='rgba(50, 165, 196, 0.95)',

        line_color='rgba(156, 165, 196, 0.5)',

    )

))

fig.add_trace(go.Scatter(

    x=df_concat['Not issued rate for uniform visas'], y=df_concat['Schengen State Year'],

    name='Not issued rate for uniform visas',

    marker=dict(

        color='rgba(204, 204, 204, 0.95)',

        line_color='rgba(217, 217, 217, 0.5)'

    )

))



fig.update_traces(mode='markers', marker=dict(line_width=1, symbol='circle', size=5))



fig.update_layout(

    title="Not issued rate",

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

    width=1000,

    height=1200,

    paper_bgcolor='white',

    plot_bgcolor='white',

    hovermode='closest',

)

fig.show()
