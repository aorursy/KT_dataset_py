import os



import pandas as pd

import numpy as np





from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

import plotly.express as px

import plotly.graph_objs as go

import matplotlib.pyplot as plt



init_notebook_mode(connected=True)
df = pd.read_csv('../input/owid-covid-data.csv')
df['death_rate'] = (df['new_deaths_smoothed_per_million'] / df['new_cases_smoothed_per_million']).replace(np.inf,np.nan)

df['population_coverage'] = df['total_tests'] / df['population']

df['datetime'] = pd.to_datetime(df['date'])
df.describe().T
problem_idx = df[(df['new_cases']<0)|(df['new_deaths']<0)|(df['new_cases_smoothed']<0)|(df['new_deaths_smoothed']<0)|(df['new_cases_per_million']<0)|(df['new_deaths_per_million']<0)

                 |(df['new_deaths_smoothed_per_million']<0)|(df['new_tests']<0)|(df['new_tests_per_thousand']<0)|(df['location'].isin(['World']))].index
df = df[~df.index.isin(problem_idx)]

df = df[~df['date'].isin(['2020-09-18', '2020-09-19'])].reset_index(drop=True)
trace1 = go.Scatter(

    x=df.groupby(['date'])['date'].apply(lambda x: np.unique(x)[0]),

    y=df.groupby(['date'])['new_tests_smoothed'].sum().astype(int),

        xaxis='x2',

    yaxis='y2',

    name = "new tests smoothed"

)

trace2 = go.Scatter(

    x=df.groupby(['date'])['date'].apply(lambda x: np.unique(x)[0]),

    y=df.groupby(['date'])['new_deaths_smoothed'].sum().astype(int),

    name = "new deaths smoothed"

)

trace3 = go.Scatter(

    x=df.groupby(['date'])['date'].apply(lambda x: np.unique(x)[0]),

    y=(df.groupby(['date'])['positive_rate'].mean() * 100).round(3),

    xaxis='x3',

    yaxis='y3',

    name = "test positive rate %"

)

trace4 = go.Scatter(

    x=df.groupby(['date'])['date'].apply(lambda x: np.unique(x)[0]),

    y=df.groupby(['date'])['new_cases_smoothed'].sum().astype(int),

    xaxis='x4',

    yaxis='y4',

    name = "new cases smoothed"

)



data = [trace1, trace2, trace3, trace4]

layout = go.Layout(

    xaxis=dict(

        domain=[0, 0.45]

    ),

    yaxis=dict(

        domain=[0, 0.45]

    ),

    xaxis2=dict(

        domain=[0.55, 1]

    ),

    xaxis3=dict(

        domain=[0, 0.45],

        anchor='y3'

    ),

    xaxis4=dict(

        domain=[0.55, 1],

        anchor='y4'

    ),

    yaxis2=dict(

        domain=[0, 0.45],

        anchor='x2'

    ),

    yaxis3=dict(

        domain=[0.55, 1]

    ),

    yaxis4=dict(

        domain=[0.55, 1],

        anchor='x4'

    ),

    title = 'New tests, deaths, cases and test positive rate'

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
trace1 = go.Scatter(

    x=df.groupby(['date'])['date'].apply(lambda x: np.unique(x)[0]),

    y=df.groupby(['date'])['new_deaths_smoothed_per_million'].mean(),

        xaxis='x2',

    yaxis='y2',

    name = "mean new deaths smoothed per million"

)

trace2 = go.Scatter(

    x=df.groupby(['date'])['date'].apply(lambda x: np.unique(x)[0]),

    y=df.groupby(['date'])['new_tests_smoothed_per_thousand'].mean(),

    name = "mean new tests smoothed per thousand"

)

trace3 = go.Scatter(

    x=df.groupby(['date'])['date'].apply(lambda x: np.unique(x)[0]),

    y=(df.groupby(['date'])['death_rate'].mean().replace([np.inf],np.nan).interpolate(method='linear', limit_direction='forward', axis=0) * 100).round(3),

    xaxis='x3',

    yaxis='y3',

    name = "interpolated death rate %"

)

trace4 = go.Scatter(

    x=df.groupby(['date'])['date'].apply(lambda x: np.unique(x)[0]),

    y=((df.groupby(['date'])['new_cases_per_million'].apply(lambda x: np.mean(x/1e+6))) * 100).round(6),

    xaxis='x4',

    yaxis='y4',

    name = "mean covid population d2d coverage %"

)



data = [trace1, trace2, trace3, trace4]

layout = go.Layout(

    xaxis=dict(

        domain=[0, 0.45]

    ),

    yaxis=dict(

        domain=[0, 0.45]

    ),

    xaxis2=dict(

        domain=[0.55, 1]

    ),

    xaxis3=dict(

        domain=[0, 0.45],

        anchor='y3'

    ),

    xaxis4=dict(

        domain=[0.55, 1],

        anchor='y4'

    ),

    yaxis2=dict(

        domain=[0, 0.45],

        anchor='x2'

    ),

    yaxis3=dict(

        domain=[0.55, 1]

    ),

    yaxis4=dict(

        domain=[0.55, 1],

        anchor='x4'

    ),

    title = 'Mean new deaths per 1M, new tests per 1K, death rate and covid mean coverage'

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
trace1 = go.Scatter(

                    x = df[(df['continent']=='Asia')].groupby(['date','continent'])['date'].apply(lambda x: np.unique(x)[0]),

                    y = df[(df['continent']=='Asia')].groupby(['date','continent'])['new_deaths_smoothed'].sum(),

                    mode = "lines",

                    name = "Asia",

                    marker = dict(color = 'green'),

)



trace2 = go.Scatter(

                    x = df[(df['continent']=='Europe')].groupby(['date','continent'])['date'].apply(lambda x: np.unique(x)[0]),

                    y = df[(df['continent']=='Europe')].groupby(['date','continent'])['new_deaths_smoothed'].sum(),

                    mode = "lines",

                    name = "Europe",

                    marker = dict(color = 'red'),

)



trace3 = go.Scatter(

                    x = df[(df['continent']=='Africa')].groupby(['date','continent'])['date'].apply(lambda x: np.unique(x)[0]),

                    y = df[(df['continent']=='Africa')].groupby(['date','continent'])['new_deaths_smoothed'].sum(),

                    mode = "lines",

                    name = "Africa",

                    marker = dict(color = 'blue'),

                    #text= df.university_name

)



trace4 = go.Scatter(

                    x = df[(df['continent']=='North America')].groupby(['date','continent'])['date'].apply(lambda x: np.unique(x)[0]),

                    y = df[(df['continent']=='North America')].groupby(['date','continent'])['new_deaths_smoothed'].sum(),

                    mode = "lines",

                    name = "North America",

                    marker = dict(color = 'black'),

)



trace5 = go.Scatter(

                    x = df[(df['continent']=='South America')].groupby(['date','continent'])['date'].apply(lambda x: np.unique(x)[0]),

                    y = df[(df['continent']=='South America')].groupby(['date','continent'])['new_deaths_smoothed'].sum(),

                    mode = "lines",

                    name = "South America",

                    marker = dict(color = 'brown'),

)



data = [trace1,trace2,trace3,trace4,trace5]

layout = dict(title = 'New Deaths Smoothed',

              xaxis= dict(title= '# deaths day by day',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
trace1 = go.Scatter(

                    x = df[(df['continent']=='Asia')].groupby(['date','continent'])['date'].apply(lambda x: np.unique(x)[0]),

                    y = df[(df['continent']=='Asia')].groupby(['date','continent'])['new_tests_smoothed'].sum(),

                    mode = "lines",

                    name = "Asia",

                    marker = dict(color = 'green'),

)



trace2 = go.Scatter(

                    x = df[(df['continent']=='Europe')].groupby(['date','continent'])['date'].apply(lambda x: np.unique(x)[0]),

                    y = df[(df['continent']=='Europe')].groupby(['date','continent'])['new_tests_smoothed'].sum(),

                    mode = "lines",

                    name = "Europe",

                    marker = dict(color = 'red'),

)



trace3 = go.Scatter(

                    x = df[(df['continent']=='Africa')].groupby(['date','continent'])['date'].apply(lambda x: np.unique(x)[0]),

                    y = df[(df['continent']=='Africa')].groupby(['date','continent'])['new_tests_smoothed'].sum(),

                    mode = "lines",

                    name = "Africa",

                    marker = dict(color = 'blue'),

)



trace4 = go.Scatter(

                    x = df[(df['continent']=='North America')].groupby(['date','continent'])['date'].apply(lambda x: np.unique(x)[0]),

                    y = df[(df['continent']=='North America')].groupby(['date','continent'])['new_tests_smoothed'].sum(),

                    mode = "lines",

                    name = "North America",

                    marker = dict(color = 'black'),

)



trace5 = go.Scatter(

                    x = df[(df['continent']=='South America')].groupby(['date','continent'])['date'].apply(lambda x: np.unique(x)[0]),

                    y = df[(df['continent']=='South America')].groupby(['date','continent'])['new_tests_smoothed'].sum(),

                    mode = "lines",

                    name = "South America",

                    marker = dict(color = 'brown'),

)



data = [trace1,trace2,trace3,trace4,trace5]

layout = dict(title = 'New tests smoothed',

              xaxis= dict(title= '# tests day by day',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
trace1 = go.Scatter(

                    x = df[(df['continent']=='Asia')&(df['datetime']>='2020-03-01')].groupby(['date','continent'])['date'].apply(lambda x: np.unique(x)[0]),

                    y = df[(df['continent']=='Asia')&(df['datetime']>='2020-03-01')].groupby(['date','continent'])['death_rate'].mean()*100,

                    mode = "lines",

                    name = "Asia",

                    marker = dict(color = 'green'),

)



trace2 = go.Scatter(

                    x = df[(df['continent']=='Europe')&(df['datetime']>='2020-03-01')].groupby(['date','continent'])['date'].apply(lambda x: np.unique(x)[0]),

                    y = df[(df['continent']=='Europe')&(df['datetime']>='2020-03-01')].groupby(['date','continent'])['death_rate'].mean()*100,

                    mode = "lines",

                    name = "Europe",

                    marker = dict(color = 'red'),

)



trace3 = go.Scatter(

                    x = df[(df['continent']=='Africa')&(df['datetime']>='2020-03-01')].groupby(['date','continent'])['date'].apply(lambda x: np.unique(x)[0]),

                    y = df[(df['continent']=='Africa')&(df['datetime']>='2020-03-01')].groupby(['date','continent'])['death_rate'].mean()*100,

                    mode = "lines",

                    name = "Africa",

                    marker = dict(color = 'blue'),

)



trace4 = go.Scatter(

                    x = df[(df['continent']=='North America')&(df['datetime']>='2020-03-01')].groupby(['date','continent'])['date'].apply(lambda x: np.unique(x)[0]),

                    y = df[(df['continent']=='North America')&(df['datetime']>='2020-03-01')].groupby(['date','continent'])['death_rate'].mean()*100,

                    mode = "lines",

                    name = "North America",

                    marker = dict(color = 'black'),

)



trace5 = go.Scatter(

                    x = df[(df['continent']=='South America')&(df['datetime']>='2020-03-01')].groupby(['date','continent'])['date'].apply(lambda x: np.unique(x)[0]),

                    y = df[(df['continent']=='South America')&(df['datetime']>='2020-03-01')].groupby(['date','continent'])['death_rate'].mean(),

                    mode = "lines",

                    name = "South America",

                    marker = dict(color = 'brown'),

)



data = [trace1,trace2,trace3,trace4,trace5]

layout = dict(title = 'Mean death rate over continents',

              xaxis= dict(title= 'mean deaths/cases %',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
trace1 = go.Scatter(

                    x = df[(df['continent']=='Asia')&(df['datetime']>='2020-03-01')].groupby(['date','continent'])['date'].apply(lambda x: np.unique(x)[0]),

                    y = df[(df['continent']=='Asia')&(df['datetime']>='2020-03-01')].groupby(['date','continent'])['death_rate'].mean()*100,

                    mode = "lines",

                    name = "Asia",

                    marker = dict(color = 'green'),

)



trace2 = go.Scatter(

                    x = df[(df['continent']=='Europe')&(df['datetime']>='2020-03-01')].groupby(['date','continent'])['date'].apply(lambda x: np.unique(x)[0]),

                    y = df[(df['continent']=='Europe')&(df['datetime']>='2020-03-01')].groupby(['date','continent'])['population_coverage'].mean()*100,

                    mode = "lines",

                    name = "Europe",

                    marker = dict(color = 'red'),

)



trace3 = go.Scatter(

                    x = df[(df['continent']=='Africa')&(df['datetime']>='2020-03-01')].groupby(['date','continent'])['date'].apply(lambda x: np.unique(x)[0]),

                    y = df[(df['continent']=='Africa')&(df['datetime']>='2020-03-01')].groupby(['date','continent'])['population_coverage'].mean()*100,

                    mode = "lines",

                    name = "Africa",

                    marker = dict(color = 'blue'),

)



trace4 = go.Scatter(

                    x = df[(df['continent']=='North America')&(df['datetime']>='2020-03-01')].groupby(['date','continent'])['date'].apply(lambda x: np.unique(x)[0]),

                    y = df[(df['continent']=='North America')&(df['datetime']>='2020-03-01')].groupby(['date','continent'])['population_coverage'].mean()*100,

                    mode = "lines",

                    name = "North America",

                    marker = dict(color = 'black'),

)



trace5 = go.Scatter(

                    x = df[(df['continent']=='South America')&(df['datetime']>='2020-03-01')].groupby(['date','continent'])['date'].apply(lambda x: np.unique(x)[0]),

                    y = df[(df['continent']=='South America')&(df['datetime']>='2020-03-01')].groupby(['date','continent'])['population_coverage'].mean(),

                    mode = "lines",

                    name = "South America",

                    marker = dict(color = 'brown'),

)



data = [trace1,trace2,trace3,trace4,trace5]

layout = dict(title = 'Mean population test coverage over continents',

              xaxis= dict(title= 'mean tests/population %',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
df_grouped = df.groupby(['location','continent']).agg({'new_deaths': np.sum, 'gdp_per_capita': np.mean, 'new_cases':np.sum}).reset_index()

df_grouped = df_grouped[(~df_grouped['new_deaths'].isnull())&(~df_grouped['new_cases'].isnull())&(~df_grouped['gdp_per_capita'].isnull())&(~df_grouped['continent'].isnull())]
fig = px.scatter(df_grouped, 

                 x="new_deaths", y="gdp_per_capita", size="new_cases", color="continent",

                 hover_name="location", log_x=True, size_max=60)

fig.show()
df_grouped = df.groupby(['location','continent']).agg({'handwashing_facilities': np.mean, 'new_deaths_smoothed_per_million': np.sum, 'extreme_poverty':np.mean}).reset_index()

df_grouped = df_grouped[(~df_grouped['handwashing_facilities'].isnull())&(~df_grouped['new_deaths_smoothed_per_million'].isnull())&(~df_grouped['extreme_poverty'].isnull())&(~df_grouped['continent'].isnull())]
fig = px.scatter(df_grouped, 

                 x="new_deaths_smoothed_per_million", y="handwashing_facilities", size="extreme_poverty", color="continent",

                 hover_name="location", log_x=True, size_max=60)

fig.show()
df_grouped = df.groupby(['location','continent']).agg({'population_density': np.mean, 'new_deaths_smoothed_per_million': np.sum, 'aged_70_older':np.mean}).reset_index()

df_grouped = df_grouped[(~df_grouped['population_density'].isnull())&(~df_grouped['new_deaths_smoothed_per_million'].isnull())&(~df_grouped['aged_70_older'].isnull())&(~df_grouped['continent'].isnull())]
fig = px.scatter(df_grouped, 

                 x="new_deaths_smoothed_per_million", y="aged_70_older", size="population_density", color="continent",

                 hover_name="location", log_x=True, size_max=60)

fig.show()
df_grouped = df.groupby(['location','continent']).agg({'life_expectancy': np.mean, 'new_deaths_smoothed_per_million': np.sum, 'hospital_beds_per_thousand':np.mean}).reset_index()

df_grouped = df_grouped[(~df_grouped['life_expectancy'].isnull())&(~df_grouped['new_deaths_smoothed_per_million'].isnull())&(~df_grouped['hospital_beds_per_thousand'].isnull())&(~df_grouped['continent'].isnull())]
fig = px.scatter(df_grouped, 

                 x="new_deaths_smoothed_per_million", y="life_expectancy", size="hospital_beds_per_thousand", color="continent",

                 hover_name="location", log_x=True, size_max=60)

fig.show()
df_grouped = df.groupby(['location','continent']).agg({'death_rate': np.mean, 'stringency_index': np.mean, 'new_cases':np.sum}).reset_index()

df_grouped = df_grouped[(~df_grouped['death_rate'].isnull())&(~df_grouped['stringency_index'].isnull())&(~df_grouped['new_cases'].isnull())&(~df_grouped['continent'].isnull())]
fig = px.scatter(df_grouped, 

                 x="death_rate", y="stringency_index", size="new_cases", color="continent",

                 hover_name="location", log_x=True, size_max=60)

fig.show()
df['year_month'] = df['date'].apply(lambda x: x[:7])

df_grouped = df.groupby(['location','year_month']).agg({'new_deaths': np.sum, 'gdp_per_capita': np.mean, 'new_cases':np.sum}).reset_index()

df_grouped = df_grouped[(~df_grouped['new_deaths'].isnull())&(~df_grouped['new_cases'].isnull())&(~df_grouped['gdp_per_capita'].isnull())&(~df_grouped['location'].isnull())]

del df['year_month']
df_grouped = df_grouped[df_grouped['year_month'].isin(['2020-04','2020-05','2020-06','2020-07','2020-08','2020-09'])]
fig = px.scatter(df_grouped[df_grouped['location'].isin(['United States','India','Brazil','Russia'])], 

                 x="new_cases", y="new_deaths", animation_frame="year_month", animation_group="location",

                 size="new_cases", color="location", hover_name="location", facet_col="location",

                 log_x=True, size_max=60,range_x=[5000,10000000], range_y=[100,70000])

fig.show()