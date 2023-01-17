import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly

import plotly.express as px

import plotly.graph_objects as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot, plot_mpl

import plotly.offline as py

init_notebook_mode(connected=True)
# Resident population of the United States by sex and age as of July 1, 2018

population_df = pd.DataFrame({

    'Age Range Start':       [0, 0, 5, 5, 10, 10, 15, 15, 20, 20, 25, 25, 30, 30, 35, 35, 40, 40, 45, 45, 50, 50, 55, 55, 60, 60, 65, 65, 70, 70, 75, 75, 80, 80, 85, 85],

    'Age Range End':         [4, 4, 9, 9, 14, 14, 19, 19, 24, 24, 29, 29, 34, 34, 39, 39, 44, 44, 49, 49, 54, 54, 59, 59, 64, 64, 69, 69, 74, 74, 79, 79, 84, 84, 130, 130],

    'Sex':                   ['M',  'F', 'M',  'F', 'M',  'F', 'M',  'F', 'M',  'F', 'M',  'F', 'M',  'F', 'M',  'F', 'M',  'F', 'M',  'F', 'M',  'F', 'M',  'F', 'M',  'F', 'M',  'F', 'M',  'F', 'M',  'F', 'M',  'F', 'M',  'F'],

    'Population (millions)': [10.13, 9.68, 10.32, 9.88, 10.66, 10.22, 10.77, 10.32, 11.2, 10.67, 12.02, 11.54, 11.19, 10.94, 10.79, 10.77, 9.8, 9.92, 10.26, 10.48, 10.28, 10.61, 10.67, 11.27, 9.73, 10.6, 8.03, 9.05, 6.21, 7.19, 4.14, 5.12, 2.59, 3.54, 2.33, 4.22]

})



population_df = population_df.groupby(['Age Range Start', 'Age Range End']).sum().reset_index()



age_range_formatter = lambda x: "{:.0f}{}".format(x['Age Range Start'], f"-{x['Age Range End']:.0f}" if x['Age Range End'] < 100 else "+")



population_df['Age Range'] = population_df.apply(age_range_formatter, axis=1)



fig = px.bar(population_df, x="Age Range", y="Population (millions)")

fig.update_traces(marker_color='lightsalmon', marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6)

fig.update_layout(

    font=dict(

        size=18,

        #color="#7f7f7f"

    )

)

fig.show()
death_rate_df = pd.DataFrame({

    'Age Range Start': [0, 10, 20, 30, 40, 50, 60, 70, 80],

    'Age Range End': [9, 19, 29, 39, 49, 59, 69, 79, 130],

    'Death Rate': [0, 0.002, 0.002, 0.002, 0.004, 0.013, 0.036, 0.08, 0.148]

})



death_rate_df['Age Range'] = death_rate_df.apply(age_range_formatter, axis=1)



fig = px.bar(death_rate_df, x="Age Range", y="Death Rate")

fig.update_traces(marker_color='lightsalmon', marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6)

fig.update_layout(

    font=dict(

        size=18,

        #color="#7f7f7f"

    )

)

fig.show()
# Match age ranges

def cross_join(df1, df2):

    df1_copy = df1.copy()

    df2_copy = df2.copy()

    df1_copy['key'] = 0

    df2_copy['key'] = 0

    return df1_copy.merge(df2_copy, how='outer', on='key')



df = cross_join(population_df, death_rate_df)

df = df[(df["Age Range Start_x"] >= df["Age Range Start_y"]) & (df['Age Range End_x'] <= df['Age Range End_y'])]

df = df.groupby(['Age Range Start_y', 'Age Range End_y']).agg({'Population (millions)': 'sum', 'Death Rate': 'max'}).reset_index()

df = df.rename(columns={'Age Range Start_y': 'Age Range Start', 'Age Range End_y': 'Age Range End'})

df['Age Range'] = df.apply(age_range_formatter, axis=1)

df['Population - After (millions)'] = df['Population (millions)'] * (1 - df['Death Rate'])

df = df.rename(columns={'Population (millions)': 'Population - Before (millions)'})



# Tidy up the dataframe

df = pd.melt(df, id_vars=['Age Range Start', 'Age Range End', 'Age Range', 'Death Rate'], value_vars=['Population - Before (millions)', 'Population - After (millions)'], var_name='Relative Time to COVID-19', value_name='Population (millions)')

df['Relative Time to COVID-19'] = df['Relative Time to COVID-19'].apply(lambda x: 'Before' if 'Before' in x else 'After')
fig = px.bar(df, x='Age Range', y='Population (millions)', color='Relative Time to COVID-19', barmode='group')



y_name = 'Population (millions)'

x_name = 'Age Range'

color_name = 'Relative Time to COVID-19'



fig = go.Figure()

fig.add_trace(go.Bar(

    x=df[df[color_name] == 'Before'][x_name],

    y=df[df[color_name] == 'Before'][y_name],

    name='Before',

    marker_color='indianred'

))

fig.add_trace(go.Bar(

    x=df[df[color_name] == 'After'][x_name],

    y=df[df[color_name] == 'After'][y_name],

    name='After',

    marker_color='lightsalmon'

))



fig.update_traces(marker_line_color='rgb(8,48,107)', marker_line_width=1.5)



fig.update_layout(

    title=dict(

        text="Potential Impact of COVID-19 on U.S. Population",

        x=0.5,

        xanchor='center'

    ),

    barmode='group',

    xaxis_tickangle=-45,

    yaxis=dict(

        title=y_name

    ),

    xaxis=dict(

        title=x_name

    ),

    legend=dict(

        title=color_name

    ),

    font=dict(

        size=18

    )

)



fig.show()