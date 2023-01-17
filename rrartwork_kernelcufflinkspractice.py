from IPython.display import display, HTML

import pandas as pd

import numpy as np

import math



# Using plotly + cufflinks in offline mode



import plotly.plotly as py

import plotly.graph_objs as go

import plotly

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import cufflinks as cf

cf.set_config_file(offline=True)
df = pd.read_csv('../input/restaurants.csv',header=0)

df = df.groupby(by="Country").count()["Name"]

df = df.sort_values(ascending=False)
df.iplot(kind='bar')
df1 = pd.read_csv('../input/Pokemon.csv')

df1 = df1.drop('#', axis=1)
df2 = pd.read_csv("../input/fifa18.csv")

df2 = df2.drop(['Photo', 'Flag', 'Club Logo'], axis=1)
df = df1.drop(['Name','Type 2','Legendary','Generation'],axis=1)

df = df.groupby('Type 1').mean()

df['Type 1'] = df.index



df.iplot(kind='barh', y='Attack', x='Type 1', colorscale='rdylbu', title='Attack Strength')
df = df1.drop(['Total', 'Attack', 'Defense', 'Speed', 'HP', 'Type 2', 'Generation', 'Legendary'], axis=1)

df = df.groupby('Type 1').mean()



df.iplot(kind='bar', title='Special Attack and Defense Scores')
df = df1[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']]

df.iplot(kind='box')
df2['Age'].iplot(kind='hist', opacity=0.75, color='rgb(12, 128, 128)', title='Age Distribtution', yTitle='Count', xTitle='Age', bargap = 0.20)
df2['Overall'].iplot(kind='hist', opacity=0.75, color='#007959', title='Overall Rating Distribution', yTitle='Count', xTitle='Overall Rating', bargap = 0.20)
df_spain = pd.DataFrame(df2.loc[df2['Nationality'] == 'Spain']['Overall']).reset_index(drop=True)

df_spain = df_spain.rename(columns={'Overall': 'Spain'})

df_brazil = pd.DataFrame(df2.loc[df2['Nationality'] == 'Brazil']['Overall']).reset_index(drop=True)

df_brazil = df_brazil.rename(columns={'Overall': 'Brazil'})

df_england = pd.DataFrame(df2.loc[df2['Nationality'] == 'England']['Overall']).reset_index(drop=True)

df_england = df_england.rename(columns={'Overall': 'England'})

frames = [df_spain, df_brazil, df_england]

df = pd.concat(frames, sort=False)



colors = ['rgba(171, 50, 96, 0.6)', '#051e3e', 'rgba(80, 26, 80, 0.8)']



df.iplot(

    kind='hist',

    barmode='overlay',

    xTitle='Rating',

    yTitle='Count',

    title='Distribution of Overall Rating by Country',

    opacity=0.75,

    color=colors,

    theme='white')

df_russia = pd.DataFrame(df2.loc[df2['Nationality'] == 'Russia']['Finishing']).reset_index(drop=True)

df_russia = df_russia.rename(columns={'Finishing': 'Russia'})

df_france = pd.DataFrame(df2.loc[df2['Nationality'] == 'France']['Finishing']).reset_index(drop=True)

df_france = df_france.rename(columns={'Finishing': 'France'})

df_argentina = pd.DataFrame(df2.loc[df2['Nationality'] == 'Argentina']['Finishing']).reset_index(drop=True)

df_argentina = df_argentina.rename(columns={'Finishing': 'Argentina'})

df_germany = pd.DataFrame(df2.loc[df2['Nationality'] == 'Germany']['Finishing']).reset_index(drop=True)

df_germany = df_germany.rename(columns={'Finishing': 'Germany'})



frames = [df_russia, df_france, df_argentina, df_germany]

df = pd.concat(frames, sort=False)



df.iplot(kind='box',

        yTitle='Rating',

        title='Descriptive Stats of Finishing Ability by Country')

df = df2.nlargest(100, 'Finishing')

df['Wage'] = df['Wage'] / 10000



df[['Composure', 'Finishing', 'Name', 'Wage']].iplot(

    y='Finishing', mode='markers', x='Composure', colorscale='rdylbu',

    xTitle='Composure', yTitle='Finishing',

    text='Name', title='Player Finshing vs Composure', size=df['Wage'])
df = df2

df = df.nlargest(100, 'Overall')

df['Rank'] = ''

df['Rank'] = np.arange(1, len(df_) + 1)



colors = ['rgba(16, 112, 2, 0.8)', 'rgba(80, 26, 80, 0.8)']



df[['Rank', 'Wage', 'Value', 'Name']].iplot(

    y='Wage', mode='lines+markers', secondary_y = 'Value',

    secondary_y_title='Value', xTitle='Rank', yTitle='Wage',

    text='Name', title='Wage and Rating by Rank', color=colors, theme='white')
df = df2.nlargest(500, 'Value')

df = df.rename(columns={'Overall': 'Actual Rating'})

colors=['#007959', '#FFA505']



df[['Value', 'Actual Rating', 'Potential', 'Name']].iplot(

    kind='scatter',

    mode='markers',

    x='Value',

    y='Actual Rating',

    secondary_y = 'Potential',

    color=colors,

    text='Name',

    name=names,

    xTitle='Market Valuation',

    yTitle='Ratings',

    title='Actual vs Potential Rating by Player Value')
df = df2[['Age', 'Value', 'Wage', 'Potential','Acceleration','Shot Power', 'Sprint Speed', 'Finishing', 'Stamina', 'Strength', 'Vision', 'Ball Control']]



df.corr().iplot(kind='heatmap',colorscale='ylgn')
df = df2.nlargest(100, 'Overall')

df = pd.DataFrame(df.groupby('Club').size())

df.columns = ['Count']

df['Club'] = df.index





df.iplot(kind='pie', labels='Club', values='Count', title='Number of Players by Club', hoverinfo="label+percent+name", hole=0.3, theme='white')
df = df2.groupby("Nationality").size().reset_index(name="Count")



df.iplot(

    kind='choropleth', locations='Nationality',  z ='Count',

    text = 'Nationality', locationmode = 'country names', theme='white',

    colorscale='oranges', title = "Nationalities of FIFA 18 Players",

    projection = dict(

            type = 'natural earth'

        ))
df = df2.nlargest(50, 'Potential')

df.iplot(x='Composure', y='Positioning', z='Finishing', kind='scatter3d', xTitle='Composure', yTitle='Positioning',

         zTitle='Finishing', theme='pearl', text= 'Name',

         categories='Club', title='Intersection between Composure, Positioning and Finishing Ability')