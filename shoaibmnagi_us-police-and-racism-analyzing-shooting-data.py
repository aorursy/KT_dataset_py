import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go







import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



df = pd.read_csv('/kaggle/input/data-police-shootings/fatal-police-shootings-data.csv')

df.head()

df.shape
#splitting date into its components. It might help us later in the analysis. 

df['month'] = pd.to_datetime(df['date']).dt.month

df['year'] = pd.to_datetime(df['date']).dt.year

df.head()
#lets identify the columns that have a lot of missing values

print(df.isna().sum()*100/5416)
#race

df_clean = df.dropna(subset = ['race', 'age'])



df_clean['armed'].fillna('unarmed', inplace=True)

df_clean['flee'].fillna('Not fleeing', inplace=True)



print(df_clean[df_clean['gender'].isna() == True]) #missing value's name is 'Scout Schultz'. He was an intersex, non-binary student.

#We don't have the right to decide his gender for him. So we will drop his name from the data.



df_clean = df_clean.dropna(subset = ['gender'])

df_clean.shape
print(df['race'].unique())
shootings_race = df_clean[df_clean['manner_of_death'] == 'shot']

shootings_race.shape #4564 shootings
fig = px.histogram(shootings_race, x = 'race',

                  title = "Distribution of Police Shootings by Race")



#let's add a line to represent the number of shootings for each race if they were proportional to their population in the US

#white: 0.618, b: 0.132, h: 0.178, a: 0.053, n: 0.01, o: 0.009 These are rough estimates based on 2015 values.

races = ['A','W','H','B','N','O']

race_prop = np.array([0.053, 0.618, 0.178, 0.132, 0.01, 0.009])

fig.add_trace(go.Scatter(x = races, y = shootings_race.shape[0] * race_prop))



fig.show()

unarmed = df_clean[df_clean['armed'] == 'unarmed']

fig2 = px.histogram(unarmed, x = 'race',

                   title = 'Distribution of Police Shootings by Race where the Suspect was Unarmed') #this is messing the order on the x axis compared to the last graph, idk why

races2 = ['H','W','B','N','O','A'] 

race_prop2 = np.array([0.178, 0.618, 0.132, 0.01, 0.009, 0.053])

fig2.add_trace(go.Scatter(x = races2, y = unarmed.shape[0] * race_prop2))

fig2.show()
import plotly.figure_factory as ff

np.random.seed(1)

age_data = df_clean['age']

age_labels = ['Age']

fig = ff.create_distplot([age_data], age_labels)

fig.update_layout(title_text='Distribution of Police Shooting Victims by Age')



fig.show()

df_black = df_clean[df_clean['race'] == 'B']

black_age_data = df_black['age']

fig = ff.create_distplot([black_age_data], age_labels)

fig.update_layout(title_text='Distribution of Black Police Shooting Victims by Age')



fig.show()
from plotly.subplots import make_subplots



trigger_states = df_clean['state'].value_counts()[:20]

trigger_states = pd.DataFrame(trigger_states)

trigger_states = trigger_states.reset_index()

trigger_states['percent'] = trigger_states['state']*100/df_clean.shape[0]

s = [11.91, 8.74, 6.47, 2.19, 1.74, 3.2, 1.19, 3.52, 3.16, 2.29, 1.85, 2.06, 1.40, 3.86, 3.82, 1.48, 0.63, 2.03, 5.86, 2.57]

trigger_states['state_prop'] = s



fig = make_subplots(rows = 1, cols = 2, specs = [[{},{}]], shared_xaxes = True,

                  shared_yaxes = False, vertical_spacing = 0.001)



fig.append_trace(go.Bar(

    x = trigger_states['percent'],

    y = trigger_states['index'],

    marker = dict(

        color = 'red',

        line = dict(

            color = 'red',

            width = 1),

    ),

    name = 'Distribution of Shootings by State',

    orientation = 'h',

), 1, 1)



fig.append_trace(go.Bar(

    x = trigger_states['state_prop'],

    y = trigger_states['index'],

    marker = dict(

        color = 'goldenrod',

        line = dict(

            color = 'goldenrod',

            width = 1),

    ),

    name = '% of total US populaiton in the State',

    orientation = 'h',

), 1, 2)





fig.update_layout(

    title = 'Twenty US States with the most Police Shootings',

    yaxis = dict(

        showgrid = False,

        showline = False,

        showticklabels = True,

        domain = [0, 0.85],

    ),

    yaxis2 = dict(

        showgrid = False,

        showline = True,

        showticklabels = True,

        linecolor = 'rgba(102, 102, 102, 0.8)',

        domain = [0, 0.85],

    ),

    xaxis = dict(

        zeroline = False,

        showline = False,

        showticklabels = True,

        showgrid = True,

        domain = [0, 0.42],

    ),

    xaxis2 = dict(

        zeroline = False,

        showline = False,

        showticklabels = True,

        showgrid = True,

        domain = [0.5, 0.92],

    ),

    legend = dict(x = 0.029, y = 1.038, font_size = 10),

    margin = dict(l = 100, r = 20, t = 70, b = 70),

    paper_bgcolor = 'rgb(248, 248, 255)',

    plot_bgcolor = 'rgb(248, 248, 255)',

)



fig.show()
years = df_clean[['year','race']]

years['shootings'] = 1

years = years.groupby(['year','race']).sum()

years = years.reset_index()

fig = px.bar(years , y = 'shootings', x = 'year',color = 'race', barmode = 'group', title = 'Shootings from 2015-2020',

            color_discrete_sequence = px.colors.qualitative.D3)

fig.show()