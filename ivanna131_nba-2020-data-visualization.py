# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.graph_objects as go

import plotly.express as px

from plotly.subplots import make_subplots

import matplotlib.pyplot as plt

import seaborn as sns

from datetime import datetime

from datetime import date

import plotly.figure_factory as ff







# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("../input/nba2k20-player-dataset/nba2k20-full.csv",parse_dates=True)

data.head()
data.describe()
data.dtypes
data['weight'] = [float(data['weight'][i].split()[3]) for i in range(len(data))]

data['height'] = [float(data['height'][i].split()[-1]) for i in range(len(data))]

data['salary'] = [int(data['salary'][i].split('$')[1]) for i in range(len(data))]

data['jersey'] = [int(data['jersey'][i].split('#')[1]) for i in range(len(data))]



data['b_day'] = data['b_day'].apply(lambda x: datetime.strptime(x, '%m/%d/%y').date())

data['age'] = (datetime.today().date() - data['b_day']).astype('<m8[Y]').astype('int64')



data['draft_round'] = data['draft_round'].apply(lambda x: 0 if x=='Undrafted' else int(x)) 

data['draft_peak'] = data['draft_peak'].apply(lambda x: 0 if x=='Undrafted' else int(x)) 



data['college'] = data['college'].fillna('No college')

data['team'] = data['team'].fillna('No team')

data.head()
data.dtypes
plt.figure(figsize=(30,15))

sns.set(font_scale=1.8)

sns.heatmap(data.corr(),cmap='Blues',annot=True)
fig = make_subplots(rows=1, cols=2, specs=[[{"type": "histogram"}, {"type": "histogram"}]])

fig.add_trace(go.Histogram(x=data['height']*100,

                           xbins=dict(

                               start=150,

                               end=280,

                               size=3

                           ),

                           name='height, cm', hovertemplate='Count: %{y}<br>Height: %{x}cm'

                           ), col=1, row=1)

fig.add_trace(go.Scatter(x=[data['height'].mean()*100, data['height'].mean()*100], y=[0, 91],

                         mode='lines',

                         name='Mean height', hovertemplate='Mean: %{x:.2f}'))

fig.update_layout(hovermode='x')

fig.add_trace(go.Histogram(x=data['weight'],

                           xbins=dict(

                               start=min(data['weight']),

                               end=max(data['weight']),

                               size=3

                           ),

                           name='weight, kg', hovertemplate='Count: %{y}<br>Weight: %{x}kg'

                           ), col=2, row=1)

fig.add_trace(go.Scatter(x=[data['weight'].mean(), data['weight'].mean()], y=[0, 91],

                         mode='lines',

                         name='Mean weight', hovertemplate='Mean: %{x:.2f}'),col=2, row=1)



fig.update_layout(title={

        'text': "Height and weight distribution",

        'y':0.9,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'})

fig.show()
fig = px.scatter(data, x="weight", y="height", 

                 marginal_x="box", marginal_y="violin",

                 color_discrete_sequence=['orange']

                )

fig.show()
country_count = data['country'].value_counts()

fig = go.Figure(go.Pie(labels=country_count.index, values=country_count.values, hole=0.4,textinfo= "none"))

fig.update_layout(title={

        'text': "Percentage of players by country",

        'y':0.9,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'})



fig.show()
college_count = data['college'].value_counts()

fig = go.Figure(go.Pie(labels=college_count.index, values=college_count.values, hole=0.4,textinfo= "none"))

fig.update_layout(title={

        'text': "Percentage of players by college",

        'y':0.9,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'})



fig.show()
data['position'] = data['position'].apply(lambda x: 'F-C' if x=='C-F' else x) #union related positions

data['position'] = data['position'].apply(lambda x: 'F-G' if x=='G-F' else x)




fig = px.scatter(data.sort_values(by='salary', ascending=False)[:50], 

                 x="salary", y="age",

                 size="rating", color="position", 

                 title="Top-50 players",

                 log_x=True, size_max=20)

fig.show()
fig = px.scatter(

    data, x='rating', y='salary', opacity=0.65,

    trendline='ols', trendline_color_override='darkblue', 

    facet_col='position', facet_col_wrap=3, color='salary'

)

fig.show()
fig = px.scatter(data, x='rating', y='salary', opacity=0.65,

                 trendline='ols', trendline_color_override='darkblue', 

                 facet_col='draft_round', facet_col_wrap=3, color='salary'

                )

fig.show()
data_team = data[['team', 'rating']].groupby('team').mean().reset_index()

data_team = data_team.sort_values(by='rating', ascending=False)
fig = px.bar(data_team.query("team != 'No team'"), 

             x='team', y='rating', color='team', 

             labels={'rating':'mean rating of players'},

             title='Mean rating of players for each team',

             color_discrete_sequence=px.colors.qualitative.Safe)

fig.show()
data_height = data[['height', 'country']].groupby('country').mean().reset_index()

data_height = data_height.sort_values(by='height', ascending=False)

fig = px.bar(data_height, 

             x='country', y='height', color='country', 

             labels={'height':'mean height of players'},

             color_discrete_sequence=px.colors.qualitative.Vivid,

             title='Mean height of each country'

            )

fig.show()
fig = px.box(data, x="draft_round", y="salary", 

             color="draft_round",

             title='Salary exploring by draft_round',

             points='all'

            )

fig.update_traces(quartilemethod="exclusive") 

fig.show()