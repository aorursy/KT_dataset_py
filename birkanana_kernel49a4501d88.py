# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
import plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.express as px
import plotly.offline as pyOff
import colorlover as cl
import ipywidgets as widgets
from ipywidgets import interact, interactive
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')
test_df = pd.read_csv('/kaggle/input/titanic/test.csv')
test_PassengerId = test_df['PassengerId']
train_df.describe()
@interact
def show_articles_more_than(column='Age', x=30):
    return train_df.loc[train_df[column]> x ]
trace1 = go.Scatter3d(
    x=train_df.Age,
    y=train_df.Pclass,
    z=train_df.Survived,
    mode='markers',
    marker=dict(
        size=10,
        color='rgb(255,0,0)',                
    )
)

data = [trace1]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0  
    )
    
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
trace1 = go.Scatter(
    x=train_df.Age,
    y=train_df.Survived,
    name = "research"
)
trace2 = go.Scatter(
    x=train_df.Sex,
    y=train_df.Survived,
    xaxis='x2',
    yaxis='y2',
    name = "citations"
)
trace3 = go.Scatter(
    x=train_df.Pclass,
    y=train_df.Survived,
    xaxis='x3',
    yaxis='y3',
    name = "income"
)
trace4 = go.Scatter(
    x=train_df.SibSp,
    y=train_df.Survived,
    xaxis='x4',
    yaxis='y4',
    name = "total_score"
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
    title = 'Research, citation, income and total score VS World Rank of Universities'
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)

50100150200501005010015020040608010050100150200406080100501001502006080
# first line plot
trace1 = go.Scatter(
    x=train_df.Pclass,
    y=train_df.Age,
    name = "teaching",
    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
)
# second line plot
trace2 = go.Scatter(
    x=train_df.Pclass,
    y=train_df.SibSp,
    xaxis='x2',
    yaxis='y2',
    name = "income",
    marker = dict(color = 'rgba(160, 112, 20, 0.8)'),
)
data = [trace1, trace2]
layout = go.Layout(
    xaxis2=dict(
        domain=[0.6, 0.95],
        anchor='y2',        
    ),
    yaxis2=dict(
        domain=[0.6, 0.95],
        anchor='x2',
    ),
    title = 'Income and Teaching vs World Rank of Universities'

)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
@interact
def scatter_plot(x=list(train_df.select_dtypes('number').columns),
                 y=list(train_df.select_dtypes('number').columns)[1:],
                 theme=list(cf.themes.THEMES.keys()),
                 colorscale=list(cf.colors._scales_names.keys())):
                train_df.iplot(kind='scatter', x=x, y=y, mode='markers',
                              xTitle=x.title(), yTitle=y.title(),
                               title=f'{y.title()} vs {x.title()}',
                              theme=theme, colorscale=colorscale)
@widgets.interact_manual(
color=['blue', 'red', 'green'], lw=(1., 10.))
def plot(freq=1., color='blue', lw=2, grid=True):
    t= np.linspace(-1., +1., 1000)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(t, np.sin(2 * np.pi * freq * t),
           lw=lw, color=color)
    ax.grid(grid)
trace0 = go.Box(
    y=train_df.Survived,
    name = 'Survived',
    marker = dict(
        color = 'rgb(12, 12, 140)',
    )
)
trace1 = go.Box(
    y=train_df.Parch,
    name = 'Parch',
    marker = dict(
        color = 'rgb(12, 128, 128)',
    )
)
data = [trace0, trace1]
iplot(data)