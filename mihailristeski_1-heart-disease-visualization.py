import plotly

import numpy as np

import pandas as pd

import plotly.graph_objs as go

import plotly.figure_factory as ff



import os

print(os.listdir("../input"))



import seaborn as sns

import matplotlib.pyplot as plt

plotly.offline.init_notebook_mode(connected=True)



# Прочитај ги податоците во .csv file

data = pd.read_csv('../input/heart.csv', sep=',')
# "Опис" на податоците со број, средина, минимум, максимум.

data.describe()
labels = ['Маш','Жена']

values = [len(data[(data["sex"] == 1)]), len(data[(data["sex"] == 0)])]

colors = ['#0000FF', '#FF00FF']

trace = go.Pie(labels=labels, values=values, hoverinfo='label+percent', textinfo='value'

               , textfont=dict(size=30), marker=dict(colors=colors))

plotly.offline.iplot([trace], filename='styled_pie_chart')



labels = ['fbs > 120','fbs <= 120']

values = [len(data[(data["fbs"] == 1)]), len(data[(data["fbs"] == 0)])]

colors = ['#FF0000', '#00ff00']

trace = go.Pie(labels=labels, values=values, hoverinfo='label+percent', textinfo='value'

               , textfont=dict(size=30), marker=dict(colors=colors))

plotly.offline.iplot([trace], filename='styled_pie_chart')



labels = ['Да','Не']

values = [len(data[(data["exang"] == 1)]), len(data[(data["exang"] == 0)])]

colors = ['#00FF0', '#FF0000']

trace = go.Pie(labels=labels, values=values, hoverinfo='label+percent', textinfo='value'

               , textfont=dict(size=30), marker=dict(colors=colors))

plotly.offline.iplot([trace], filename='styled_pie_chart')



labels = ['ca0','ca1', 'ca2', 'ca3']

values = [len(data[(data["ca"] == 0)]), len(data[(data["ca"] == 1)])

          , len(data[(data["ca"] == 2)]), len(data[(data["ca"] == 4)])]

colors = ['#00FF0', '#FF0000', '#0000FF', '#00FFFF']

trace = go.Pie(labels=labels, values=values, hoverinfo='label+percent', textinfo='value'

               , textfont=dict(size=30), marker=dict(colors=colors))

plotly.offline.iplot([trace], filename='styled_pie_chart')



labels = ['normal','fixed defect', 'reversable defect']

values = [len(data[(data["thal"] == 0)]), len(data[(data["thal"] == 1)]), len(data[(data["thal"] == 2)])]

colors = ['#00FF0', '#FF0000', '#0000FF']

trace = go.Pie(labels=labels, values=values, hoverinfo='label+percent', textinfo='value'

               , textfont=dict(size=30), marker=dict(colors=colors))

plotly.offline.iplot([trace], filename='styled_pie_chart')



labels = ['Срцев удар','Нема срцев удар']

values = [len(data[(data["target"] == 1)]), len(data[(data["target"] == 0)])]

colors = ['#00FF0', '#FF0000']

trace = go.Pie(labels=labels, values=values, hoverinfo='label+percent', textinfo='value'

               , textfont=dict(size=30), marker=dict(colors=colors))

plotly.offline.iplot([trace], filename='styled_pie_chart')
# Табела на корелација помеѓу сите колони од податоците.

data.corr()
sns.set()

sns.catplot(x="sex", y="thalach",col="target",hue="cp", kind="swarm",data=data)

sns.catplot(x="sex", y="oldpeak",col="target",hue="exang", kind="swarm",data=data)