# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
init_notebook_mode(connected=True)

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

import cufflinks as cf
cf.go_offline()
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("..//input/StudentsPerformance.csv")
df.info()
df.head()
df.describe()
df.iloc[:,:5].describe()
layout = dict(title = "Stacked Count", xaxis = dict(title = 'Type'), yaxis = dict(title = 'Count'),
              barmode='stack')
trace= []
for each in df.select_dtypes(include=['object']).columns:
    df2 = df[each].value_counts()
    for i, each2 in enumerate(df2.index):
        trace.append(go.Bar(x = [each], y =[df2.values[i]],name=each2,legendgroup= each))

fig = go.Figure(data= trace, layout=layout)
py.offline.iplot(fig)
py.offline.iplot([go.Parcats({'dimensions':[{'label': each, 'values':df[each].values} for each in df.select_dtypes(include=['object']).columns]})])
fig = ff.create_distplot(hist_data=[df[each] for each in df.select_dtypes(include=['int64']).columns], group_labels=df.select_dtypes(include=['int64']).columns,
                        curve_type='normal',histnorm='probability')
fig['layout'].update(title='Distribution Plot of Scores')
py.offline.iplot(fig)
layout = dict(title = "Cumulative Distribution of Scores",xaxis = dict(title = 'Score'), yaxis = dict(title = '%'))
trace= []
for each in df.select_dtypes(include=['int64']).columns:
    trace.append(go.Histogram(x = df[each], cumulative=dict(enabled=True),histnorm='percent',name=each))

fig = go.Figure(data= trace, layout=layout)
py.offline.iplot(fig)
layout = dict(title = "Box Plot of Scores",xaxis = dict(title = 'Score Type'), yaxis = dict(title = 'Score'))
trace= []
for each in df.select_dtypes(include=['int64']).columns:
    trace.append(go.Box(y = df[each],orientation='v', name=each))

fig = go.Figure(data= trace, layout=layout)
py.offline.iplot(fig)
layout = dict(title = "Histogram of Scores",xaxis = dict(title = 'Score'), yaxis = dict(title = 'Count'))
trace= []
for each in df.select_dtypes(include=['int64']).columns:
    trace.append(go.Histogram(x = df[each],name=each))

fig = go.Figure(data= trace, layout=layout)
py.offline.iplot(fig)
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(df.corr(method='spearman'),
                       ax=ax,cmap='coolwarm',
                       annot=True)
for typ in df.select_dtypes(include=['object']).columns:
    for score_typ in df.select_dtypes(include=['Int64']).columns:
        trace= []
        for uniq_typ in df[typ].unique():
            layout = dict(title = "Cumulative Distribution of " + score_typ +  " by " + typ ,xaxis = dict(title = 'Score'), yaxis = dict(title = '%'))
            trace.append(go.Histogram(x = df[df[typ] == uniq_typ][score_typ], cumulative=dict(enabled=True),histnorm='percent',name=uniq_typ))
        fig = go.Figure(data= trace, layout=layout)
        py.offline.iplot(fig)
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=500)
X = pd.concat([pd.get_dummies(df.select_dtypes(include=['object']),prefix=df.select_dtypes(include=['object']).columns, prefix_sep='_')], axis=1)
layout = dict(title = "Feature Importance", yaxis = dict(title = '%'),
              barmode='stack')
trace= []
for each in df.select_dtypes(include=['int64']).columns:
    rf.fit(X, df[each])
    d = dict(zip(X.columns, rf.feature_importances_*100))
    trace.append(go.Bar(x = list(d.keys()), y =list(d.values()),name=each))
fig = go.Figure(data= trace, layout=layout)
py.offline.iplot(fig)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df2 = pd.DataFrame([le.fit_transform(df[each]) for each in df.select_dtypes(include=['object']).columns]).T
df2.columns = df.select_dtypes(include=['object']).columns
layout = dict(title = "Feature Importance", yaxis = dict(title = '%'),
              barmode='stack')
trace= []
for each in df.select_dtypes(include=['int64']).columns:
    rf.fit(df2, df[each])
    d = dict(zip(df2.columns, rf.feature_importances_*100))
    trace.append(go.Bar(x = list(d.keys()), y =list(d.values()),name=each))
fig = go.Figure(data= trace, layout=layout)
py.offline.iplot(fig)