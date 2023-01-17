# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

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
df = pd.read_csv("..//input/who_suicide_statistics.csv")
df.info()
df.head(5)
df['year'] = df['year'].astype('object')
df.select_dtypes(include=['object']).describe()
df.select_dtypes(exclude=['object']).describe()
df['rank_suicide_year_country'] = df.groupby(['year', 'country'])['suicides_no'].rank(ascending=False,method='dense')
dic = dict(df.groupby(['year', 'country'])['suicides_no'].sum())
def f(x):
    return (dic[(x['year'], x['country'])])
dic2 = dict(df.groupby(['year', 'country'])['population'].sum())
def f2(x):
    return (dic2[(x['year'], x['country'])])
df['tot_suicide_year'] = df[['year', 'country']].apply(f, axis=1)
df['tot_population_year'] = df[['year', 'country']].apply(f2, axis=1)
df['Among_year'] = df['year'].map(dict(df.groupby(['year'])['suicides_no'].size()))
df['rank_tot_suicide_year'] = df.groupby(['year'])['tot_suicide_year'].rank(ascending=False,method='dense')
df['Text1'] = df.apply(lambda x: "<b>Year: {:} </b><br><b>".format(x['year']), axis=1)
#df['Text2'] = df.apply(lambda x: "</b><br><b> Top Among {:,.0f} Countries with {:,.0f} Suicides</b><br>".format(x['Among_year'],x['tot_suicide_year']), axis=1)
df['Text2'] = df.apply(lambda x: "</b><br><b> Top with {:,.0f} Suicides</b><br>".format(x['tot_suicide_year']), axis=1)
df['Text3'] = df.apply(lambda x: "<b> with {:,.0f} Suicides</b><br>".format(x['tot_suicide_year']), axis=1)
df['Text4'] = df.apply(lambda x: "<b> Most Affected is {:}s of Age Group {:} with {:,.0f} Suicides</b>".format(x['sex'], x['age'], x['suicides_no']), axis=1)
layout = dict(title = "Suicide - Top Five Countries by Year", xaxis = dict(title = 'Year'), yaxis = dict(title = 'Total Suicides'), barmode='stack')
trace= []
dic = dict(zip(range (1, 6), ["First", "Second", "Third", "Fourth", "Fifth"]))
for each in range(5,0,-1):
    df2 = df[(df['rank_tot_suicide_year'] == each) & (df['rank_suicide_year_country'] == 1)]
    if each == 1:
        trace.append(go.Bar(x = df2.year, y = df2['tot_suicide_year'], hovertext = df2['Text1'] +  df2['country'] + df2['Text2'] + df2['Text4'] , name=dic[each],  hoverinfo = "text"))
    else:
        trace.append(go.Bar(x = df2.year, y = df2['tot_suicide_year'], hovertext = '<b>' + df2['country'] + '</b>' + '<br><b>'+ dic[each] + '</b>' + df2['Text3'] + df2['Text4'] , name=dic[each],  hoverinfo = "text"))
fig = go.Figure(data= trace, layout=layout)
py.offline.iplot(fig)

