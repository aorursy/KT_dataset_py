import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()
import seaborn as sns
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
%matplotlib inline
import os
!ls ../input/
df = pd.read_csv(r'../input/311-service-requests-rodent-baiting.csv', parse_dates = ['Creation Date','Completion Date'])
df.head()
df = df[df.creationdate.dt.year==2018]
df.columns = [col.lower().replace(' ','') for col in df.columns]
df.columns
premwithrats = df.groupby('ward')['numberofpremiseswithrats'].sum().sort_values(ascending = False).reset_index()
premwithrats.head()
data = [go.Bar(x=premwithrats['ward'], y=premwithrats['numberofpremiseswithrats'])]
layout = go.Layout(
    title='Number of Premises with Rats per Ward 2018',
    xaxis=dict(
        title='Ward',
        tickmode='linear',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ))
fig = go.Figure(data = data, layout = layout)
iplot(fig)

df['completiondate']=pd.to_datetime(df['completiondate'])
TimeGroupDF = df.groupby(df.creationdate.dt.month)['numberofpremisesbaited'].sum()
TimeGroupDF.plot()
pltTS = pd.DataFrame(df.groupby(df.creationdate)['numberofpremisesbaited'].sum())
pltTS = pltTS.reset_index()
pltTS.dtypes
pltTS = pltTS[pltTS.numberofpremisesbaited != pltTS['numberofpremisesbaited'].max()]
#pltTS['CreationDate'] = pltTS['CreationDate'].apply(lambda x: str(x))
sns.distplot(pltTS['numberofpremisesbaited'], hist = True)
# sepcify that we want a scatter plot with, with date on the x axis and meet on the y axis
data = [go.Scatter(x=pltTS.creationdate, y=pltTS.numberofpremisesbaited)]
fig = data#dict(data = data, layout = layout)
iplot(fig)