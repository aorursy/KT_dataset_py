import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

#import Dataset
dataset = pd.read_csv("../input/bus-breakdown-and-delays.csv")

#Delete the wrong data because one data occured in 2020 (Now it is 2018)
import datetime
now = datetime.datetime.now().strftime("%Y-%m-%d")

for i in range(0,len(dataset['Occurred_On'])):
    if dataset.iloc[i]['Occurred_On'] > now:
        df_new = dataset.drop(dataset.index[i])
# import plotly
import plotly.plotly as py
import plotly.graph_objs as go

# allow code to show up in a notebook!
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()
df_new['YearMonth'] = df_new['Occurred_On'].apply(lambda x: x[:7])
vc = pd.DataFrame(df_new['YearMonth'].value_counts().sort_index())
vc = vc.reset_index()
vc = vc.rename(index=str, columns={"index": "YearMonth", "YearMonth": "Count"})
data = [go.Scatter(x=vc['YearMonth'], y=vc['Count'])]

#specify the layout of our figure
layout = dict(title = "Number of Breakdowns per Month",
              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False))

# create and show our figure
fig = dict(data = data, layout = layout)
iplot(fig)

runningLate = df_new['YearMonth'][df_new['Breakdown_or_Running_Late'] == 'Running Late'].value_counts().sort_index()
breakDown = df_new['YearMonth'][df_new['Breakdown_or_Running_Late'] == 'Breakdown'].value_counts().sort_index()
from plotly import tools

trace1 = go.Scatter(
    x=vc['YearMonth'],
    y=runningLate.tolist(),
    showlegend=False,
    name='Running Late'
)
trace2 = go.Scatter(
    x=vc['YearMonth'],
    y=breakDown.tolist(),
    showlegend=False,
    name='Breakdown'
)

fig = tools.make_subplots(
    rows=2, 
    cols=1,
    subplot_titles=('Running Late', 'Breakdowns'), 
    shared_xaxes=False)

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 2, 1)

iplot(fig)
df_new['dummy'] = 1

schoolYearReason = df_new.groupby(by=['School_Year','Reason']).count()['dummy'].unstack()
trace1 = go.Bar(
    x=schoolYearReason.columns.tolist(),
    y=schoolYearReason.iloc[0].tolist(),
    name='2015-2016'
)
trace2 = go.Bar(
    x=schoolYearReason.columns.tolist(),
    y=schoolYearReason.iloc[1].tolist(),
    name='2016-2017'
)
trace3 = go.Bar(
    x=schoolYearReason.columns.tolist(),
    y=schoolYearReason.iloc[2].tolist(),
    name='2017-2018'
)
trace4 = go.Bar(
    x=schoolYearReason.columns.tolist(),
    y=schoolYearReason.iloc[3].tolist(),
    name='2018-2019'
)

data = [trace1, trace2, trace3, trace4]
layout = go.Layout(
    barmode='group', title = "Reason of Breakdowns",
)

fig = dict(data = data, layout = layout)
iplot(fig)