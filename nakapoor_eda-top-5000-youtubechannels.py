import pandas as pd
import numpy as np

from IPython.display import HTML, display
import tabulate
from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
ytData = pd.read_csv('../input/data.csv')
print('size of data : ', ytData.shape)
ytData.head()
ytData.tail()
ytData.info()
total = ytData.isnull().sum().sort_values(ascending = False)
percent = (ytData.isnull().sum() / ytData.isnull().count()*100).sort_values(ascending = False)
missing_application_train_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_application_train_data.head(20)
print((ytData['Video Uploads'] == '--').value_counts())
print((ytData['Subscribers'] == '-- ').value_counts())
def bar_chart(lables, values):
    trace = go.Bar(
        x=lables,
        y=values,
        showlegend=False,
        marker=dict(
            color='rgba(28,32,56,0.84)',
        )
    )
    return trace

feats_counts = ytData['Grade'].value_counts()
#print(feats_counts)
trace = bar_chart(lables = feats_counts.index, values = feats_counts)

fig = tools.make_subplots(rows=1, cols=1, vertical_spacing=0.06)#, 
                          #subplot_titles=["# of subscribers for youtube channels"])
fig.append_trace(trace, 1, 1)

fig['layout'].update(height=400, width=600, paper_bgcolor='rgb(233,233,233)', title="Grades distribution for top 5000 subscribers")

py.iplot(fig, filename='plots_1')
def bar_chart(lables, values):
    trace = go.Bar(
        x=lables,
        y=values,
        showlegend=False,
        marker=dict(
            color='rgba(28,32,56,0.84)',
        )
    )
    return trace

feats_counts = (ytData[ytData['Grade'] == 'A++ ']['Video views'])
trace1 = bar_chart(lables = feats_counts.index, values = feats_counts)

feats_counts = ytData[ytData['Grade'] == 'A+ ']['Video views']
trace2 = bar_chart(lables = feats_counts.index, values = feats_counts)

feats_counts = ytData[ytData['Grade'] == 'A ']['Video views']
trace3 = bar_chart(lables = feats_counts.index, values = feats_counts)

feats_counts = ytData[ytData['Grade'] == 'A- ']['Video views']
trace4 = bar_chart(lables = feats_counts.index, values = feats_counts)

feats_counts = ytData[ytData['Grade'] == 'B+ ']['Video views']
trace5 = bar_chart(lables = feats_counts.index, values = feats_counts)

fig = tools.make_subplots(rows=3, cols=2, vertical_spacing=0.06, 
                          subplot_titles=["A++ : # of Video views","A+ : # of Video Upviewsloads",
                                          "A+ : # of Video views","A- : # of Video views",
                                          "B+ : # of Video views",])
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig.append_trace(trace3, 2, 1)
fig.append_trace(trace4, 2, 2)
fig.append_trace(trace5, 3, 1)

fig['layout'].update(height=1200, width=700, paper_bgcolor='rgb(233,233,233)', title="# of Video views for youtube channels")

py.iplot(fig, filename='plots_3')
def bar_chart(lables, values):
    trace = go.Bar(
        x=lables,
        y=values,
        showlegend=False,
        marker=dict(
            color='rgba(28,32,56,0.84)',
        )
    )
    return trace

feats_counts = (ytData[ytData['Grade'] == 'A++ ']['Subscribers'])
trace1 = bar_chart(lables = feats_counts.index, values = feats_counts)

feats_counts = ytData[ytData['Grade'] == 'A+ ']['Subscribers']
trace2 = bar_chart(lables = feats_counts.index, values = feats_counts)

feats_counts = ytData[ytData['Grade'] == 'A ']['Subscribers']
trace3 = bar_chart(lables = feats_counts.index, values = feats_counts)

feats_counts = ytData[ytData['Grade'] == 'A- ']['Subscribers']
trace4 = bar_chart(lables = feats_counts.index, values = feats_counts)

feats_counts = ytData[ytData['Grade'] == 'B+ ']['Subscribers']
trace5 = bar_chart(lables = feats_counts.index, values = feats_counts)

fig = tools.make_subplots(rows=3, cols=2, vertical_spacing=0.06, 
                          subplot_titles=["A++ : # of subscribers","A+ : # of subscribers",
                                          "A+ : # of subscribers","A- : # of subscribers",
                                          "B+ : # of subscribers",])
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig.append_trace(trace3, 2, 1)
fig.append_trace(trace4, 2, 2)
fig.append_trace(trace5, 3, 1)

fig['layout'].update(height=1200, width=700, paper_bgcolor='rgb(233,233,233)', title="# of subscribers for top 100 youtube channels")

py.iplot(fig, filename='plots_1')
def bar_chart(lables, values):
    trace = go.Bar(
        x=lables,
        y=values,
        showlegend=False,
        marker=dict(
            color='rgba(28,32,56,0.84)',
        )
    )
    return trace

feats_counts = (ytData[ytData['Grade'] == 'A++ ']['Video Uploads'])
trace1 = bar_chart(lables = feats_counts.index, values = feats_counts)

feats_counts = ytData[ytData['Grade'] == 'A+ ']['Video Uploads']
trace2 = bar_chart(lables = feats_counts.index, values = feats_counts)

feats_counts = ytData[ytData['Grade'] == 'A ']['Video Uploads']
trace3 = bar_chart(lables = feats_counts.index, values = feats_counts)

feats_counts = ytData[ytData['Grade'] == 'A- ']['Video Uploads']
trace4 = bar_chart(lables = feats_counts.index, values = feats_counts)

feats_counts = ytData[ytData['Grade'] == 'B+ ']['Video Uploads']
trace5 = bar_chart(lables = feats_counts.index, values = feats_counts)

fig = tools.make_subplots(rows=3, cols=2, vertical_spacing=0.06, 
                          subplot_titles=["A++ : # of Video Uploads","A+ : # of Video Uploads",
                                          "A+ : # of Video Uploads","A- : # of Video Uploads",
                                          "B+ : # of Video Uploads",])
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig.append_trace(trace3, 2, 1)
fig.append_trace(trace4, 2, 2)
fig.append_trace(trace5, 3, 1)

fig['layout'].update(height=1200, width=700, paper_bgcolor='rgb(233,233,233)', title="# of Video Uploads youtube channels")

py.iplot(fig, filename='plots_3')
ytData.loc[ytData['Video Uploads'] == '--', 'Video Uploads'] = 0
ytData.loc[ytData['Subscribers'] == '-- ', 'Subscribers'] = 0
ytData['vidUploadsPerSub'] = ytData['Video Uploads'].astype(np.int64) / ytData['Subscribers'].astype(np.int64)
ytData['vidViewsPerSub'] = ytData['Video views'].astype(np.int64) / ytData['Subscribers'].astype(np.int64)
ytData.info()
def bar_chart(lables, values):
    trace = go.Bar(
        x=lables,
        y=values,
        showlegend=False,
        marker=dict(
            color='rgba(28,32,56,0.84)',
        )
    )
    return trace

feats_counts = round((ytData[ytData['Grade'] == 'A++ ']['vidViewsPerSub']), 0)
trace1 = bar_chart(lables = feats_counts.index, values = feats_counts)

feats_counts = round(ytData[ytData['Grade'] == 'A+ ']['vidViewsPerSub'] ,0)
trace2 = bar_chart(lables = feats_counts.index, values = feats_counts)

feats_counts = round(ytData[ytData['Grade'] == 'A ']['vidViewsPerSub'], 0)
trace3 = bar_chart(lables = feats_counts.index, values = feats_counts)

feats_counts = round(ytData[ytData['Grade'] == 'A- ']['vidViewsPerSub'], 0)
trace4 = bar_chart(lables = feats_counts.index, values = feats_counts)

feats_counts = round(ytData[ytData['Grade'] == 'B+ ']['vidViewsPerSub'], 0)
trace5 = bar_chart(lables = feats_counts.index, values = feats_counts)

fig = tools.make_subplots(rows=3, cols=2, vertical_spacing=0.06, 
                          subplot_titles=["A++ : Avg # of VidViews per subs.","A+ : Avg # of VidViews per subs.",
                                          "A : Avg # of VidViews per subs.","A- : Avg # of VidViews per subs.",
                                          "B+ : Avg # of VidViews per subs.",])
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig.append_trace(trace3, 2, 1)
fig.append_trace(trace4, 2, 2)
fig.append_trace(trace5, 3, 1)

fig['layout'].update(height=1200, width=700, paper_bgcolor='rgb(233,233,233)', title="# of Video Uploads youtube channels")

py.iplot(fig, filename='plots_2')

