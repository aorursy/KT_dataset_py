import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly as py

import plotly.express as px

import plotly.graph_objs as go

from plotly.subplots import make_subplots

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from wordcloud import WordCloud



init_notebook_mode(connected=True) 



import warnings

warnings.filterwarnings('ignore')
df_train = pd.read_csv("../input/data-science-bowl-2019/train.csv", parse_dates=["timestamp"])

df_train_labels = pd.read_csv("../input/data-science-bowl-2019/train_labels.csv")

df_specs = pd.read_csv("../input/data-science-bowl-2019/specs.csv")

df_sample_submission = pd.read_csv("../input/data-science-bowl-2019/sample_submission.csv")

df_test = pd.read_csv("../input/data-science-bowl-2019/test.csv", parse_dates=["timestamp"])
df_train.info()
df_test.info()
print("Train Set Total Row Number: {0} \nTrain Set Total Col Number: {1}".format(df_train.shape[0], df_train.shape[1]))
print("Test Set Total Row Number: {0} \nTest Set Total Col Number: {1}".format(df_test.shape[0], df_test.shape[1]))
df_train.head()
print(df_train.loc[:, df_train.isnull().any()].isnull().sum())
df_train.describe().T
train_types = df_train["type"].value_counts()

test_types = df_test["type"].value_counts()
fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])



fig.add_trace(

    go.Pie(values=train_types, labels=train_types.index.tolist(), name="Train" , hole=.3),

    1, 1)



fig.add_trace(

    go.Pie(values=test_types, labels=test_types.index.tolist(), name="Test" , hole=.3),

    1, 2)



fig.update_traces(hoverinfo='label+percent+value', textinfo='percent', textfont_size=17, textposition="inside",

                  marker=dict(colors=['gold', 'mediumturquoise', 'darkorange', 'plum'],  

                              line=dict(color='#000000', width=2)))



fig.update_layout(

    title_text="Media Type of The Game or Video",

    height=500, width=800,

    annotations=[dict(text='Train', x=0.18, y=0.5, font_size=20, showarrow=False),

                 dict(text='Test', x=0.82, y=0.5, font_size=20, showarrow=False)])



fig.show()
train_worlds = df_train["world"].value_counts()

test_worlds = df_test["world"].value_counts()
fig = make_subplots(rows=1, cols=2, specs=[[{'type':'xy'}, {'type':'xy'}]])



fig.add_trace(

    go.Bar(y=train_worlds.values, x=train_worlds.index),

    row=1, col=1)



fig.add_trace(

    go.Bar(y=test_worlds.values, x=test_worlds.index),

    row=1, col=2)



fig.update_layout(

    title_text="World of Apps",

    height=500, width=800, showlegend=False)



fig['layout']['xaxis1'].update(title='Train')

fig['layout']['xaxis2'].update(title='Test')



fig.show()
eventbyinstallation = df_train.groupby(["installation_id"])["event_code"].nunique()



fig = px.histogram(x=eventbyinstallation,

                   title='Unique Event Code Count by Installation Id',

                   opacity=0.8,

                   color_discrete_sequence=['indianred'])



fig.update_layout(

    yaxis_title_text='',

    xaxis_title_text='',

    height=500, width=800)



fig.update_traces(marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.8

                 )



fig.show()
event_id_by_ins_id_1 = df_train.groupby(["installation_id"])["event_id"].agg("count")[df_train.groupby(["installation_id"])["event_id"].agg("count")<2000]

event_id_by_ins_id_2 = df_train.groupby(["installation_id"])["event_id"].agg("count")[df_train.groupby(["installation_id"])["event_id"].agg("count")>=2000]



fig = make_subplots(rows=1, cols=2)



trace1 = go.Histogram(x=event_id_by_ins_id_1,

                      marker_color='#FF9999',

                      opacity=0.2,

                      nbinsx=40)



trace2 = go.Histogram(x=event_id_by_ins_id_2,

                      marker_color='#9999CC',

                      opacity=0.75,

                      nbinsx=40)



fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 1, 2)





fig.update_layout(

    height=500, width=800, showlegend=False,

    title='Event Count by Installation Id')



fig['layout']['xaxis1'].update(title='Part 1: 0-5k')

fig['layout']['xaxis2'].update(title='Part 2: 5k-60k')



fig.update_traces(marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.8

                 )



fig.show()
df_events = df_train.loc[:,['timestamp', 'event_id','game_time']]

df_events["date"] = df_events['timestamp'].dt.date
event_count = df_events.groupby(['date'])['event_id'].agg('count')

game_time_sum = df_events.groupby(['date'])['game_time'].agg('sum')



fig = go.Figure()



fig.add_trace(go.Scatter(x=event_count.index, y=event_count.values,

                         line=dict(color='firebrick', width=3)))



fig.update_layout(title='Event Counts By Date',

                   xaxis_title='Date',

                   yaxis_title='Count',

                   width=750, height=400)



fig.show()



fig = go.Figure()



fig.add_trace(go.Scatter(x=game_time_sum.index, y=game_time_sum.values,

                         line=dict(color='midnightblue', width=3)))



fig.update_layout(title='Total Game Time By Date',

                   xaxis_title='Date',

                   yaxis_title='Total',

                   width=750, height=400)



fig.show()
df_events["weekdays"] = df_events['timestamp'].dt.weekday_name



gametime_wdays = df_events.groupby(['weekdays'])['game_time'].agg('sum')

gametime_wdays = gametime_wdays.T[['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']]



fig = px.bar(x=gametime_wdays.index, y=gametime_wdays.values)



fig.update_traces(marker_color='mediumvioletred', marker_line_color='rgb(8,48,107)',

                  marker_line_width=2, opacity=0.7)



fig.update_layout(title='Total Game Time By Day',

                   xaxis_title='Weekdays',

                   yaxis_title='Total',

                   width=600, height=400

                 )



fig.show()
title_words = []



for i in df_test["title"]:

    for j in i.split(" "):

        title_words.append(j) 
plt.subplots(figsize=(14,7))

wc=wordcloud = WordCloud(collocations=False,

                          background_color='black',

                          width=512,

                          height=384

                         ).generate(" ".join(title_words)

                                   )



plt.imshow(wc)

plt.axis('off')

plt.title("Frequented Words in Title", fontsize=18)

plt.imshow(wc.recolor(colormap= 'viridis', random_state=2), alpha=0.90)



plt.show()