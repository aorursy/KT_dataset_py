import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)import os

from pathlib import Path

import warnings

warnings.filterwarnings('ignore')



import matplotlib.pyplot as plt

import plotly.graph_objects as go

from plotly.subplots import make_subplots



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
class config:

    PATH = Path("/kaggle/input/riiid-test-answer-prediction")

    

    dtype = {'row_id': 'int64', 

             'timestamp': 'int64', 

             'user_id': 'int32', 

             'content_id': 'int16', 

             'content_type_id': 'int8',

             'task_container_id': 'int16',

             'user_answer': 'int8', 

             'answered_correctly': 'int8', 

             'prior_question_elapsed_time': 'float32', 

             'prior_question_had_explanation': 'boolean',

             }

    

    LINE_WIDTH = 1
submission = pd.read_csv(config.PATH/"example_sample_submission.csv")

submission.head()
train_df = pd.read_csv(config.PATH/'train.csv', low_memory=False, nrows=10**5, 

                       dtype=config.dtype

                      )

train_df
train_df.describe()
train_df.info()
f"Number of Unique students in {train_df.shape[0]} samples are {train_df['user_id'].nunique()}"
def apply_plot_layout(fig, feature, annot="", annot_size=60, y_title="", title="", tickangle=-90, unified=True):

    fig.update_layout(

        hovermode='x unified' if unified else 'x',

        title=title,

        xaxis= {"tickangle":tickangle,

                "showgrid":False,

                "showline":False,

                "gridwidth":.1,

                "zeroline":False,

                },

        yaxis= {"showline":False,

                "gridcolor":'rgba(203, 210, 211,.3)',

                "gridwidth":.1,

                "zeroline":False,

                "title":y_title

                },

        #xaxis_title="Toggle the legends to show/hide corresponding curve",

        plot_bgcolor='#ffffff',

        paper_bgcolor='#ffffff',

    ) 

    return fig
feature = 'user_id'

agg_fun = 'count'

df = train_df.groupby([feature])[feature].agg([agg_fun]).reset_index()



trace1 = go.Bar(x=df[feature].astype(str) + "-",

                y=df[agg_fun],

                hovertext=['{} : {},\n{} : {:,d}'.format(feature, id,

                            agg_fun, c) for id, c in zip(df[feature], df[agg_fun])],

                hovertemplate='%{hovertext}' +

                            '<extra></extra>'

)



fig = go.Figure(data=[trace1])

fig = apply_plot_layout(fig, feature=feature, annot_size=60, y_title=agg_fun, title="Number of interactions by each user")

fig.show() 
feature = 'content_id'

agg_fun = 'count'

df = train_df.groupby([feature])[feature].agg([agg_fun]).reset_index()



trace1 = go.Bar(x=df[feature].astype(str) + "-",

                y=df[agg_fun],

                hovertext=['{} : {},\n{} : {:,d}'.format(feature, id,

                            agg_fun, c) for id, c in zip(df[feature], df[agg_fun])],

                hovertemplate='%{hovertext}' +

                            '<extra></extra>'

)



fig = go.Figure(data=[trace1])

fig = apply_plot_layout(fig, feature=feature, annot_size=60, y_title=agg_fun, title="Number of interactions for each content")

fig.show() 
labels = ["solved question", 'watched lecture']

values = train_df.content_type_id.value_counts().values



fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent',

                             insidetextorientation='radial'

                            )])



fig.update_layout(

    title_text="Questions vs Lectures",

)

fig.show()
feature = 'task_container_id'

agg_fun = 'count'

df = train_df.groupby([feature])[feature].agg([agg_fun]).reset_index()



trace1 = go.Bar(x=df[feature].astype(str) + "-",

                y=df[agg_fun],

                hovertext=['{} : {},\n{} : {:,d}'.format("Batch id", id,

                            agg_fun, c) for id, c in zip(df[feature], df[agg_fun])],

                hovertemplate='%{hovertext}' +

                            '<extra></extra>'

)



fig = go.Figure(data=[trace1])

fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6)



fig = apply_plot_layout(fig, feature=feature, annot_size=60, y_title=agg_fun, title="Number of interactions per batch")

fig.show() 
feature = 'user_answer'

agg_fun = 'count'

df = train_df.groupby([feature])[feature].agg([agg_fun]).reset_index()



trace1 = go.Bar(x=df[feature],

                y=df[agg_fun],

                hovertext=['{} : {:,d},\n{} : {:,d}'.format("User's answer", id,

                            agg_fun, c) for id, c in zip(df[feature], df[agg_fun])],

                hovertemplate='%{hovertext}' +

                            '<extra></extra>'

)



fig = go.Figure(data=[trace1])

fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6)



fig = apply_plot_layout(fig, feature=feature, annot_size=60, y_title=agg_fun, title="User's answer to the question", tickangle=0, unified=False)

fig.show() 
feature = 'answered_correctly'

agg_fun = 'count'

df = train_df[train_df[feature] != -1].groupby([feature])[feature].agg([agg_fun]).reset_index()



is_correct = {-1: "Watching lecture", 0:'Wrong', 1:'Correct'}



trace1 = go.Bar(x=df[feature],

                y=df[agg_fun],

                hovertext=['{} answer,\n{} : {:,d}'.format(is_correct[id],

                            agg_fun, c) for id, c in zip(df[feature], df[agg_fun])],

                hovertemplate='%{hovertext}' +

                            '<extra></extra>'

)



fig = go.Figure(data=[trace1])



fig = apply_plot_layout(fig, feature=feature, annot_size=60, y_title=agg_fun, title="Answered correctly ?", tickangle=0, unified=False)

fig.show() 
feature = 'user_answer'

agg_fun = 'count'

wrongly_answered = train_df.query('user_answer != -1 and answered_correctly == 0')

df = wrongly_answered.groupby([feature])[feature].agg([agg_fun]).reset_index()



trace1 = go.Bar(x=df[feature],

                y=df[agg_fun],

                name="Wrong",

                hovertext=['{} : {:,d},\n{} : {:,d}'.format("User's answer", id,

                            agg_fun, c) for id, c in zip(df[feature], df[agg_fun])],

                hovertemplate='%{hovertext}' +

                            '<extra></extra>'

)



correctly_answered = train_df.query('user_answer != -1 and answered_correctly == 1')

df = correctly_answered.groupby([feature])[feature].agg([agg_fun]).reset_index()



trace2 = go.Bar(x=df[feature],

                y=df[agg_fun],

                name="Correct",

                hovertext=['{} : {:,d},\n{} : {:,d}'.format("User's answer", id,

                            agg_fun, c) for id, c in zip(df[feature], df[agg_fun])],

                hovertemplate='%{hovertext}' +

                            '<extra></extra>'

)



fig = go.Figure(data=[trace2, trace1])

#fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',

#                  marker_line_width=1.5, opacity=0.6)



fig = apply_plot_layout(fig, feature=feature, annot_size=60, y_title=agg_fun, title="Correct vs Wrong proportion for each User's answer", tickangle=0, unified=False)

fig.show() 
feature = 'user_id'

agg_fun = 'unique'

df = train_df.groupby(['user_id'])['timestamp', 'task_container_id', 'prior_question_elapsed_time'].agg([agg_fun]).reset_index()



all_traces = []

TOP = None

for i in range(len(df)):

    user_id = str(df.iloc[i]['user_id'].values[0])

    x = df.iloc[i]['timestamp']['unique']

    y = df.iloc[i]['prior_question_elapsed_time']['unique']

    trace = go.Scatter(x=x,

                    y=y,

                    text=x,

                    name=f'user {user_id}',

                    mode='markers+lines',

                    line={"width":config.LINE_WIDTH, "shape":'spline'},

                    hovertext=['user {} took time : {}'.format(user_id, tk) for tk in y],

                    hovertemplate='%{hovertext}' +

                                '<extra></extra>'



    )

    all_traces.append(trace)

    if TOP and i > TOP:

        break

    

fig = go.Figure(data=all_traces)

fig.update_layout(

    

    hovermode='x unified',

    title='Average time taken for solving prior question bundle',

    xaxis= {

                #"tickangle":tickangle,

                "showgrid":False,

                "showline":False,

                "gridwidth":.1,

                "zeroline":False,

                "title":"Timestamp"

                },

    yaxis= {    "showline":False, #linecolor='#272e3e',

                "gridcolor":'rgba(203, 210, 211,.3)',

                "gridwidth":.1,

                "zeroline":False,

                "title":'prior question elapsed time'

                },

    #xaxis_title="Toggle the legends to show/hide corresponding curve",

    plot_bgcolor='#ffffff',

    paper_bgcolor='#ffffff',

) 



fig.show() 
df = train_df[train_df['answered_correctly'] != -1].prior_question_had_explanation.value_counts().reset_index()



trace1 = go.Bar(x=df["index"],

                y=df.prior_question_had_explanation,

                hovertext=['{} : {:,d}'.format(id, c) for id, c in zip(df["index"], df.prior_question_had_explanation)],

                hovertemplate='%{hovertext}' +

                            '<extra></extra>'

)



fig = go.Figure(data=[trace1])

#fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',

#                  marker_line_width=1.5, opacity=0.6)



fig = apply_plot_layout(fig, feature='prior_question_had_explanation', annot_size=60, y_title='Count', 

                        title="Seen explanation for prior question (ignoring lectures)?", tickangle=0,

                        unified = False)

fig.show() 