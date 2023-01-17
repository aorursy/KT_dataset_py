import pandas as pd

import math

import plotly.express as px

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
df_fnc = pd.read_csv("/kaggle/input/trends-assessment-prediction/fnc.csv")

df_loading = pd.read_csv("/kaggle/input/trends-assessment-prediction/loading.csv")

df_train_scores = pd.read_csv("/kaggle/input/trends-assessment-prediction/train_scores.csv")
df_fnc.head()
print("fnc dataset total row number: {0} \nfnc dataset Total Col Number: {1}".format(df_fnc.shape[0], df_fnc.shape[1]))
df_loading.head()
print("loading dataset total row number: {0} \nloading dataset Total Col Number: {1}".format(df_loading.shape[0], df_loading.shape[1]))
df_fnc.info()
df_loading.describe().T
f,ax = plt.subplots(figsize=(10,10))

sns.heatmap(df_loading.iloc[:,1:].corr(),annot=True, 

            linewidths=.5, fmt='.1f', ax=ax, cbar=0

           )



plt.show()
df_train_scores.head()
print(df_train_scores.loc[:, df_train_scores.isnull().any()].isnull().sum())
df_train_scores.describe().T
fig = px.histogram(x=df_train_scores["age"],

                   title='Distribution of Age',

                   opacity=0.8,

                   color_discrete_sequence=['darkorange'],

                   nbins=30 )



fig.update_layout(

    yaxis_title_text='',

    xaxis_title_text='',

    height=450, width=600)



fig.update_traces(marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.8

                 )



fig.show()
fig = make_subplots(rows=2, cols=2, subplot_titles=('domain1_var1', 'domain1_var2','domain2_var1', 'domain2_var2'))



fig.add_trace(go.Histogram(x=df_train_scores["domain1_var1"],

                      marker_color='#FF9999',

                      opacity=0.2,

                      nbinsx=50),

    row=1, col=1)



fig.add_trace(go.Histogram(x=df_train_scores["domain1_var2"],

                      marker_color='#FF9999',

                      opacity=0.2,

                      nbinsx=40),

    row=1, col=2)



fig.add_trace(go.Histogram(x=df_train_scores["domain2_var1"],

                      marker_color='#FF9999',

                      opacity=0.2,

                      nbinsx=40),

    row=2, col=1)



fig.add_trace(go.Histogram(x=df_train_scores["domain2_var2"],

                      marker_color='#FF9999',

                      opacity=0.2,

                      nbinsx=40),

    row=2, col=2)





fig.update_layout(

    height=500, width=800, 

    showlegend=False,

    title="Distribution of domain_vars"

)





fig.update_traces(marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.8

                 )



fig.show()
loading_cols = df_loading.columns[1:]



fig = make_subplots(rows=13, cols=2, 

                    subplot_titles=loading_cols

                   )



for i, col in enumerate(loading_cols):

    fig.add_trace(go.Histogram(x=df_loading[col],

                      marker_color='rosybrown',

                      opacity=0.2,

                      nbinsx=50),

    row=math.ceil((i+1)/2), col=(i%2)+1)



fig.update_layout(

    height=2500, width=800, showlegend=False,

    title="Distribution of ICs"

)





fig.update_traces(marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.8

                 )



fig.show()    