import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
%matplotlib inline
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
# First let us load the datasets into different Dataframes
train_df = pd.read_csv('../input/train.csv')

# Dimensions
print('Train shape:', train_df.shape)
# Set of features we have are: date, store, and item
display(train_df.sample(10))
trgt_cnts = train_df.groupby(['target']).target.count()
trgt_cnts = trgt_cnts.to_frame()
trgt_cnts.columns=['counts']
trgt_cnts.head(20)
x_labels = ['sincere' if i==0 else 'insencere' for i in trgt_cnts.index]
data = [go.Bar(x=x_labels, y=trgt_cnts.counts)]
layout = go.Layout(title='Sincere vs Insincere Question distribution',
    xaxis=dict(title='Sincere and Insincere questions'),
    yaxis=dict(title='Question Counts'))
iplot(go.Figure(data=data, layout=layout))
tmp_df = train_df.copy(deep=True)
tmp_df['q_length'] = tmp_df.question_text.apply(lambda qt: len(qt))
tmp_df.sort_values(['q_length'], inplace=True)
print('Min question length', tmp_df.q_length.min())
print('Max question length', tmp_df.q_length.max())
print('Avg question length', tmp_df.q_length.mean())
counts = tmp_df.groupby(['q_length'])['q_length'].count()
cln = counts.to_frame()
cln.columns=['counts']
#cln.head()
data = [go.Bar(x=cln.index, y=cln.counts)]
layout = go.Layout(title='Question length distribution',
    xaxis=dict(title='Question length'),
    yaxis=dict(title='Question Counts'))
iplot(go.Figure(data=data, layout=layout))
tmp_df['w_length'] = tmp_df.question_text.apply(lambda qt: len(qt.split()))
tmp_df.sort_values(['w_length'], inplace=True)
print('Min words', tmp_df.w_length.min())
print('Max words', tmp_df.w_length.max())
print('Avg words', tmp_df.w_length.mean())
word_counts = tmp_df.groupby(['w_length'])['w_length'].count()
cln = word_counts.to_frame()
cln.columns=['counts']
#cln.head()
data = [go.Bar(x=cln.index, y=cln.counts)]
layout = go.Layout(title='Number of words distribution',
    xaxis=dict(title='Number of words'),
    yaxis=dict(title='Question Counts'))
iplot(go.Figure(data=data, layout=layout))