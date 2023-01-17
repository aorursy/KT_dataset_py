from kaggle.competitions import twosigmanews
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import gc
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(font_scale=1)

import warnings
import missingno as msno

pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 100)
pd.options.mode.chained_assignment = None
# dir(pd.options.display)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

plt.style.use('ggplot')

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
env = twosigmanews.make_env()
(market_train, news_train) = env.get_training_data()
del news_train
gc.enable()
gc.collect()
ret = market_train.returnsOpenNextMktres10
univ = market_train.universe
label = (ret > 0).astype(int)
def ir(label, window):
    global market_train, ret, univ
    time_idx = market_train.time.factorize()[0]
    # (label * 2 - 1) : perfect confidence value
    x_t = (label * 2 - 1) * ret * univ
    x_t_sum = x_t.groupby(time_idx).sum()
    x_t_sum = x_t_sum[window:]
    score = x_t_sum.mean() / x_t_sum.std()
    return score
ir_l = [ir(label, t) for t in range(0, market_train.time.nunique(), 10)]
trace = go.Scatter(
    x = np.arange(0, market_train.time.nunique(), 10),
    y = ir_l,
    mode = 'lines+markers',
    marker = dict(
        size = 4,
        color = 'lightblue'
    ),
    line = dict(
        width = 1
    )
)
data = [trace]
layout = go.Layout(dict(
    title = 'Eval Metric trend',
    xaxis = dict(title = 'operational days passed ( window start point )'),
    yaxis = dict(title = 'Evaluation metric'),
    height = 400,
    width = 750
))
py.iplot(dict(data=data, layout=layout), filename='IR trend')
op = ['mean', 'std']
df = market_train[['time', 'returnsOpenPrevRaw1']].groupby('time').agg({
    'returnsOpenPrevRaw1' : op,
}).reset_index()
df.columns = ['time'] + [o + '_returnsOpenPrevRaw1' for o in op]
trace = go.Scatter(
    x = df.time,
    y = df.std_returnsOpenPrevRaw1,
    mode = 'lines+markers',
    marker = dict(
        size = 4,
        color = 'pink'
    ),
    line = dict(
        width = 1
    )
)
data = [trace]
layout = go.Layout(dict(
    title = 'std of returnsOpenPrevRaw1',
    xaxis = dict(title = 'date'),
    yaxis = dict(title = 'std of returnsOpenPrevRaw1'),
    height = 400,
    width = 750
))
py.iplot(dict(data=data, layout=layout), filename='.')