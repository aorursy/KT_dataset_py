# based on the work of
# https://www.kaggle.com/pestipeti/simple-eda-two-sigma
from kaggle.competitions import twosigmanews
# You can only call make_env() once, so don't lose it!
env = twosigmanews.make_env()
import numpy as np
import pandas as pd

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999
mt_df, nt_df = env.get_training_data()
print("{:,} news samples".format(nt_df.shape[0]))
nt_df.dtypes
nt_df.isna().sum()
nt_df.nunique()
urgency = nt_df.groupby(nt_df['urgency'])['urgency'].count().sort_values(ascending=False)
# Create a trace
trace1 = go.Pie(
    labels = urgency.index,
    values = urgency.values
)

layout = dict(title = "Urgency (1:alert, 3:article)")
data = [trace1]

py.iplot(dict(data=data, layout=layout), filename='basic-line')

articlesOnly = nt_df[nt_df['urgency'] == 3]
articleSources = articlesOnly.groupby(articlesOnly['provider'])['provider'].count()
topArticleSources = articleSources.sort_values(ascending=False)[0:10]

# Create a trace
trace1 = go.Pie(
    labels = topArticleSources.index,
    values = topArticleSources.values
)

layout = dict(title = "Top article sources")
data = [trace1]

py.iplot(dict(data=data, layout=layout), filename='basic-line')
alertsOnly = nt_df[nt_df['urgency'] == 1]
alertsSources = alertsOnly.groupby(alertsOnly['provider'])['provider'].count()
alertsSources.sort_values(ascending=False)[0:10]
