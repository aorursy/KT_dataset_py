import numpy as np
import pandas as pd

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

from kaggle.competitions import twosigmanews

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

env = twosigmanews.make_env()
mt_df, nt_df = env.get_training_data()
mt_df.head()
print("{:,} market samples.".format(mt_df.shape[0]))
mt_df.dtypes
mt_df.isna().sum()
mt_df['price_diff'] = mt_df['close'] - mt_df['open']
grouped = mt_df.groupby('time').agg({'price_diff': ['std', 'min']}).reset_index()
print(f"Average standard deviation of price change within a day in {grouped['price_diff']['std'].mean():.4f}.")
print("There are {:,} unique assets in the training set".format(mt_df['assetCode'].nunique()))
print("There are {} missing values in the `assetCode` column".format(mt_df['time'].isna().sum()))
volumesByTradingDay = mt_df.groupby(mt_df['time'].dt.date)['volume'].sum()
trace1 = go.Bar(
    x = volumesByTradingDay.index,
    y = volumesByTradingDay.values
)

layout = dict(title = "Trading volumes by date",
              xaxis = dict(title = 'Year'),
              yaxis = dict(title = 'Volume'),
              )
data = [trace1]

py.iplot(dict(data=data, layout=layout), filename='basic-line')
mt_df['open'].describe()
mt_df['returnsOpenNextMktres10'].describe()
outliers = mt_df[(mt_df['returnsOpenNextMktres10'] > 1) |  (mt_df['returnsOpenNextMktres10'] < -1)]
outliers['returnsOpenNextMktres10'].describe()
woOutliers = mt_df[(mt_df['returnsOpenNextMktres10'] < 1) &  (mt_df['returnsOpenNextMktres10'] > -1)]
woOutliers['returnsOpenNextMktres10'].describe()
trace1 = go.Histogram(
    x = woOutliers.sample(n=10000)['returnsOpenNextMktres10'].values
)

layout = dict(title = "returnsOpenNextMktres10 (random 10.000 sample; without outliers)")
data = [trace1]

py.iplot(dict(data=data, layout=layout), filename='basic-line')
nt_df.head()
print(f'{nt_df.shape[0]} samples and {nt_df.shape[1]} features in the training news dataset.')
import matplotlib.pyplot as plt 
nt_df['sentence_word_count'] =  nt_df['wordCount'] / nt_df['sentenceCount']
plt.boxplot(nt_df['sentence_word_count'][nt_df['sentence_word_count'] < 40]);
(nt_df['headlineTag'].value_counts() / 1000)[:10].plot('barh');
plt.title('headlineTag counts (thousands)');