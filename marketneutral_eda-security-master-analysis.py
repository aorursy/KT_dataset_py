import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
import cufflinks as cf
init_notebook_mode(connected=False)
cf.set_config_file(offline=True, world_readable=True, theme='polar')

plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = 12, 7
# Make environment and get data
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()
(market_train_df, news_train_df) = env.get_training_data()
df = market_train_df
df['has_data'] = df.volume.notnull().astype('int')
lifetimes_df = df.groupby(
        by='assetCode'
    ).agg(
        {'time': [np.min, np.max],
         'has_data': 'sum'
        }
)
lifetimes_df.columns = lifetimes_df.columns.droplevel()
lifetimes_df.rename(columns={'sum': 'has_data_sum'}, inplace=True)
lifetimes_df['days_alive'] = np.busday_count(
    lifetimes_df.amin.values.astype('datetime64[D]'),
    lifetimes_df.amax.values.astype('datetime64[D]')
)
#plt.hist(lifetimes_df.days_alive.astype('int'), bins=25);
#plt.title('Histogram of Asset Lifetimes (business days)');
data = [go.Histogram(x=lifetimes_df.days_alive.astype('int'))]
layout = dict(title='Histogram of Asset Lifetimes (business days)',
              xaxis=dict(title='Business Days'),
              yaxis=dict(title='Asset Count')
             )
fig = dict(data = data, layout = layout)
iplot(fig)
lifetimes_df['alive_no_data'] = np.maximum(lifetimes_df['days_alive'] - lifetimes_df['has_data_sum'],0)
lifetimes_df.sort_values('alive_no_data', ascending=False ).head(10)

df.set_index('time').query('assetCode=="VNDA.O"').returnsOpenNextMktres10.iplot(kind='scatter',mode='markers', title='VNDA.O');
#plt.hist(lifetimes_df['alive_no_data'], bins=25);
#plt.ylabel('Count of Assets');
#plt.xlabel('Count of missing days');
#plt.title('Missing Days in Asset Lifetime Spans');

data = [go.Histogram(x=lifetimes_df['alive_no_data'])]
layout = dict(title='Missing Days in Asset Lifetime Spans',
              xaxis=dict(title='Count of missing days'),
              yaxis=dict(title='Asset Count')
             )
fig = dict(data = data, layout = layout)
iplot(fig)
df.groupby('assetName')['assetCode'].nunique().sort_values(ascending=False).head(20)
df[df.assetName=='T-Mobile US Inc'].assetCode.unique()
lifetimes_df.loc[['PCS.N', 'TMUS.N', 'TMUS.O']]
(1+df[df.assetName=='T-Mobile US Inc'].set_index('time').returnsClosePrevRaw1).cumprod().plot(title='Time joined cumulative return');
news_train_df[news_train_df.assetName=='T-Mobile US Inc'].T
df.groupby('assetCode')['assetName'].nunique().sort_values(ascending=False).head(20)
df[df.assetName=='Alphabet Inc'].assetCode.unique()

lifetimes_df.loc[['GOOG.O', 'GOOGL.O']]
df = market_train_df.reset_index().sort_values(['assetCode', 'time']).set_index(['assetCode','time'])
grp = df.groupby('assetCode')
df['volume_avg20'] = (
    grp.apply(lambda x: x.volume.rolling(20).mean())
    .reset_index(0, drop=True)
)
(df.reset_index().set_index('time')
 .query('assetCode=="VNDA.O"').loc['2007-03-15':'2009-06', ['volume', 'volume_avg20']]
)
df = df.reset_index().sort_values(['assetCode', 'time']).reset_index(drop=True)
df['volume_avg20d'] = (df
    .groupby('assetCode')
    .rolling('20D', on='time')     # Note the 'D' and on='time'
    .volume
    .mean()
    .reset_index(drop=True)
)
df.reset_index().set_index('time').query('assetCode=="VNDA.O"').loc['2007-03-15':'2009-06', ['volume', 'volume_avg20', 'volume_avg20d']]
