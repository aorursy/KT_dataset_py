from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()
(market_train_df, news_train_df) = env.get_training_data()
(market_train_df, news_train_df) = env.get_training_data()
import pandas as pd
import numpy as np
from matplotlib import pyplot
import seaborn as sns
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
from sklearn import svm

sns.set(rc={'figure.figsize':(14,10.27)}, font_scale=1.5)
current_palette = sns.color_palette()
market_correlation = market_train_df.corr()
colors = [[0, 'white'], [0.5, 'grey'], [1.0, 'black']]
data = [go.Heatmap( z=market_correlation.values.tolist(), x=market_correlation.columns, y=market_correlation.columns, colorscale=colors)]
py.iplot(data)
top_tech_companies = ['AMZN.O', 'CSCO.O', 'INTC.O', 'MSFT.O', 'FB.O', 'BABA.N', 'GOOG.O', 'GOOGL.O']
years = market_train_df.time.dt.year.unique()
data = market_train_df[market_train_df.assetCode.str.contains('|'.join(top_tech_companies))]
data = data[~data.assetCode.str.contains('MAFB.O')]
plot_data = [go.Scatter(x=years, y=data[data['assetCode'].isin([a_code])]['volume'].tolist(), name=a_code) for a_code in top_tech_companies]

py.iplot(plot_data)