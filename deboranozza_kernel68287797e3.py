import plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.plotly as py
import plotly.graph_objs as go
init_notebook_mode(connected=True)
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()

# Any results you write to the current directory are saved as output.
(market_train_df, news_train_df) = env.get_training_data()
market_assets_counts = market_train_df["assetName"].value_counts().to_frame()
news_assets_counts = news_train_df["assetName"].value_counts().to_frame()
market_assets_counts.reset_index(level=0, inplace=True)
news_assets_counts.reset_index(level=0, inplace=True)
market_news_assets_counts = market_assets_counts.merge(news_assets_counts, on="index")
market_news_assets_counts["mean_count"] = market_news_assets_counts[['assetName_x', 'assetName_y']].mean(axis=1)
market_news_assets_counts.columns = ["assetName","count_market_presence","count_news_presence","mean_count"]
market_news_assets_counts = market_news_assets_counts.sort_values(["mean_count"], ascending=0)
market_news_assets_counts
market_train_df_apple = market_train_df[market_train_df["assetName"]=="Apple Inc"]
apple_close_chart = go.Scatter(x=market_train_df_apple.time, y=market_train_df_apple["close"], name= 'Apple Close Price')
layout = go.Layout(
title="Apple Close Price"
)
plotly.offline.iplot({"data":[apple_close_chart], "layout": layout})

market_train_df_apple[market_train_df_apple["close"]==93.7]
market_train_df_jpm = market_train_df[market_train_df["assetName"]=="JPMorgan Chase & Co"]
jpm_close_chart = go.Scatter(x=market_train_df_jpm.time, y=market_train_df_jpm["close"], name= 'JPMorgan Close Price',
                               marker = dict(color = 'rgb(10, 200, 30)'))
layout = go.Layout(
title="JPMorgan Close Price"
)
plotly.offline.iplot({"data":[jpm_close_chart], "layout": layout})
news_train_df_apple = news_train_df[news_train_df["assetName"]=="Apple Inc"]
news_train_df_apple["date"] = pd.to_datetime(news_train_df_apple["time"]).apply(lambda x: x.date)
news_train_df_apple_daily_count = news_train_df_apple.groupby("date").count()
news_train_df_apple_daily_mean = news_train_df_apple.groupby("date").mean()
apple_news_chart = go.Scatter(x=news_train_df_apple_daily.index, y=news_train_df_apple_daily_count["time"], name= 'Apple News Volume')
apple_sentiment_chart_neg = go.Scatter(x=news_train_df_apple_daily.index, y=news_train_df_apple_daily_mean["urgency"]*news_train_df_apple_daily_mean["relevance"]*news_train_df_apple_daily_mean["sentimentNegative"]*100, name= 'Apple News Sentiment Neg')
apple_sentiment_chart_pos = go.Scatter(x=news_train_df_apple_daily.index, y=news_train_df_apple_daily_mean["urgency"]*news_train_df_apple_daily_mean["relevance"]*news_train_df_apple_daily_mean["sentimentPositive"]*100, name= 'Apple News Sentiment Pos')

layout = go.Layout(
title="Apple News Volume"
)
plotly.offline.iplot({"data":[apple_close_chart,apple_sentiment_chart_neg,apple_sentiment_chart_pos], "layout": layout})
market_train_df_jpm = market_train_df[market_train_df["assetName"]=="JPMorgan Chase & Co"]
news_train_df_jpm = news_train_df[news_train_df["assetName"]=="JPMorgan Chase & Co"]
news_train_df_jpm["date"] = pd.to_datetime(news_train_df_jpm["time"]).apply(lambda x: x.date)
news_train_df_jpm_daily_count = news_train_df_jpm.groupby("date").count()
news_train_df_jpm_daily_mean = news_train_df_jpm.groupby("date").mean()
jpm_news_chart = go.Scatter(x=news_train_df_jpm_daily_count.index, y=news_train_df_jpm_daily_count["time"], name= 'jpm News Volume')
jpm_sentiment_chart_neg = go.Scatter(x=news_train_df_jpm_daily_mean.index, y=news_train_df_jpm_daily_mean["urgency"]*news_train_df_jpm_daily_mean["relevance"]*news_train_df_jpm_daily_mean["sentimentNegative"]*10, name= 'jpm News Sentiment Neg')
jpm_sentiment_chart_pos = go.Scatter(x=news_train_df_jpm_daily_mean.index, y=news_train_df_jpm_daily_mean["urgency"]*news_train_df_jpm_daily_mean["relevance"]*news_train_df_jpm_daily_mean["sentimentPositive"]*10, name= 'jpm News Sentiment Pos')

layout = go.Layout(
title="jpm News Volume"
)
plotly.offline.iplot({"data":[jpm_close_chart,jpm_sentiment_chart_neg,jpm_sentiment_chart_pos], "layout": layout})
news_train_df_apple = news_train_df[news_train_df["assetName"]=="JPMorgan Chase & Co"]
news_train_df_apple["date"] = pd.to_datetime(news_train_df_apple["time"]).apply(lambda x: x.date)
news_train_df_apple_daily = news_train_df_apple.groupby("date").count()
jpm_news_chart = go.Scatter(x=news_train_df_apple_daily.index, y=news_train_df_apple_daily["time"], name= 'JPMorgan News Volume',
                               marker = dict(color = 'rgb(10, 200, 30)'))
layout = go.Layout(
title="JPMorgan News Volume"
)
plotly.offline.iplot({"data":[jpm_news_chart], "layout": layout})
news_train_df_apple = news_train_df[news_train_df["assetName"]=="Apple Inc"]
news_train_df_apple["date"] = pd.to_datetime(news_train_df_apple["time"]).apply(lambda x: x.date)
news_train_df_apple_daily_count = news_train_df_apple.groupby("date").count()
news_train_df_apple_daily_mean = news_train_df_apple.groupby("date").mean()
apple_news_chart = go.Scatter(x=news_train_df_apple_daily.index, y=news_train_df_apple_daily_count["time"], name= 'Apple News Volume')
apple_sentiment_chart_neg = go.Scatter(x=news_train_df_apple_daily.index, y=news_train_df_apple_daily_mean["sentimentNegative"], name= 'Apple News Sentiment Negative')
apple_sentiment_chart_pos = go.Scatter(x=news_train_df_apple_daily.index, y=news_train_df_apple_daily_mean["sentimentPositive"], name= 'Apple News Sentiment Positive')
apple_sentiment_chart = go.Scatter(x=news_train_df_apple_daily.index, y=news_train_df_apple_daily_mean["sentimentClass"], name= 'Apple News Sentiment Class')
layout = go.Layout(
title="Apple News Volume"
)
plotly.offline.iplot({"data":[apple_sentiment_chart_neg,apple_sentiment_chart_pos, apple_sentiment_chart], "layout": layout})

a.head()
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import datetime
stopwords = set(STOPWORDS)

year = 2014
month = 6
day = 9
for day in range (3,10):
    a = news_train_df_apple[news_train_df_apple["date"]==datetime.date(year, month, day)]

    if (len(a)>0):
        wordcloud = WordCloud(
                                  background_color='white',
                                  stopwords=stopwords,
                                  max_words=15,
                                  max_font_size=40, 
                                  random_state=42
                                 ).generate(str(a['headline']))
        #print(wordcloud)
        fig = plt.figure(1)
        plt.title(str(year)+"-"+str(month)+"-"+str(day), fontdict ={"fontsize":30},pad=30)
        plt.imshow(wordcloud)
        plt.axis('off')
        plt.show()
a