import pandas as pd

import numpy as np

from gensim.parsing.preprocessing import stem_text, strip_multiple_whitespaces, strip_punctuation

import plotly.graph_objects as go
combined_news_dija_df = pd.read_csv('../input/stocknews/Combined_News_DJIA.csv')

djia_table_df = pd.read_csv('../input/stocknews/DJIA_table.csv')

reddit_news_df = pd.read_csv('../input/stocknews/RedditNews.csv')
def preprocess_news_headline_text(text):

    if type(text) is np.int64 or type(text) is int or pd.isna(text):

        return text

#     print(type(text), text)

    text = stem_text(strip_multiple_whitespaces(strip_punctuation(text)))

    if text[:2] == 'b ':

        text = text[2:]

    return text
combined_news_dija_df
combined_news_dija_df[combined_news_dija_df.isna().any(axis=1)]
combined_news_dija_df.set_index('Date', inplace=True)

combined_news_dija_df = combined_news_dija_df.applymap(preprocess_news_headline_text)
combined_news_dija_df
djia_table_df.set_index('Date', inplace=True)
djia_table_df.index
fig = go.Figure(data=[go.Candlestick(x=djia_table_df.index,

                open=djia_table_df['Open'],

                high=djia_table_df['High'],

                low=djia_table_df['Low'],

                close=djia_table_df['Close'])])



fig.show()

djia_table_df