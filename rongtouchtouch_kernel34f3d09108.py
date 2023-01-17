# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from kaggle.competitions import twosigmanews

# get training data
env = twosigmanews.make_env()
(market_train_df, news_train_df) = env.get_training_data()
market_train_df.describe()
market_train_df.head()
market_train_df.tail()
news_train_df.describe()
news_train_df.head()
news_train_df.tail()
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
%matplotlib inline

data = []
for asset in np.random.choice(market_train_df['assetName'].unique(), 10):
    asset_df = market_train_df[(market_train_df['assetName'] == asset)]

    data.append(go.Scatter(
        x = asset_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
        y = asset_df['open'].values,
        name = asset
    ))
layout = go.Layout(dict(title = "Closing prices of 10 random assets",
                  xaxis = dict(title = 'Month'),
                  yaxis = dict(title = 'Price (USD)'),
                  ),legend=dict(
                orientation="h"))
py.iplot(dict(data=data, layout=layout), filename='basic-line')
data = []
for i in [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]:
    price_df = market_train_df.groupby('time')['close'].quantile(i).reset_index()

    data.append(go.Scatter(
        x = price_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
        y = price_df['close'].values,
        name = f'{i} quantile'
    ))
layout = go.Layout(dict(title = "Trends of closing prices by quantiles",
                  xaxis = dict(title = 'Month'),
                  yaxis = dict(title = 'Price (USD)'),
                  ),legend=dict(
                orientation="h"),
    annotations=[
        dict(
            x='2008-09-01 22:00:00+0000',
            y=82,
            xref='x',
            yref='y',
            text='Collapse of Lehman Brothers',
            showarrow=True,
            font=dict(
                family='Courier New, monospace',
                size=16,
                color='#ffffff'
            ),
            align='center',
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='#636363',
            ax=20,
            ay=-30,
            bordercolor='#c7c7c7',
            borderwidth=2,
            borderpad=4,
            bgcolor='#ff7f0e',
            opacity=0.8
        ),
        dict(
            x='2011-08-01 22:00:00+0000',
            y=85,
            xref='x',
            yref='y',
            text='Black Monday',
            showarrow=True,
            font=dict(
                family='Courier New, monospace',
                size=16,
                color='#ffffff'
            ),
            align='center',
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='#636363',
            ax=20,
            ay=-30,
            bordercolor='#c7c7c7',
            borderwidth=2,
            borderpad=4,
            bgcolor='#ff7f0e',
            opacity=0.8
        ),
        dict(
            x='2014-10-01 22:00:00+0000',
            y=120,
            xref='x',
            yref='y',
            text='Another crisis',
            showarrow=True,
            font=dict(
                family='Courier New, monospace',
                size=16,
                color='#ffffff'
            ),
            align='center',
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='#636363',
            ax=-20,
            ay=-30,
            bordercolor='#c7c7c7',
            borderwidth=2,
            borderpad=4,
            bgcolor='#ff7f0e',
            opacity=0.8
        ),
        dict(
            x='2016-01-01 22:00:00+0000',
            y=120,
            xref='x',
            yref='y',
            text='Oil prices crash',
            showarrow=True,
            font=dict(
                family='Courier New, monospace',
                size=16,
                color='#ffffff'
            ),
            align='center',
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='#636363',
            ax=20,
            ay=-30,
            bordercolor='#c7c7c7',
            borderwidth=2,
            borderpad=4,
            bgcolor='#ff7f0e',
            opacity=0.8
        )
    ])
py.iplot(dict(data=data, layout=layout), filename='basic-line')
import matplotlib.pyplot as plt

def plot_random_asset(market):
    """
    Get random asset, show price, volatility and volume
    """
    # Get any asset
    ass = market_train_df.assetCode.sample(1, random_state=24).iloc[0]
    ass_market = market_train_df[market_train_df['assetCode'] == ass]
    ass_market.index = ass_market.time

    # Plotting
    f, axs = plt.subplots(3,1, sharex=True, figsize=(12,8))
    # Close price 
    ass_market.close.plot(ax=axs[0])
    axs[0].set_ylabel("Price")

    # Volatility (close-open)
    volat_df = (ass_market.close - ass_market.open)
    (ass_market.close - ass_market.open).plot(color='green', ax = axs[1])
    axs[1].set_ylabel("Volatility")

    # Volume
    ass_market.volume.plot(ax=axs[2], color='darkred')
    axs[2].set_ylabel("Volume")

    # Show the plot
    f.suptitle("Asset: %s" % ass, fontsize=22)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.show()

plot_random_asset(market_train_df)
market_train_df['price_diff'] = market_train_df['close'] - market_train_df['open']
grouped = market_train_df.groupby('time').agg({'price_diff': ['std', 'min']}).reset_index()
g = grouped.sort_values(('price_diff', 'std'), ascending=False)[:10]
g['min_text'] = 'Maximum price drop: ' + (-1 * g['price_diff']['min']).astype(str)
trace = go.Scatter(
    x = g['time'].dt.strftime(date_format='%Y-%m-%d').values,
    y = g['price_diff']['std'].values,
    mode='markers',
    marker=dict(
        size = g['price_diff']['std'].values,
        color = g['price_diff']['std'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = g['min_text'].values
    #text = f"Maximum price drop: {g['price_diff']['min'].values}"
    #g['time'].dt.strftime(date_format='%Y-%m-%d').values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Top 10 months by standard deviation of price change within a day',
    hovermode= 'closest',
    yaxis=dict(
        title= 'price_diff',
        ticklen= 5,
        gridwidth= 2,
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')
market_train_df.sort_values('price_diff')[:10]
# sorted by time
market_train_orig = market_train_df.sort_values('time')
news_train_orig = news_train_df.sort_values('time')
market_train_df = market_train_orig.copy()
news_train_df = news_train_orig.copy()
del market_train_orig
del news_train_orig
import datetime

market_train_df = market_train_df.loc[market_train_df['time'].dt.date>=datetime.date(2009,1,1)]
news_train_df = news_train_df.loc[news_train_df['time'].dt.date>=datetime.date(2009,1,1)]
print('Check null data:')
market_train_df.isna().sum()
column_market = ['returnsClosePrevMktres1','returnsOpenPrevMktres1','returnsClosePrevMktres10', 'returnsOpenPrevMktres10']
column_raw = ['returnsClosePrevRaw1', 'returnsOpenPrevRaw1','returnsClosePrevRaw10', 'returnsOpenPrevRaw10']
#for i in range(len(column_raw)):
#    market_train_df[column_market[i]] = market_train_df[column_market[i]].fillna(market_train_df[column_raw[i]])
# 这个地方raw应该是不能直接移到mktres中的 之后有空考虑用随机森林填数据
market_train_df['close_open_ratio'] = np.abs(market_train_df['close']/market_train_df['open'])
threshold = 0.5
print('In %i lines price increases by 50%% or more in a day' %(market_train_df['close_open_ratio']>=1+threshold).sum())
print('In %i lines price decreases by 50%% or more in a day' %(market_train_df['close_open_ratio']<=1-+threshold).sum())
market_train_df = market_train_df.loc[market_train_df['close_open_ratio'] < 1.5]
market_train_df = market_train_df.loc[market_train_df['close_open_ratio'] > 0.5]
market_train_df = market_train_df.drop(columns=['close_open_ratio'])
print('Removing strange data ...')
orig_len = market_train_df.shape[0]
market_train_df = market_train_df[~market_train_df['assetCode'].isin(['PGN.N','EBRYY.OB'])]
#market_train_df = market_train_df[~market_train_df['assetName'].isin(['Unknown'])]
new_len = market_train_df.shape[0]
rmv_len = np.abs(orig_len-new_len)
print('There were %i lines removed' %rmv_len)
print('Removing outliers ...')
column_return = column_market + column_raw + ['returnsOpenNextMktres10']
orig_len = market_train_df.shape[0]
for column in column_return:
    market_train_df = market_train_df.loc[market_train_df[column]>=-2]
    market_train_df = market_train_df.loc[market_train_df[column]<=2]
new_len = market_train_df.shape[0]
rmv_len = np.abs(orig_len-new_len)
print('There were %i lines removed' %rmv_len)
print('Check null data:')
news_train_df.isna().sum()
from wordcloud import WordCloud 
import matplotlib.pyplot as plt

text = ' '.join(news_train_df['headline'].str.lower().values[-1000000:])
wordcloud = WordCloud(max_font_size=None,stopwords=None, background_color='white',
                      width=1200, height=1000).generate(text)
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud)
plt.title('Top words in headline')
plt.axis("off")
plt.show()
def remove_outliers(data_frame, column_list, low=0.02, high=0.98):
    for column in column_list:
        this_column = data_frame[column]
        quant_df = this_column.quantile([low,high])
        low_limit = quant_df[low]
        high_limit = quant_df[high]
        data_frame[column] = data_frame[column].clip(lower=low_limit, upper=high_limit)
    return data_frame
columns_outlier = ['takeSequence', 'bodySize', 'sentenceCount', 'wordCount', 'sentimentWordCount', 'firstMentionSentence','noveltyCount12H',\
                  'noveltyCount24H', 'noveltyCount3D', 'noveltyCount5D', 'noveltyCount7D', 'volumeCounts12H', 'volumeCounts24H',\
                  'volumeCounts3D','volumeCounts5D','volumeCounts7D']
print('Clipping news outliers ...')
news_train_df = remove_outliers(news_train_df, columns_outlier)

print(f'{news_train_df.shape[0]} samples and {news_train_df.shape[1]} features in the training news dataset.')
import matplotlib.pyplot as plt
import seaborn as sns

columns_corr_market = ['volume', 'open', 'close','returnsClosePrevRaw1','returnsOpenPrevRaw1',\
           'returnsClosePrevMktres1','returnsOpenPrevMktres1','returnsClosePrevMktres10','returnsOpenPrevRaw10',\
           'returnsClosePrevMktres10', 'returnsOpenPrevMktres10', 'returnsOpenNextMktres10']
colormap = plt.cm.RdBu
plt.figure(figsize=(18,15))
sns.heatmap(market_train_df[columns_corr_market].astype(float).corr(), linewidths=0.1, vmax=1.0, vmin=-1., square=True, cmap=colormap, linecolor='white', annot=True)
plt.title('Pair-wise correlation')
# Plot correlation
columns_corr = ['urgency', 'takeSequence', 'companyCount','marketCommentary','sentenceCount',\
           'firstMentionSentence','relevance','sentimentClass','sentimentWordCount','noveltyCount24H',\
           'noveltyCount3D', 'noveltyCount5D', 'noveltyCount7D','volumeCounts24H','volumeCounts3D','volumeCounts5D','volumeCounts7D']
colormap = plt.cm.RdBu
plt.figure(figsize=(18,15))
sns.heatmap(news_train_df[columns_corr].astype(float).corr(), linewidths=0.1, vmax=1.0, vmin=-1., square=True, cmap=colormap, linecolor='white', annot=True)
plt.title('Pair-wise correlation')

"""
g = sns.pairplot(market_train_df[[u'volume', 'open', 'close','returnsClosePrevRaw1','returnsOpenPrevRaw1',\
           'returnsClosePrevMktres1','returnsOpenPrevMktres1','returnsClosePrevMktres10','returnsOpenPrevRaw10',\
           'returnsClosePrevMktres10', 'returnsOpenPrevMktres10', 'returnsOpenNextMktres10']], hue='open', palette = 'seismic',size=1.2,diag_kind = 'kde',diag_kws=dict(shade=True),plot_kws=dict(s=10) )
g.set(xticklabels=[])
"""

"""
asset_code_dict = {k: v for v, k in enumerate(market_train_df['assetCode'].unique())}
drop_columns = [col for col in news_train_df.columns if col not in ['sourceTimestamp', 'urgency', 'takeSequence', 'bodySize', 'companyCount', 
               'sentenceCount', 'firstMentionSentence', 'relevance','firstCreated', 'assetCodes']]
columns_news = ['firstCreated','relevance','sentimentClass','sentimentNegative','sentimentNeutral',
               'sentimentPositive','noveltyCount24H','noveltyCount7D','volumeCounts24H','volumeCounts7D','assetCodes','sourceTimestamp',
               'assetName','audiences', 'urgency', 'takeSequence', 'bodySize', 'companyCount', 
               'sentenceCount', 'firstMentionSentence','time']
"""
"""
# Data processing function
def data_prep(market_df,news_df):
    market_df['date'] = market_df.time.dt.date
    market_df['close_to_open'] = market_df['close'] / market_df['open']
    market_df.drop(['time'], axis=1, inplace=True)
    
    news_df = news_df[columns_news]
    news_df['sourceTimestamp']= news_df.sourceTimestamp.dt.hour
    news_df['firstCreated'] = news_df.firstCreated.dt.date
    news_df['assetCodesLen'] = news_df['assetCodes'].map(lambda x: len(eval(x)))
    news_df['assetCodes'] = news_df['assetCodes'].map(lambda x: list(eval(x))[0])
    news_df['asset_sentiment_count'] = news_df.groupby(['assetName', 'sentimentClass'])['time'].transform('count')
    news_df['len_audiences'] = news_train_df['audiences'].map(lambda x: len(eval(x)))
    kcol = ['firstCreated', 'assetCodes']
    news_df = news_df.groupby(kcol, as_index=False).mean()
    market_df = pd.merge(market_df, news_df, how='left', left_on=['date', 'assetCode'], 
                            right_on=['firstCreated', 'assetCodes'])
    del news_df
    market_df['assetCodeT'] = market_df['assetCode'].map(asset_code_dict)
    market_df = market_df.drop(columns = ['firstCreated','assetCodes','assetName']).fillna(0) 
    return market_df

print('Merging Data ...')
market_train_df = data_prep(market_train_df,news_train_df)
market_train_df.head()
    
"""
print(len(market_train_df.assetName.unique().categories))
print(market_train_df.count())

# How many total records and assets are in the data
print("Total count: %d records of %d assets", len(market_train_df.assetName.unique().categories), market_train_df.count())
# Common libs
import pandas as pd
import numpy as np
import sys
import os
import os.path
import random
from pathlib import Path
from time import time
from itertools import chain
import gc

# Image processing
import imageio
import skimage
import skimage.io
import skimage.transform
#from skimage.transform import rescale, resize, downscale_local_mean

# Charts
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

# ML
import scipy
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
#from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
#from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer,StandardScaler, MinMaxScaler,OneHotEncoder, LabelEncoder, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
#from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.sequence import TimeseriesGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, BatchNormalization, LSTM, Embedding
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
import tensorflow
def learning_rate_power(current_round):
    base_learning_rate = 0.19000424246380565
    min_learning_rate = 0.01
    lr = base_learning_rate * np.power(0.995,current_round)
    return max(lr, min_learning_rate)
class MarketPrepro:
    # features
    assetcode_encoded = []
    time_cols = ['year', 'week', 'day', 'dayofweek']
    numeric_cols = ['volume', 'close', 'open', 'returnsClosePrevRaw1', 'returnsOpenPrevRaw1',
                    'returnsClosePrevMktres1', 'returnsOpenPrevMktres1', 'returnsClosePrevRaw10',
                    'returnsOpenPrevRaw10', 'returnsClosePrevMktres10', 'returnsOpenPrevMktres10']
    feature_cols = ['assetCode_encoded'] + time_cols + numeric_cols

    # labels
    label_cols = ['returnsOpenNextMktres10']

    def __init__(self):
        self.cats = {}
        self.numeric_scaler = StandardScaler()

    def fit(self, market_train_idx, market):
        """
        Fit preprocessing scalers, encoders on given train df.
        Store given indices to generate batches_from.
        @param market_train_df: train data to fit on
        """
        market_train_df = market.loc[market_train_idx.index].copy()
        # Clean bad data. We fit on train dataset and it's ok to remove bad data
        market_train_df = self.fix_train(market_train_df)

        # Extract day, week, year from time
        market_train_df = self.prepare_time_cols(market_train_df)
        # Fit for numeric and time
        # self.numeric_scaler = QuantileTransformer()
        self.numeric_scaler.fit(market_train_df[self.numeric_cols + self.time_cols])

        # Fit asset encoding
        self.encode_asset(market_train_df, is_train=True)

    def fix_train(self, train_df):
        """
        Remove bad data. For train dataset only
        """
        # Remove strange cases with close/open ratio > 2
        max_ratio = 2
        train_df = train_df[(train_df['close'] / train_df['open']).abs() <= max_ratio].loc[:]
        # Fix outliers etc like for test set
        train_df = self.safe_fix(train_df)
        return train_df

    def safe_fix(self, df):
        """
        Fill na, fix outliers. Safe for test dataset, no rows removed.
        """
        # Fill nans
        df[self.numeric_cols] = df[self.numeric_cols].fillna(0)
        # Fix outliers
        df[self.numeric_cols] = df[self.numeric_cols].clip(df[self.numeric_cols].quantile(0.01),
                                                           df[self.numeric_cols].quantile(0.99), axis=1)
        return df

    def get_X(self, df):
        """
        Preprocess and return X without y
        """
        df = df.copy()
        df = self.safe_fix(df)

        # Add day, week, year
        df = self.prepare_time_cols(df)
        # Encode assetCode
        df = self.encode_asset(df)
        # Scale numeric features and labels

        df = df.set_index(['assetCode', 'time'], drop=False)
        df[self.numeric_cols + self.time_cols] = self.numeric_scaler.transform(
            df[self.numeric_cols + self.time_cols].astype(float))

        # print(df.head())
        # Return X
        return df[self.feature_cols]

    def get_y(self, df, is_raw_y=False):
        if is_raw_y:
            return df[self.label_cols]
        else:
            return (df[self.label_cols] >= 0).astype(float)

    def encode_asset(self, df, is_train=False):
        def encode(assetcode):
            """
            Encode categorical features to numbers
            """
            try:
                # Transform to index of name in stored names list
                index_value = self.assetcode_encoded.index(assetcode) + 1
            except ValueError:
                # If new value, add it to the list and return new index
                self.assetcode_encoded.append(assetcode)
                index_value = len(self.assetcode_encoded)

            # index_value = 1.0/(index_value)
            index_value = index_value / (self.assetcode_train_count + 1)
            return (index_value)

        # Store train assetcode_train_count for use as a delimiter for test data encoding
        if is_train:
            self.assetcode_train_count = len(df['assetCode'].unique()) + 1

        df['assetCode_encoded'] = df['assetCode'].apply(lambda assetcode: encode(assetcode))
        return (df)

    @staticmethod
    def prepare_time_cols(df):
        """
        Extract time parts, they are important for time series
        """
        df['year'] = pd.to_datetime(df['time']).dt.year
        # Maybe remove month because week of year can handle the same info
        df['day'] = pd.to_datetime(df['time']).dt.day
        # Week of year
        df['week'] = pd.to_datetime(df['time']).dt.week
        df['dayofweek'] = pd.to_datetime(df['time']).dt.dayofweek
        return df
class NewsPrepro:
    """
    Aggregate news by day and asset. Normalize numeric values.
    """
    news_cols_numeric = ['urgency', 'takeSequence', 'wordCount', 'sentenceCount', 'companyCount',
                         'marketCommentary', 'relevance', 'sentimentNegative', 'sentimentNeutral',
                         'sentimentPositive', 'sentimentWordCount', 'noveltyCount12H', 'noveltyCount24H',
                         'noveltyCount3D', 'noveltyCount5D', 'noveltyCount7D', 'volumeCounts12H',
                         'volumeCounts24H', 'volumeCounts3D', 'volumeCounts5D', 'volumeCounts7D']

    feature_cols = news_cols_numeric

    def fit(self, idx, news):
        """
        Fit preprocessing scalers, encoders on given train df.
        @param idx: index with time, assetCode
        """
        # Save indices[assetCode, time, news_index] for all news
        self.all_news_idx = self.news_idx(news)

        # Get news only related to market idx
        news_idx = idx.merge(self.all_news_idx, on=['assetCode', 'time'], suffixes=['_idx', ''])[
            ['news_index', 'assetCode', 'time']]
        news_train_df = news_idx.merge(news, left_on='news_index', right_index=True, suffixes=['_idx', ''])[
            self.news_cols_numeric]

        # Numeric data normalization
        self.numeric_scaler = StandardScaler()
        news_train_df.fillna(0, inplace=True)

        # Fit scaler
        self.numeric_scaler.fit(news_train_df)

    def get_X(self, idx, news):
        """
        Preprocess news for asset code and time from given index
        """
        news_idx = idx.merge(self.all_news_idx, on=['assetCode', 'time'], suffixes=['_idx', ''])[
            ['news_index', 'assetCode', 'time']]
        news_df = news_idx.merge(news, left_on='news_index', right_index=True, suffixes=['_idx', ''])[
            ['time', 'assetCode'] + self.news_cols_numeric]
        news_df = self.aggregate_news(news_df)

        return self.safe_fix(news_df)

    def safe_fix(self, news_df):
        """
        Scale, fillna
        """
        # Normalize, fillna etc without removing rows.
        news_df.fillna(0, inplace=True)
        if not news_df.empty:
            news_df[self.news_cols_numeric] = self.numeric_scaler.transform(news_df[self.news_cols_numeric])
        return news_df

    def news_idx(self, news):
        """
        Get asset code, time -> news id
        :param news:
        :return:
        """

        # Fix asset codes (str -> list)
        asset_codes_list = news['assetCodes'].str.findall(f"'([\w\./]+)'")

        # Expand assetCodes
        assetCodes_expanded = list(chain(*asset_codes_list))

        assetCodes_index = news.index.repeat(asset_codes_list.apply(len))
        assert len(assetCodes_index) == len(assetCodes_expanded)
        df_assetCodes = pd.DataFrame({'news_index': assetCodes_index, 'assetCode': assetCodes_expanded})

        # Create expanded news (will repeat every assetCodes' row)
        #        df_expanded = pd.merge(df_assetCodes, news, left_on='level_0', right_index=True)
        df_expanded = pd.merge(df_assetCodes, news[['time']], left_on='news_index', right_index=True)
        # df_expanded = df_expanded[['time', 'assetCode'] + self.news_cols_numeric].groupby(['time', 'assetCode']).mean()

        return df_expanded

    def with_asset_code(self, news):
        """
        Update news index to be time, assetCode
        :param news:
        :return:
        """
        if news.empty:
            if 'assetCode' not in news.columns:
                news.columns = news.columns + 'assetCode'
            return news

        # Fix asset codes (str -> list)
        news['assetCodesList'] = news['assetCodes'].str.findall(f"'([\w\./]+)'")

        # Expand assetCodes
        assetCodes_expanded = list(chain(*news['assetCodesList']))

        assetCodes_index = news.index.repeat(news['assetCodesList'].apply(len))
        assert len(assetCodes_index) == len(assetCodes_expanded)
        df_assetCodes = pd.DataFrame({'level_0': assetCodes_index, 'assetCode': assetCodes_expanded})

        # Create expanded news (will repeat every assetCodes' row)
        #        df_expanded = pd.merge(df_assetCodes, news, left_on='level_0', right_index=True)
        df_expanded = pd.merge(df_assetCodes, news, left_on='level_0', right_index=True)
        df_expanded = df_expanded[['time', 'assetCode'] + self.news_cols_numeric].groupby(['time', 'assetCode']).mean()

        return df_expanded

    def aggregate_news(self, df):
        """
        News are rare for an asset. We get mean value for 10 days
        :param df:
        :return:
        """
        if df.empty:
            return df

        # News are rare for the asset, so aggregate them by rolling period say 10 days
        rolling_days = 10
        df_aggregated = df.groupby(['assetCode', 'time']).mean().reset_index(['assetCode', 'time'])
        df_aggregated = df_aggregated.groupby('assetCode') \
            .rolling(rolling_days, on='time') \
            .apply(np.mean, raw=False) \
            .reset_index('assetCode')
        #df_aggregated.set_index(['time', 'assetCode'], inplace=True)
        return df_aggregated
class JoinedPreprocessor:
    def __init__(self, market_prepro, news_prepro):
        self.market_prepro = market_prepro
        self.news_prepro = news_prepro

    def get_X(self, market, news):
        """
        Returns preprocessed market + news
        :return: X
        """
        # Market row
        market_X = self.market_prepro.get_X(market)
        news_idx = self.news_prepro.news_idx(news)
        news_X = self.news_prepro.get_X(news_idx, news)
        #news_X.time = news_X.time.astype('datetime64')
        # X = market X + news X
        X = market_X.merge(news_X, how='left', on=['time', 'assetCode'], left_index=True)
        X = X.fillna(0)
        X = X[self.market_prepro.feature_cols + self.news_prepro.feature_cols]
        return X

    def get_Xy(self, idx, market, news, is_train=False, is_raw_y=False):
        """
        Returns preprocessed features and labels for given indices
        """
        # Get market data for index
        market_df = market.loc[idx.index]
        # We can remove bad data in train
        if is_train:
            market_df = self.market_prepro.fix_train(market_df)
        market_Xy = self.market_prepro.get_X(market_df)
        # Get news data for index
        news_X = self.news_prepro.get_X(idx, news)
        #news_X.time = pd.to_datetime(news_X.time, utc=True)
        #news_X.time = news_X.time.astype('datetime64')
        # Merge and return
        Xy = market_Xy.merge(news_X, how='left', on=['time', 'assetCode'], left_index=True)
        Xy = Xy.fillna(0)
        X = Xy[self.market_prepro.feature_cols + self.news_prepro.feature_cols]
        y = self.market_prepro.get_y(market_df, is_raw_y)

        return X, y

    def with_look_back(self, X, y, look_back, look_back_step):
        """
        Add look back window values to prepare dataset for LSTM
        """
        look_back_fixed = look_back_step * (look_back // look_back_step)
        # Fill look_back rows before first
        first_xrow = X.values[0]
        first_xrow.shape = [1, X.values.shape[1]]
        first_xrows = np.repeat(first_xrow, look_back_fixed, axis=0)
        X_values = np.append(first_xrows, X.values, axis=0)

        if y is not None:
            first_yrow = y.values[0]
            first_yrow.shape = [1, y.values.shape[1]]
            first_yrows = np.repeat(first_yrow, look_back_fixed, axis=0)
            y_values = np.append(first_yrows, y.values, axis=0)

        # for i in range(0, len(X) - look_back + 1):
        X_processed = []
        y_processed = []
        for i in range(look_back_fixed , len(X_values)):
            # Add lookback to X
            x_window = X_values[i - (look_back_fixed//look_back_step)*look_back_step:i+1:look_back_step, :]
            X_processed.append(x_window)
            # If input is X only, we'll not output y
            if y is None:
                continue
            # Add lookback to y
            y_window = y_values[i - (look_back_fixed//look_back_step)*look_back_step:i+1:look_back_step, :]
            y_processed.append(y_window)
        # Return Xy for train/test or X for prediction
        if y is not None:
            #return np.array(X_processed), np.array(y_processed)
            return np.array(X_processed), y.values
        else:
            return np.array(X_processed)
class JoinedGenerator:
    """
    Keras standard approach to generage batches for model.fit_generator() call.
    """

    def __init__(self, prepro, idx, market, news):
        """
        @param preprocessor: market and news join preprocessor
        @param market: full loaded market df
        @param news: full loaded news df
        @param index_df: df with assetCode and time of train or validation market data. Batches will be taken from them.
        """
        self.market = market
        self.prepro = prepro
        self.news = news
        self.idx = idx

    def flow_lstm(self, batch_size, is_train, look_back, look_back_step):
        """
        Generate batch data for LSTM NN
        Each cycle in a loop we yield a batch for one training step in epoch.
        """
        while True:
            # Get market indices of random assets, sorted by assetCode, time.
            batch_idx = self.get_random_assets_idx(batch_size)

            # Get X, y data for this batch, containing market and news, but without look back yet
            X, y = self.prepro.get_Xy(batch_idx, self.market, self.news, is_train)
            # Add look back data to X, y
            X, y = self.prepro.with_look_back(X, y, look_back, look_back_step)
            yield X, y

    def get_random_assets_idx(self, batch_size):
        """
        Get random asset and it's last market data indices.
        Repeat for next asset until we reach batch_size.
        """
        asset_codes = self.idx['assetCode'].unique().tolist()

        # Insert first asset
        asset = np.random.choice(asset_codes)
        asset_codes.remove(asset)
        #asset = 'ADBE.O'
        batch_index_df = self.idx[self.idx.assetCode == asset].tail(batch_size)

        return batch_index_df.sort_values(by=['assetCode', 'time'])
class ModelFactory:
    # LSTM look back window size
    look_back = 90
    # In windows size look back each look_back_step days
    look_back_step = 10

    def lstm_128(input_size):
        model = Sequential()
        # Add an input layer market + news
        model.add(LSTM(units=128, return_sequences=True, input_shape=(None, input_size)))
        model.add(LSTM(units=64, return_sequences=True))
        model.add(LSTM(units=32, return_sequences=False))

        # Add an output layer
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

        return (model)

    def train(model, toy, join_generator, val_generator):
        weights_file = 'best_weights.h5'

        earlystopper = EarlyStopping(patience=5, verbose=1)

        checkpointer = ModelCheckpoint(weights_file
                                       # ,monitor='val_acc'
                                       , verbose=1
                                       , save_best_only=True
                                       , save_weights_only=True)

        reduce_lr = ReduceLROnPlateau(factor=0.1, patience=2, min_lr=0.001)

        # Set fit parameters
        # Rule of thumb: steps_per_epoch = TotalTrainingSamples / TrainingBatchSize
        #                validation_steps = TotalvalidationSamples / ValidationBatchSize
        if toy:
            batch_size = 1000
            validation_batch_size = 1000
            steps_per_epoch = 5
            validation_steps = 2
            epochs = 5
            look_back = 10
            look_back_step = 2
        else:
            batch_size = 1000
            validation_batch_size = 1000
            steps_per_epoch = 20
            validation_steps = 5
            epochs = 20
            look_back = 90
            look_back_step = 10

        print(f'Toy:{toy}, epochs:{epochs}, steps per epoch: {steps_per_epoch}, validation steps:{validation_steps}')
        print(f'Batch_size:{batch_size}, validation batch size:{validation_batch_size}')

        # Fit
        training = model.fit_generator(join_generator.flow_lstm(batch_size=batch_size
                                                                , is_train=True
                                                                , look_back=look_back
                                                                , look_back_step=look_back_step)
                                       , epochs=epochs
                                       , validation_data=val_generator.flow_lstm(batch_size=validation_batch_size
                                                                                 , is_train=False
                                                                                 , look_back=look_back
                                                                                 , look_back_step=look_back_step)
                                       , steps_per_epoch=steps_per_epoch
                                       , validation_steps=validation_steps
                                       , callbacks=[earlystopper, checkpointer, reduce_lr])
        # Load best weights saved
        model.load_weights(weights_file)
        return training
class TrainValTestSplit:

    @staticmethod
    def train_val_test_split(market, size):
        """
        Get train, validation, test sample indices - time, assetCode, market index in original market df
        @return: train, validation, test df.  Columns - time, assetCode, market_index, news_index
        """
        market_idx = market[['assetCode', 'time']]
        start_date = pd.datetime(2000, 1, 1).date()
        market_idx = market_idx.loc[market_idx.time >= start_date] \
            .sort_values(by=['time', 'assetCode']) \
            .tail(size).copy()

        # Split to train, validation and test
        train_idx, test_idx = train_test_split(market_idx, shuffle=False, random_state=24)
        train_idx, val_idx = train_test_split(train_idx, test_size=0.1, shuffle=False, random_state=24)

        return train_idx, val_idx, test_idx
# general mode
def train_test_val_split(market):
    """
    Get sample of assets but each asset has full market data after 2009
    Split to time sorted train, validation and test.
    @return: train, validation, test df. Short variant - time and asset columns only
    """
    # Work with data after 2009
    market_idx = market[market.time > '2009'][['time', 'assetCode']]
    if toy: market_idx = market_idx.sample(100000)
    else: market_idx = market_idx.sample(1000000)
    # Split to train, validation and test
    market_idx = market_idx.sort_values(by=['time'])
    market_train_idx, market_test_idx = train_test_split(market_idx, shuffle=False, random_state=24)
    market_train_idx, market_val_idx = train_test_split(market_train_idx, test_size=0.1, shuffle=False, random_state=24)
    return(market_train_idx, market_val_idx, market_test_idx)
class Predictor:
    def __init__(self, prepro, market_prepro, news_prepro, model, look_back, look_back_step):
        self.prepro = prepro
        self.market_prepro = market_prepro
        self.news_prepro = news_prepro
        self.model = model
        self.look_back = look_back
        self.look_back_step = look_back_step

    def predict(self, market, news):
        X = self.prepro.get_X(market, news)
        X = self.prepro.with_look_back(X, None, self.look_back, self.look_back_step)
        y = self.model.predict(X) * 2 - 1
        return y

    def predict_idx(self, pred_idx, market, news):
        # Get preprocessed X, y
        X_test, y_test = self.prepro.get_Xy(pred_idx, market, news, is_train=False, is_raw_y=True)
        # look back
        X_test, y_test = self.prepro.with_look_back(X_test, y_test, 
                                                    look_back=self.look_back,
                                                    look_back_step=self.look_back_step)
        y_pred = self.model.predict(X_test) * 2 - 1
        return y_pred, y_test

plt.style.use('seaborn')
# Set random seed to make results reproducable
np.random.seed(42)
tensorflow.set_random_seed(42)
os.environ['PYTHONHASHSEED'] = '42'
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 200)

input_dir = '../input'
print(os.listdir("../input"))
market = pd.read_csv(input_dir + '/marketdata_sample.csv')
news = pd.read_csv(input_dir + '/news_sample.csv')

# !!! Hack
news.time = pd.to_datetime('2007-02-01 23:35')
# Restrict datetime to date
#news.time = pd.to_datetime(news.time.astype('datetime64').dt.date, utc=True)
news.time = news.time.astype('datetime64').dt.date
market.time = market.time.astype('datetime64').dt.date

# Split to train, validation and test
toy = True
if toy:
    sample_size = 10000
else:
    sample_size = 500000
train_idx, val_idx, test_idx = TrainValTestSplit.train_val_test_split(market, sample_size)

# print(market)
# print(news)
# print(train_idx)
plt.style.use('seaborn')
# Set random seed to make results reproducable
np.random.seed(42)
tensorflow.set_random_seed(42)
os.environ['PYTHONHASHSEED'] = '42'
# pd.set_option('display.max_columns', 100)
# pd.set_option('display.width', 200)

market, news = env.get_training_data()

market = pd.DataFrame(market)
news = pd.DataFrame(news)


# Restrict datetime to date
news.time = pd.to_datetime(news.time.astype('datetime64').dt.date)
market.time = pd.to_datetime(market.time.astype('datetime64').dt.date)


# Split to train, validation and test
toy = False
if toy:
    sample_size = 10000
else:
    sample_size = 500000
#train_idx, val_idx, test_idx = TrainValTestSplit.train_val_test_split(market, sample_size)
train_idx, val_idx, test_idx = train_test_val_split(market)

# print(market)
# print(news)
# print(train_idx)
print(train_idx.shape)
print(val_idx.shape)
print(test_idx.shape)
# Create preprocessors
market_prepro = MarketPrepro()
market_prepro.fit(train_idx, market)

news_prepro = NewsPrepro()
news_prepro.fit(train_idx, news)

prepro = JoinedPreprocessor(market_prepro, news_prepro)

# prediction_prepro = PredictionPreprocessor(prepro, market_prepro, news_prepro)
# x = prediction_prepro.get_X_with_lookback(market, news, 4,2)


# Train data generator instance
join_generator = JoinedGenerator(prepro, train_idx, market, news)
val_generator = JoinedGenerator(prepro, val_idx, market, news)
print('Generators created')

# Create and train model
model = ModelFactory.lstm_128(len(market_prepro.feature_cols) + len(news_prepro.feature_cols))
# model.load_weights("best_weights.h5")

print(model.summary())
training = ModelFactory.train(model, toy, join_generator, val_generator)


"""
# Predict

y_pred, y_test = predictor.predict_idx(test_idx, market, news)

y_pred = predictor.predict(market, news)

plt.plot(y_pred)
plt.plot(y_test)
plt.legend(["pred", "test"])
plt.show()
"""
# # Plotting
# f, axs = plt.subplots(3,1, sharex=True, figsize=(12,8))
# # Close price 
# ass_market.close.plot(ax=axs[0])
# axs[0].set_ylabel("Price")

plt.figure(1, figsize=(8,3))
plt.subplot(121)
plt.plot(training.history['loss'])
plt.plot(training.history['val_loss'])
plt.title("Loss and validation loss")
plt.legend(["Loss", "Validation loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.subplot(122)
plt.plot(training.history['acc'])
plt.plot(training.history['val_acc'])
plt.title("Acc and validation acc")
plt.legend(["Acc", "Validation acc"])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.suptitle('Training history', fontsize=16)
plt.show()
def predict_on_test():
    # Predict on last test data
    pred_size=1000
    pred_idx = test_idx.tail(pred_size + ModelFactory.look_back)
    y_pred, y_test = predictor.predict_idx(pred_idx, market, news)
    #market_df = market.loc[pred_idx.index]
    #y_test = market_df['returnsOpenNextMktres10'].values
    # Plot
    ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
    ax1.plot(y_test, linestyle='none', marker='.', color='darkblue')
    ax1.plot(y_pred, linestyle='none', marker='.', color='darkorange')
    ax1.legend(["Ground truth","Predicted"])
    ax1.set_title("Both")
    ax1.set_xlabel("Epoch")
    ax2 = plt.subplot2grid((2, 2), (0, 1), colspan=1,rowspan=1)
    ax2.plot(y_test, linestyle='none', marker='.', color='darkblue')
    ax2.set_title("Ground truth")
    ax3 = plt.subplot2grid((2, 2), (1, 1), colspan=1,rowspan=1)
    ax3.plot(y_pred, linestyle='none', marker='.', color='darkorange')
    ax3.set_title("Predicted")
    plt.tight_layout()
    plt.show()

predictor = Predictor( prepro, market_prepro, news_prepro, model, ModelFactory.look_back, ModelFactory.look_back_step)  
predict_on_test()
def predict_random_asset():
    """
    Get random asset from test set, predict on it, plot ground truth and predicted value
    """
    # Get any asset
    asset = test_idx['assetCode'].sample(1, random_state=66).values[0]
    pred_idx = test_idx[test_idx.assetCode == asset]
    y_pred, y_test = predictor.predict_idx(pred_idx, market, news)
    # Plot
    plt.plot(y_test, linestyle='none', marker='.', color='darkblue')
    plt.plot(y_pred, linestyle='none', marker='.', color='darkorange')
    plt.xticks(rotation=45)
    plt.title(asset)
    plt.legend(["Ground truth", "predicted"])
    plt.show()
    
predict_random_asset()
def get_score():
    """
    Calculation of actual metric that is used to calculate final score
    @param r: returnsOpenNextMktres10
    @param u: universe
    where rti is the 10-day market-adjusted leading return for day t for instrument i, and uti is a 0/1 universe variable (see the data description for details) that controls whether a particular asset is included in scoring on a particular day.    
    """
    # Get test sample to calculate score on
    pred_idx = test_idx #.sample(10000, random_state=24)
    y_pred, y_test = predictor.predict_idx(pred_idx, market, news)    
    look_back=ModelFactory.look_back
    market_df = market.loc[pred_idx.index]
    r=market_df['returnsOpenNextMktres10'].values#.values[look_back:]
    u=market_df['universe'].values#.values[look_back:]
    confidence=y_pred
    # calculation of actual metric that is used to calculate final score
    r = r.clip(-1,1) # get rid of outliers. Where do they come from??
    x_t_i = confidence.reshape(r.shape) * r * u

    #print(x_t_i.iloc[0])
    d = (market_df['time'].dt.day).values #[look_back:]
    data = {'day' : d, 'x_t_i' : x_t_i}
    df = pd.DataFrame(data)
    x_t = df.groupby('day').sum().values.flatten()
    mean = np.mean(x_t)
    std = np.std(x_t)
    score = mean / std
    return score
    
print(f"Sigma score: {get_score()}") 
def calc_acc():
    # Get X_test, y_test with look back for LSTM
    pred_idx = test_idx.sample(10000)
    y_pred, y_test = predictor.predict_idx(pred_idx, market, news)
    
    #y_pred = pd.DataFrame(market_prepro.y_scaler.inverse_transform(model.predict(X_test)))
    print("Accuracy: %f" % accuracy_score(y_test >= 0, y_pred >= 0))
    #score = get_score(market_df, confidence, market_df.returnsOpenNextMktres10, market_df.universe)
    print('Predictions size: ', len(y_pred))
    print('y_test size:', len(y_test))
     # Show distribution of confidence that will be used as submission
    plt.hist(y_test, bins='auto', alpha=0.3)
    plt.hist(y_pred, bins='auto', alpha=0.3, color='darkorange')
    plt.legend(['Ground truth', 'Predicted'])
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.title("predicted confidence")
    plt.show()

# Call accuracy calculation and plot    
calc_acc()
# market['time']
def make_predictions(market_obs_df, news_obs_df, predictions_template_df):
    """
    Predict confidence for one day and update predictions_template_df['confidenceValue']
    @param market_obs_df: market_obs_df returned from env
    @param predictions_template_df: predictions_template_df returned from env.
    @return: None. prediction_template_df updated instead. 
    """
    # Predict
    market_obs_df = pd.DataFrame(market_obs_df)
    news_obs_df = pd.DataFrame(news_obs_df)
    
    # Restrict datetime to date
    news_obs_df['time'] = news_obs_df['time'].astype('datetime64')
    market_obs_df['time'] = market_obs_df['time'].astype('datetime64')
    
    y_pred = predictor.predict(market_obs_df, news_obs_df)
    confidence_df=pd.DataFrame(y_pred, columns=['confidence'])

    # Merge predicted confidence to predictions template
    pred_df = pd.concat([predictions_template_df, confidence_df], axis=1).fillna(0)
    predictions_template_df.confidenceValue = pred_df.confidence
##########################
# Submission code

# Save data here for later debugging on it
days_saved_data = []

# Store execution info for plotting later
predicted_days=[]
predicted_times=[]
last_predictions_template_df = None

# Predict day by day
days = env.get_prediction_days()
# market['time'].astype('datetime64')
last_year=None
for (market_obs_df, news_obs_df, predictions_template_df) in days:
    # Store the data for later debugging on it
    days_saved_data.append((market_obs_df, news_obs_df, predictions_template_df))
   
    # For later plotting
    predicted_days.append(market_obs_df.iloc[0].time.strftime('%Y-%m-%d'))
    time_start = time()
    # For logging
    cur_year = market_obs_df.iloc[0].time.strftime('%Y')
    if cur_year != last_year:
        print(f'Predicting {cur_year}...')
        last_year = cur_year
    # Call prediction func
    make_predictions(market_obs_df, news_obs_df, predictions_template_df)
    #!!!
    env.predict(predictions_template_df)
    
    # For later plotting
    last_predictions_template_df = predictions_template_df
    predicted_times.append(time()-time_start)
    #print("Prediction completed for ", predicted_days[-1])
    
print(f"Prediction for {len(predicted_days)} days completed")
# Plot execution time 
sns.barplot(np.array(predicted_days), np.array(predicted_times))
plt.title("Execution time per day")
plt.xlabel("Day")
plt.ylabel("Execution time, seconds")
plt.show()

# Plot predicted confidence for last day
last_predictions_template_df.plot(linestyle='none', marker='.', color='darkorange')
plt.title("Predicted confidence for last observed day: %s" % predicted_days[-1])
plt.xlabel("Observation No.")
plt.ylabel("Confidence")
plt.show()
env.write_submission_file()
print([filename for filename in os.listdir('.') if '.csv' in filename])

