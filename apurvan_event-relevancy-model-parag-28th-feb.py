# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import datetime

import lightgbm as lgb

from scipy import stats

from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split

from wordcloud import WordCloud

from collections import Counter

from nltk.corpus import stopwords

from nltk.util import ngrams

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import StandardScaler

stop = set(stopwords.words('english'))





import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



from xgboost import XGBClassifier

import lightgbm as lgb

from sklearn import model_selection

from sklearn.metrics import accuracy_score

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import multiprocessing



# Any results you write to the current directory are saved as output.
from kaggle.competitions import twosigmanews

env = twosigmanews.make_env()

print('Done!')



cpu_count = 2*multiprocessing.cpu_count()-1

print('Number of CPUs: {}'.format(cpu_count))
(market_train_df, news_train_df) = env.get_training_data()
print("Size of Stock Data: ",market_train_df.shape)

print("Size of News Data: ",news_train_df.shape)
market_train_df.head(3)
news_train_df.head(3)
print(news_train_df.groupby(['headlineTag']).assetName.value_counts().sort_values(ascending = False))
# See the different news sources

news_train_df['provider'].value_counts().head(10)
# Get count headline tags ( type of news) in the news data set

(news_train_df['headlineTag'].value_counts() / 1000)[:10].plot('barh');

plt.title('headlineTag counts (thousands)');
dfsent = pd.DataFrame(news_train_df.groupby('assetName').sentimentClass.mean()).reset_index(level=['assetName'])

dfsent['sentimentClass'] = dfsent['sentimentClass']*10

dfsent[:5]
from matplotlib import pyplot as plt

ax = plt.subplot(111)

news_train_df[news_train_df['assetName']=='Naugatuck Valley Financial Corp'][:10].sentimentClass.plot(ax = ax, kind='bar',title='Sentiment Score for Naugatuck Valley Financial Corp', edgecolor = "black", color=(0, 0.8, 0,1))

#ax.get_xaxis().set_visible(False)

ax.set_xlabel('Time the news came out')

ax.set_xticklabels(news_train_df[news_train_df['assetName'] == 'Naugatuck Valley Financial Corp'][:10].sourceTimestamp, rotation=-30, ha = 'left')

plt.show()

dfsent1 = pd.concat([dfsent[dfsent['sentimentClass']>8].sample(4),dfsent[dfsent['sentimentClass']<-8].sample(5),dfsent[dfsent['sentimentClass']<2].sample(6)], axis = 0)

dfsent1.sort_values(by = 'sentimentClass', ascending = False, inplace = True)

dfsent1.reset_index(inplace = True)

dfsent1
for i, j in zip([-1, 0, 1], ['negative', 'neutral', 'positive']):

    df_sentiment = news_train_df.loc[news_train_df['sentimentClass'] == i, 'assetName']

    print(f'Top mentioned companies for {j} sentiment are:')

    print(df_sentiment.value_counts().head(5))

    print('')
# plot histogram of sentiment by company

import matplotlib.pyplot as plt

import numpy as np

%matplotlib inline



ax = plt.subplot(111)

k = 15

dfsent1.sentimentClass.plot(ax=ax, kind='barh',title='AVERAGE SENTIMENT SCORE', edgecolor = "black", color=(0, 0.8, 1,1))

xmin, xmax = -11, 11

ymin, ymax = -1, 15

X = [[.7, .5], [.7, .5]]



ax.imshow(X,interpolation='bicubic', cmap=plt.cm.Reds, extent=(xmin, xmax, ymin, ymax), alpha=1)

#ax.set_xlim(-1, 1)

ax.get_yaxis().set_visible(False)

for i, x in enumerate(dfsent[:k].assetName):

    ax.text(-12.2, i-0.3, x, ha='right', fontsize='large')
print("Number of Unique Companies in the News Dataset: ",len(news_train_df['assetName'].unique()))

print("Number of Unique Companies in the Stock Prices Dataset: ",len(market_train_df['assetCode'].unique()))
news_train_df['time'] = pd.to_datetime(news_train_df['time'])

news_train_df['time'].describe()
# select some Pharma companies and plot their stock trends

pharmalist = ['ABT.N', 'PFE.N', 'MRK.N','MDT.N','NVO.N','TEVA.N','CELG.N']

dfpharma = market_train_df[market_train_df['assetCode'].isin(pharmalist)]

dfpharma = dfpharma[~(dfpharma['assetName'] == 'Unknown')]

dfpharma.shape
dfpharma[:2]
market_train_df[:2]
# See Stock price trends for selected Pharma Companies

market_train_df['time'] = pd.to_datetime(market_train_df['time'], errors='coerce')

data = []

for asset in dfpharma.assetName.unique():

    asset_df = market_train_df[(market_train_df['assetName'] == asset)]



    data.append(go.Scatter(

        x = asset_df['time'].dt.strftime(date_format='%Y-%m-%d').values,

        y = asset_df['close'].values,

        name = asset

    ))

layout = go.Layout(dict(title = "Closing prices of 10 random assets",

                  xaxis = dict(title = 'Month'),

                  yaxis = dict(title = 'Price (USD)'),

                  ),legend=dict(

                orientation="h"))

py.iplot(dict(data=data, layout=layout), filename='basic-line')
market_train_df['close_to_open'] =  np.abs(market_train_df['close'] / market_train_df['open'])

market_train_df['assetName_mean_open'] = market_train_df.groupby('assetName')['open'].transform('mean')

market_train_df['assetName_mean_close'] = market_train_df.groupby('assetName')['close'].transform('mean')



# if open price is too far from mean open price for this company, replace it. Otherwise replace close price.

for i, row in market_train_df.loc[market_train_df['close_to_open'] >= 2].iterrows():

    if np.abs(row['assetName_mean_open'] - row['open']) > np.abs(row['assetName_mean_close'] - row['close']):

        market_train_df.iloc[i,5] = row['assetName_mean_open']

    else:

        market_train_df.iloc[i,4] = row['assetName_mean_close']

        

for i, row in market_train_df.loc[market_train_df['close_to_open'] <= 0.5].iterrows():

    if np.abs(row['assetName_mean_open'] - row['open']) > np.abs(row['assetName_mean_close'] - row['close']):

        market_train_df.iloc[i,5] = row['assetName_mean_open']

    else:

        market_train_df.iloc[i,4] = row['assetName_mean_close']





market_train_df['price_diff'] = market_train_df['close'] - market_train_df['open']

grouped = market_train_df.groupby(['time']).agg({'price_diff': ['std', 'min']}).reset_index()

g = grouped.sort_values(('price_diff', 'std'), ascending=False)[:10]

g['min_text'] = 'Maximum price drop: ' + (-1 * np.round(g['price_diff']['min'], 2)).astype(str)

trace = go.Scatter(

    x = g['time'].dt.strftime(date_format='%Y-%m-%d').values,

    y = g['price_diff']['std'].values,

    mode='markers',

    marker=dict(

        size = g['price_diff']['std'].values * 5,

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
# code mostly takes from this kernel: https://www.kaggle.com/ashishpatel26/bird-eye-view-of-two-sigma-xgb

# do a bit of feature engineering and merge market and news data

def data_prep(market_df,news_df):

    market_df['time'] = market_df.time.dt.date

    market_df['returnsOpenPrevRaw1_to_volume'] = market_df['returnsOpenPrevRaw1'] / market_df['volume']

    market_df['close_to_open'] = market_df['close'] / market_df['open']

    market_df['volume_to_mean'] = market_df['volume'] / market_df['volume'].mean()

    news_df['sentence_word_count'] =  news_df['wordCount'] / news_df['sentenceCount']

    news_df['time'] = news_df.time.dt.hour

    news_df['sourceTimestamp']= news_df.sourceTimestamp.dt.hour

    news_df['firstCreated'] = news_df.firstCreated.dt.date

    news_df['assetCodesLen'] = news_df['assetCodes'].map(lambda x: len(eval(x)))

    news_df['assetCodes'] = news_df['assetCodes'].map(lambda x: list(eval(x))[0])

    news_df['headlineLen'] = news_df['headline'].apply(lambda x: len(x))

    news_df['assetCodesLen'] = news_df['assetCodes'].apply(lambda x: len(x))

    news_df['asset_sentiment_count'] = news_df.groupby(['assetName', 'sentimentClass'])['time'].transform('count')

    news_df['asset_sentence_mean'] = news_df.groupby(['assetName', 'sentenceCount'])['time'].transform('mean')

    lbl = {k: v for v, k in enumerate(news_df['headlineTag'].unique())}

    news_df['headlineTagT'] = news_df['headlineTag'].map(lbl)

    kcol = ['firstCreated', 'assetCodes']

    news_df = news_df.groupby(kcol, as_index=False).mean()



    market_df = pd.merge(market_df, news_df, how='left', left_on=['time', 'assetCode'], 

                            right_on=['firstCreated', 'assetCodes'])



    lbl = {k: v for v, k in enumerate(market_df['assetCode'].unique())}

    market_df['assetCodeT'] = market_df['assetCode'].map(lbl)

    

    market_df = market_df.dropna(axis=0)

    

    return market_df



# market_train.drop(['price_diff', 'assetName_mean_open', 'assetName_mean_close'], axis=1, inplace=True)

market_train = data_prep(market_train_df, news_train_df)



print(market_train.shape)

up = market_train.returnsOpenNextMktres10 >= 0



fcol = [c for c in market_train.columns if c not in ['assetCode', 'assetCodes', 'assetCodesLen', 'assetName', 'assetCodeT',

                                             'firstCreated', 'headline', 'headlineTag', 'marketCommentary', 'provider',

                                             'returnsOpenNextMktres10', 'sourceId', 'subjects', 'time', 'time_x', 'universe','sourceTimestamp']]



X = market_train[fcol].values

up = up.values

r = market_train.returnsOpenNextMktres10.values



# Scaling of X values

mins = np.min(X, axis=0)

maxs = np.max(X, axis=0)

rng = maxs - mins

X = 1 - ((maxs - X) / rng)
np.bincount(up)
from sklearn import model_selection

X_train, X_test, up_train, up_test, r_train, r_test = model_selection.train_test_split(X, up, r, test_size=0.1, random_state=99)



import lightgbm as lgb

# xgb_up = XGBClassifier(n_jobs=4,

#                        n_estimators=300,

#                        max_depth=3,

#                        eta=0.15,

#                        random_state=42)

params = {'learning_rate': 0.01, 'max_depth': 12, 'boosting': 'gbdt', 'objective': 'binary'

          , 'metric': 'auc', 'is_training_metric': True, 'seed': 42}

model = lgb.train(params, train_set=lgb.Dataset(X_train, label=up_train), num_boost_round=20,

                  valid_sets=[lgb.Dataset(X_train, label=up_train), lgb.Dataset(X_test, label=up_test)],

                  early_stopping_rounds=100)
df = pd.DataFrame({'imp': model.feature_importance(), 'col':fcol})

df.sort_values(by = ['imp'], ascending = False,inplace = True)

df
def generate_color():

    color = '#{:02x}{:02x}{:02x}'.format(*map(lambda x: np.random.randint(0, 255), range(3)))

    return color



df = pd.DataFrame({'imp': model.feature_importance(), 'col':fcol})

df = df.sort_values(['imp','col'], ascending=[True, False])

data = [df]

for dd in data:  

    colors = []

    for i in range(len(dd)):

         colors.append(generate_color())



    data = [

        go.Bar(

        orientation = 'h',

        x=dd.imp,

        y=dd.col,

        name='Features',

        textfont=dict(size=20),

            marker=dict(

            color= colors,

            line=dict(

                color='#000000',

                width=0.5

            ),

            opacity = 0.87

        )

    )

    ]

    layout= go.Layout(

        title= 'Feature Importance of LGB',

        xaxis= dict(title='Columns', ticklen=5, zeroline=False, gridwidth=2),

        yaxis=dict(title='Value Count', ticklen=5, gridwidth=2),

        showlegend=True

    )



    py.iplot(dict(data=data,layout=layout), filename='horizontal-bar')