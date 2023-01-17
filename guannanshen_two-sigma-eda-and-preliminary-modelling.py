import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from datetime import datetime, timedelta
import lightgbm as lgb 
from scipy import stats
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split

# interactive plot by plotly
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

# for news data
from wordcloud import WordCloud

# loess/lowess 
# from skmisc import loess as skmloess
from scipy.interpolate import interp1d
import statsmodels.api as sm

## try pyGAM

# not fully understand all packages yet
from collections import Counter
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler




from xgboost import XGBClassifier
from sklearn import model_selection
from sklearn.metrics import accuracy_score

# We've got a submission file!
import os
from kaggle.competitions import twosigmanews
# You can only call make_env() once, so don't lose it!
env = twosigmanews.make_env()

# Get the market data
(market_train_df, news_train_df) = env.get_training_data()
market_train_df.head()
# variables Market
print("Within Market data: time is always 22:00 UTC \n assetCode is the unique ID \n assetName is not unique and can be Unknown \n ")
print(f'{market_train_df.shape[0]} samples and {market_train_df.shape[1]} features in the training market dataset.')
# summary statistics of market data
market_train_df.describe(include = 'all')
# missing data
market_train_df.isna().sum()
# Summary of raw target variable.
target = market_train_df['returnsOpenNextMktres10']
print(f'The range of target: {target.min()}, {target.max()} ')
# 
np.random.seed(1024)
# count all unique assetName
print(market_train_df['assetName'].value_counts().head() )
# whole unique name series 3511 
print(market_train_df['assetName'].nunique())
print(market_train_df['assetCode'].nunique())

# 20 random stocks from market data
print("Assets once appeared in News must have assetName, Unknown means may not have News")
# names of stocks in pilot data
pilot = np.random.choice(market_train_df['assetCode'].unique(), 20)
print(pilot) # numpy.ndarray

# select big company pilot, based on quantiles of volume and close price 
# .groupby('a')['b'].mean()
# market_train_df['assetName'].unique()
# [market_train_df.groupby('assetName')['close'].mean() >=50]
print("groupby assetName has dimension1 3780")
pilot_df = market_train_df[(market_train_df['assetCode'].isin(pilot))]

# setup empty df to save pilot data
data = []
data_big = []


# create trace plot for big company 

# create trace plot for whole pilot 
for asset in pilot:
    asset_df = market_train_df[(market_train_df['assetCode'] == asset)]

    data.append(go.Scatter(
        x = asset_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
        y = asset_df['close'].values,
        name = asset
    ))
layout = go.Layout(dict(title = "Closing prices of 20 random assets",
                  xaxis = dict(title = 'Year'),
                  yaxis = dict(title = 'Price (USD)'),
                  ),legend=dict(
                orientation="h"))
py.iplot(dict(data=data, layout=layout), filename='basic-line')

# how many days in total in this dataset
print(market_train_df['time'].nunique())
print("2498 days of market data")

# check MTD.N, ARTC.O
MTDN = market_train_df[(market_train_df['assetCode'] == 'MTD.N')]['time']
print(MTDN.size)
print(MTDN.nunique())
print(MTDN[0])
print(MTDN[(MTDN.nunique() -1)]) 
# for ARTCO
ARTCO = market_train_df[(market_train_df['assetCode'] == 'ARTC.O')]['time']
print(ARTCO.size)
print(ARTCO.nunique())
print(ARTCO[0])
print(ARTCO[(ARTCO.nunique() -1)]) 

# this is what I mentioned as 3. Intermittent missing data

# check the missing of HES.N
HESN = market_train_df[(market_train_df['assetCode'] == 'HES.N')]['time']
print(HESN.size)
# a class of loess in skmisc
x_hess = range(HESN.size)
y_hess = market_train_df[(market_train_df['assetCode'] == 'HES.N')]['close']
loess_sm = sm.nonparametric.lowess
# hess_loess = loess_sm(y_hess, x_hess,frac=1/20)
hess_loess_1 = loess_sm(y_hess, x_hess,frac=1/5, it = 3, return_sorted = False)
hess_loess_2 = loess_sm(y_hess, x_hess,frac=1/10, it = 3, return_sorted = False)
hess_loess_3 = loess_sm(y_hess, x_hess,frac=1/20, it = 3, return_sorted = False)
hess_loess_4 = loess_sm(y_hess, x_hess,frac=1/40, it = 3, return_sorted = False)
# TRY TO MAKE prediction with loess
# unpack the lowess smoothed points to their values
## lowess_x = list(zip(*hess_loess))[0]
### lowess_y = list(zip(*hess_loess))[1]

# run scipy's interpolation. There is also extrapolation I believe
## f = interp1d(lowess_x, lowess_y, bounds_error=False)
## x_hess_new = range(HESN.size, (HESN.size + 10))
## y_hess_new = f(xnew)

# plot the loess fitting
plt.figure(figsize=(12,6))
plt.scatter(x_hess,y_hess, facecolors = 'none', edgecolor = 'lightblue', label = 'HESS Close Price')
plt.plot(x_hess,hess_loess_1,color = 'magenta', label = 'Loess, 0.2: statsmodel')
plt.plot(x_hess,hess_loess_2,color = 'green', label = 'Loess, 0.1: statsmodel')
plt.plot(x_hess,hess_loess_3,color = 'red', label = 'Loess, 0.05: statsmodel')
plt.plot(x_hess,hess_loess_4,color = 'darkblue', label = 'Loess, 0.025: statsmodel')
## plt.plot(xnew, ynew, color = 'red', label = 'Loess: Prediction')
plt.legend()
plt.title('HESS STOCK 2007 - 2016: Loess Regression')
plt.show()



### loess with skmloess
## hess_loess_skm = skmloess.loess(x_hess, y_hess, weights = None, p = 1, family='gaussian', span = 0.1, degree=2)
# hess_loess_fit = skmloess.loess.fit(hess_loess_skm)
# print(hess_loess_skm)
# plot
# plt.figure(figsize=(12,6))
# plt.scatter(x_hess,y_hess, facecolors = 'none', edgecolor = 'lightblue', label = 'HESS Close Price')
# plt.plot(x_hess,hess_loess_fit,color = 'magenta', label = 'Loess: statsmodel')
# plt.legend()
# plt.title('HESS STOCK 2007 - 2016: Loess regression comparisons')
# plt.show()

# outliers in Market data 
# choose 
news_train_df.head()
news_train_df.describe()
print(f'{news_train_df.shape[0]} samples and {news_train_df.shape[1]} features in the training news dataset.')
# variables News
# The file is too huge to work with text directly
stop = set(stopwords.words('english'))
np.random.seed(1024)
# 
text = ' '.join(np.random.choice(news_train_df['headline'], 100000))
# text = ' '.join(news_train_df['headline'].str.lower().values[-1000000:])
wordcloud = WordCloud(max_font_size=None, stopwords=stop, background_color='white',
                      width=1200, height=1000).generate(text)
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud)
plt.title('Top words in random selected headline')
plt.axis("off")
plt.show()

# missing data, no missing data
news_train_df.isna().sum()
# Check match of news data and market data
print("CHDN.OQ" in market_train_df['assetCode'])
print("0857.HK" in market_train_df['assetCode'])

# transform the date
news_test = news_train_df
news_test['date'] = news_test.time.dt.date  # Add date column
print(news_test.head(3))

## check unique days of news data
print(news_test['date'].nunique())

## check days of news for one asset


