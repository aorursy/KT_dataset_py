import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime as dt # date and time
import matplotlib.pyplot as plt # plotting tool
import seaborn as sns # plotting tool
import random  #random values generator
from scipy.stats import ttest_ind ## statistical library
# this part is to connect to the dataset of the competition, It is only accessible from 
# kaggle notebook

from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()
(market_train_df, news_train_df) = env.get_training_data()
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

## Setting the golabl figure size of all plots included in this report




## Getting rid of the noisy warnings and side information when running plots 
import warnings
warnings.filterwarnings('ignore')
## setting the plot figure size
plt.rcParams['figure.figsize'] = (20.0, 10.0)
## Showing the shape (number of ) of our 1st dataset, the market dataset
print(f'{market_train_df.shape[0]} samples and {market_train_df.shape[1]} features in the training market dataset.')

## Showing the first 5 rows of our dataset
market_train_df.head()


## Showing the shape of our 2nd dataset, the news dataset
print(f'{news_train_df.shape[0]} samples and {news_train_df.shape[1]} features in the training news dataset.')
## Showing the last 5 rows of our dataset
news_train_df.tail()
# This part is to get the Apple stock rows from the original  market dataset

asset1Code = 'AAPL.O'
asset1_df = market_train_df[(market_train_df['assetCode'] == asset1Code) 
                            & (market_train_df['time'] < '2017-01-01')]
# This part is to get a random stock rows from the original market dataset
asset2Code = market_train_df['assetCode'][random.randint(0, market_train_df.shape[0])]
asset2_df = market_train_df[(market_train_df['assetCode'] == f'{asset2Code}') 
                            & (market_train_df['time'] < '2017-01-01')]
# This part is to get the Apple stock rows from the original  news dataset
asset3Name = 'Apple Inc'
asset3_df = news_train_df.loc[lambda df: df['assetName'] == asset3Name, :]

## This part is to plot a histogram that shows the distribution of returns in apple stock 
assets = tuple(asset1_df.loc[:,'returnsClosePrevRaw1'])
sns.distplot(assets, hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 2, 'color':'k'})
meanreturn = asset1_df.loc[:,'returnsClosePrevRaw1'].values.mean()
plt.axvline(meanreturn, 
            color='r',
            linestyle='dashed',
            linewidth=2)
plt.xlabel('Apple returns at close time')
plt.ylabel('Frequency')
plt.title('Apple stock returns Frequency Distribution');
## This part is to plot a histogram that shows the distribution of returns in the random stock 
sns.distplot(asset2_df.loc[:,'returnsClosePrevRaw1'], hist=True, kde=True, 
             bins=int(180/5), color = 'burlywood', 
             hist_kws={'edgecolor':'white'},
             kde_kws={'linewidth': 2, 'color':'k'});
plt.axvline(asset2_df.loc[:,'returnsClosePrevRaw1'].mean(), 
            color='r', 
            linestyle='dashed', 
            linewidth=2)
plt.xlabel(f'{asset2_df.assetName[0]} returns at close time')
plt.ylabel('Frequency')
plt.title(f'{asset2_df.assetName[0]} stock returns Frequency Distribution');
## I will merge both data sets on the time column

asset1_df['date'] = asset1_df['time'].dt.strftime(date_format='%Y-%m-%d')
asset3_df['date'] = asset3_df['time'].dt.strftime(date_format='%Y-%m-%d')
## Taking only the negative sentiment rows and grouping it by the date variable to be able to merge the tabels together

meanSent = pd.DataFrame(asset3_df.groupby('date')['sentimentNegative'].mean())

asset_merged = pd.merge(asset1_df, meanSent, on= 'date')
# plotting a scattor plot alongside a regression line that would show if there is a correlation
sns.regplot(asset_merged.loc[:,'sentimentNegative'].values,
            asset_merged.loc[:,'returnsOpenNextMktres10'].values, 
            line_kws={'color' : 'darkred'})
plt.xlabel ('Negative Sentiment Level')
plt.ylabel('Apple Stock Returns Projected')
plt.title(f'{asset1_df.assetName[0]} stock returns Frequency Distribution');
## getting the positive and negative sentiment rows
asset_negative = asset3_df.loc[lambda df: df.loc[:,'sentimentClass'] == -1, :]

asset_positive =  asset3_df.loc[lambda df: df.loc[:,'sentimentClass'] == 1, :]
meanNeg = pd.DataFrame(asset_negative.groupby('date')['sentimentNegative'].mean())
meanPos = pd.DataFrame(asset_positive.groupby('date')['sentimentPositive'].mean())
asset_merged_neg = pd.merge(asset1_df, meanNeg, on= 'date')
asset_merged_pos = pd.merge(asset1_df, meanPos, on= 'date')
## Graphing a histogram to show the different distributions
sns.distplot(asset_merged_neg.loc[:,'returnsOpenNextMktres10'], color="r")
sns.distplot(asset_merged_pos.loc[:,'returnsOpenNextMktres10'], color="g")

plt.xlabel ('returns values')
plt.ylabel('Frequency')
plt.title('Distribution of Returns of the opening of 10 days');
## Applying a t-test on the 
print(ttest_ind(asset_merged_pos.loc[:,'returnsOpenNextMktres10'], 
                asset_merged_neg.loc[:,'returnsOpenNextMktres10']))
## this part is just a showcase of one of the ways to visulaize our data.
from wordcloud import WordCloud, STOPWORDS 
stop = set(STOPWORDS)
text = ' '.join(asset_negative.loc[:,'headline'].str.lower().values)
wordcloud = WordCloud(max_font_size=None, stopwords=stop, background_color='white',
                      width=1200, height=1000).generate(text)
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud)
plt.title('Top  words in headlines classified as negative of Apple')
plt.axis("off")
plt.show();