

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
from kaggle.competitions import twosigmanews

env = twosigmanews.make_env()
(market_train_df, news_train_df) = env.get_training_data()
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))





plt.rcParams['figure.figsize'] = (20.0, 10.0)
print(f'{market_train_df.shape[0]} samples and {market_train_df.shape[1]} features in the training market dataset.')
    


market_train_df.head()
print(f'{news_train_df.shape[0]} samples and {news_train_df.shape[1]} features in the training news dataset.')
    


news_train_df.tail()
asset1Code = 'AAPL.O'
asset1_df = market_train_df[(market_train_df['assetCode'] == asset1Code) & (market_train_df['time'] < '2017-01-01')]

sns.distplot(asset1_df['returnsClosePrevRaw1'], hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
plt.axvline(asset1_df['returnsClosePrevRaw1'].mean(), color='r', linestyle='dashed', linewidth=2)
plt.xlabel('Apple returns at close time')
plt.ylabel('Frequency')
plt.title('Apple stock returns Frequency Distribution')
asset2Code = 'Apple Inc'
asset2_df = news_train_df.loc[lambda df: df['assetName'] == asset2Code, :]
asset2_df.head()
## I will only keep the day-month-year in the time column and then merge both data sets on that

asset1_df['date'] = asset1_df['time'].dt.strftime(date_format='%Y-%m-%d')
asset2_df['date'] = asset2_df['time'].dt.strftime(date_format='%Y-%m-%d')
merged_df = pd.merge(asset1_df, asset2_df, on='date')


group_ret = merged_df['returnsOpenNextMktres10'].drop_duplicates()
##group_ret2 = merged_df['returnsClosePrevRaw10'].drop_duplicates()
meanSent = merged_df.groupby('returnsOpenNextMktres10')['sentimentNegative'].mean()
##meanSent2 = merged_df.groupby('returnsCloseNextRaw10')['sentimentNegative'].mean()
merged_urgent = merged_df.loc[lambda df: df['urgency'] < 3, :]
group_urgent = merged_urgent['returnsOpenNextMktres10'].drop_duplicates()
##group_urgent2 = merged_urgent['returnsCloseNextRaw10'].drop_duplicates()
mean_urgent = merged_urgent.groupby('returnsOpenNextMktres10')['sentimentNegative'].mean()
##mean_urgent2 = merged_urgent.groupby('returnsCloseNextRaw10')['sentimentNegative'].mean()
sns.regplot(meanSent.values, group_ret.values)

sns.regplot(mean_urgent.values, group_urgent.values)
meanSent2 = merged_df.groupby('returnsOpenNextMktres10')['sentimentPositive'].mean()

sns.regplot(meanSent2.values, group_ret.values)
merged_positive = merged_df.loc[lambda df: df['sentimentPositive'] > 0.6, :]
merged_negative = merged_df.loc[lambda df: df['sentimentNegative'] > 0.6, :]
sns.distplot(merged_positive['returnsOpenNextMktres10'], color="r")
sns.distplot(merged_negative['returnsOpenNextMktres10'], color="b")
from scipy.stats import ttest_ind
print(ttest_ind(merged_positive['returnsOpenNextMktres10'], merged_negative['returnsOpenNextMktres10'], equal_var=False))
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
stop = set(STOPWORDS)
text = ' '.join(merged_urgent['headline'].str.lower().values)
wordcloud = WordCloud(max_font_size=None, stopwords=stop, background_color='white',
                      width=1200, height=1000).generate(text)
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud)
plt.title('Top  words in headlines classified as urgent of Apple')
plt.axis("off")
plt.show()