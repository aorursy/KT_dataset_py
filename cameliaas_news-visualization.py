import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import datetime
import matplotlib.pyplot as plt
import matplotlib
import re
from scipy import stats

matplotlib.rcParams['figure.figsize'] = (10, 5)
matplotlib.rcParams['font.size'] = 12

import random
random.seed(1)
import time

import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import get_scorer
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
import lightgbm as lgb
from sklearn.externals.joblib import Parallel, delayed
from sklearn.base import clone

import pickle


import warnings
warnings.filterwarnings('ignore')
matplotlib.rcParams['figure.figsize'] = (10, 5)
matplotlib.rcParams['font.size'] = 12
from kaggle.competitions import twosigmanews
# You can only call make_env() once, so don't lose it!
env = twosigmanews.make_env()
print('Done!')
(market_train_orig, news_train_orig) = env.get_training_data()

market_train_df = market_train_orig.copy()
news_train_df = news_train_orig.copy()
print('Market train shape: ',market_train_df.shape)
print('News train shape: ', news_train_df.shape)
news_train_df.describe()
# Sort values by time then extract date
news_train_df = news_train_df.sort_values(by='time')
news_train_df['date'] = news_train_df['time'].dt.date
# Function to plot time series data
def plot_vs_time(data_frame, column, calculation='mean', span=10):
    if calculation == 'mean':
        group_temp = data_frame.groupby('date')[column].mean().reset_index()
    if calculation == 'count':
        group_temp = data_frame.groupby('date')[column].count().reset_index()
    if calculation == 'nunique':
        group_temp = data_frame.groupby('date')[column].nunique().reset_index()
    group_temp = group_temp.ewm(span=span).mean()
    fig = plt.figure(figsize=(10,3))
    plt.plot(group_temp['date'], group_temp[column])
    plt.xlabel('Time')
    plt.ylabel(column)
    plt.title('%s versus time' %column)
plot_vs_time(news_train_df, 'sourceId', calculation='count', span=10)
plt.title('News count vs time')
plt.ylabel('Count')
# Plot time evolution of several parameters

columns = ['urgency', 'takeSequence', 'companyCount','marketCommentary','sentenceCount',\
           'firstMentionSentence','relevance','sentimentClass','sentimentWordCount','noveltyCount24H', 'volumeCounts24H']

for column in columns:
    plot_vs_time(news_train_df, column)
time_delay = (pd.to_datetime(news_train_df['time']) - pd.to_datetime(news_train_df['firstCreated']))
time_delay_log10 = np.log10(time_delay.dt.total_seconds()/60+1)
plt.hist(time_delay_log10, bins=np.arange(0,2.5,0.25), rwidth=0.7)
plt.xlabel('$Log_{10}$(Time delay in minutes +1)')
plt.ylabel('Counts')
plt.title('Delay time distribution')
time_delay_min = time_delay.dt.total_seconds()/60
time_delay_df = time_delay_min.to_frame().join(news_train_df['date'].to_frame())
time_delay_df.columns = ['delay','date']
plot_vs_time(time_delay_df, 'delay')
plt.ylabel('Delay (minutes)')
urgency_count = news_train_df.groupby('urgency')['sourceId'].count()
urgency_count = urgency_count/urgency_count.sum()
print('Urgency ratio')
urgency_count.sort_values(ascending=True)
take_sequence = news_train_df.groupby('takeSequence')['sourceId'].count()
take_sequence = take_sequence.sort_values(ascending= False)
take_sequence[:10].plot.barh()
plt.xlabel('Count')
plt.ylabel('Take sequence')
plt.title('Top 10 take sequence')
plt.gca().invert_yaxis()
del take_sequence

provider_count = news_train_df.groupby('provider')['sourceId'].count()
provider_sort = provider_count.sort_values(ascending= False)
provider_sort[:10].plot.barh()
plt.xlabel('Count')
plt.ylabel('Provider')
plt.title('Top 10 news provider')
plt.gca().invert_yaxis()
del provider_count
# Extract data from a single cell
def contents_to_list(contents):
    text = contents[1:-1]
    text = re.sub(r",",' ',text)
    text = re.sub(r"'","", text)
    text_list = text.split('  ')
    return text_list

# Put data from columns into dict
def get_content_dict(content_column):
    content_dict = {}
    for i in range(len(content_column)):
        this_cell = content_column[i]
        content_list = contents_to_list(this_cell)        
        for content in content_list:
            if content in content_dict.keys():
                content_dict[content] += 1
            else:
                content_dict[content] = 1
    return content_dict
subjects = news_train_df.sample(n=10000, random_state=1)['subjects']
subjects_dict = get_content_dict(subjects)
subjects_df = pd.Series(subjects_dict).sort_values(ascending=False)
subjects_df[:15].plot.barh()
plt.ylabel('Subjects')
plt.xlabel('Counts')
plt.title('Top subjects for 10k data')
plt.gca().invert_yaxis()
del subjects_df
audiences = news_train_df.sample(n=10000, random_state=1)['audiences']
audiences_dict = get_content_dict(audiences)
audiences_df = pd.Series(audiences_dict).sort_values(ascending=False)
audiences_df[:15].plot.barh()
plt.ylabel('Audiences')
plt.xlabel('Counts')
plt.title('Top audiences for 10k data')
plt.gca().invert_yaxis()
news_train_df['companyCount'].hist(bins=np.arange(0,30,1))
plt.xlabel('Company count')
plt.title('Company count distribution')
head_line = news_train_df.groupby('headlineTag')['sourceId'].count()
head_line_sort = head_line.sort_values(ascending= False)
head_line_sort[:10].plot.barh()
plt.xlabel('Count')
plt.ylabel('Head line')
plt.title('Top 10 head lines')
plt.gca().invert_yaxis()
del head_line

news_train_df['firstMentionSentence'].hist(bins=np.arange(0,20,1))
plt.xlabel('First mention sentence')
plt.ylabel('Count')
plt.title('First mention sentence distribution')
sentence_urgency = news_train_df.groupby('firstMentionSentence')['urgency'].mean()
sentence_urgency.head(5)
del sentence_urgency
news_train_df['relevance'].hist(bins=np.arange(0,1.01,0.05))
plt.xlabel('Relevance')
plt.ylabel('Count')
plt.title('Relevance distribution')
sentence_relevance = news_train_df.groupby('firstMentionSentence')['relevance'].mean()
sentence_relevance[:15].plot.barh()
plt.xlabel('Relevance')
plt.title('Relevance by sentence')
plt.gca().invert_yaxis()
del sentence_relevance
sentimentWordCount = news_train_df.groupby('sentimentWordCount')['sourceId'].count().reset_index()
plt.plot(sentimentWordCount['sentimentWordCount'], sentimentWordCount['sourceId'])
plt.xlim(0,300)
plt.xlabel('Sentiment words count')
plt.ylabel('Count')
plt.title('Sentiment words count distribution')
del sentimentWordCount
sentimentWordRatio = news_train_df.groupby('sentimentWordCount')['relevance'].mean()
plt.plot(sentimentWordRatio)
plt.xlim(0,2000)
plt.ylabel('Relevance')
plt.xlabel('Sentiment word count')
plt.title('Sentiment word count and relevance')
del sentimentWordRatio
news_train_df['sentimentRatio'] = news_train_df['sentimentWordCount']/news_train_df['wordCount']
news_train_df['sentimentRatio'].hist(bins=np.linspace(0,1.001,40))
plt.xlabel('Sentiment ratio')
plt.ylabel('Count')
plt.title('Sentiment ratio distribution')
news_train_df.sample(n=10000, random_state=1).plot.scatter('sentimentRatio', 'relevance')
plt.title('Relevance vs sentiment ratio of 10k samples')
asset_name = news_train_df.groupby('assetName')['sourceId'].count()
print('Total number of assets: ',news_train_df['assetName'].nunique())
asset_name = asset_name.sort_values(ascending=False)
asset_name[:10].plot.barh()
plt.gca().invert_yaxis()
plt.xlabel('Count')
plt.title('Top 10 assets news')
for i, j in zip([-1, 0, 1], ['negative', 'neutral', 'positive']):
    df_sentiment = news_train_df.loc[news_train_df['sentimentClass'] == i, 'assetName']
    print(f'Top mentioned companies for {j} sentiment are:')
    print(df_sentiment.value_counts().head(5))
    print('')
# Function to remove outliers
def remove_outliers(data_frame, column_list, low=0.02, high=0.98):
    temp_frame = data_frame
    for column in column_list:
        this_column = data_frame[column]
        quant_df = this_column.quantile([low,high])
        low_limit = quant_df[low]
        high_limit = quant_df[high]
        temp_frame[column] = data_frame[column].clip(lower=low_limit, upper=high_limit)
    return temp_frame
# Remove outlier
columns_outlier = ['takeSequence', 'bodySize', 'sentenceCount', 'wordCount', 'sentimentWordCount', 'firstMentionSentence','noveltyCount12H',\
                  'noveltyCount24H', 'noveltyCount3D', 'noveltyCount5D', 'noveltyCount7D', 'volumeCounts12H', 'volumeCounts24H',\
                  'volumeCounts3D','volumeCounts5D','volumeCounts7D']
news_rmv_outlier = remove_outliers(news_train_df, columns_outlier)
# Plot correlation
columns_corr = ['urgency', 'takeSequence', 'companyCount','marketCommentary','sentenceCount',\
           'firstMentionSentence','relevance','sentimentClass','sentimentWordCount','noveltyCount24H',\
           'noveltyCount3D', 'noveltyCount5D', 'noveltyCount7D','volumeCounts24H','volumeCounts3D','volumeCounts5D','volumeCounts7D']
colormap = plt.cm.RdBu
plt.figure(figsize=(18,15))
sns.heatmap(news_rmv_outlier[columns_corr].astype(float).corr(), linewidths=0.1, vmax=1.0, vmin=-1., square=True, cmap=colormap, linecolor='white', annot=True)
plt.title('Pair-wise correlation')
