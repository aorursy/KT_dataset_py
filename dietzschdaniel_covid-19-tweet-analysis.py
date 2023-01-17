# Data Processing

import numpy as np 

import pandas as pd 

import re



# Data Visualization

import matplotlib.pyplot as plt

%matplotlib inline



import plotly.express as px

import plotly.graph_objects as go

import plotly.figure_factory as ff

from plotly.colors import n_colors

from plotly.subplots import make_subplots



import missingno as msno





import seaborn as sns

sns.set(style='whitegrid')



from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
train = pd.read_csv("/kaggle/input/covid-19-nlp-text-classification/Corona_NLP_train.csv", encoding='latin-1')

test = pd.read_csv("/kaggle/input/covid-19-nlp-text-classification/Corona_NLP_test.csv")
train
plt.figure(figsize=(15,5))

b = sns.countplot(x='Sentiment', data=train, order=['Extremely Negative', 'Negative', 'Neutral', 'Positive', 'Extremely Positive'])

b.set_title("Sentiment Distribution");
train.isna().sum()
msno.matrix(train);
train['UserName'].nunique()
train['ScreenName'].nunique()
train.head()
# Remove URLs



def remove_urls(text):

    return re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text)

train['Content']=train['OriginalTweet'].apply(lambda x:remove_urls(x))
# Remove HTML



def remove_urls(text):

    return re.sub(r'<.*?>', '', text)

train['Content']=train['Content'].apply(lambda x:remove_urls(x))
fig, (ax) = plt.subplots(1,1,figsize=[15, 10])

wc = WordCloud(width=600,height=400, background_color='white', colormap="Greys").generate(" ".join(train['Content']))



ax.imshow(wc,interpolation='bilinear')

ax.axis('off')

ax.set_title('Wordcloud of Tweets');
fig, (ax) = plt.subplots(1,1,figsize=[15, 10])

wc = WordCloud(width=600,height=400, background_color='white', colormap="Greens").generate(" ".join(train['Content'][train['Sentiment'] == 'Extremely Positive']))



ax.imshow(wc,interpolation='bilinear')

ax.axis('off')

ax.set_title('Wordcloud of Tweets with Extremely Positive Sentiment');
fig, (ax) = plt.subplots(1,1,figsize=[15, 10])

wc = WordCloud(width=600,height=400, background_color='white', colormap="Blues").generate(" ".join(train['Content'][train['Sentiment'] == 'Positive']))



ax.imshow(wc,interpolation='bilinear')

ax.axis('off')

ax.set_title('Wordcloud of Tweets with Positive Sentiment');
fig, (ax) = plt.subplots(1,1,figsize=[15, 10])

wc = WordCloud(width=600,height=400, background_color='white', colormap="Purples").generate(" ".join(train['Content'][train['Sentiment'] == 'Neutral']))



ax.imshow(wc,interpolation='bilinear')

ax.axis('off')

ax.set_title('Wordcloud of Tweets with Neutral Sentiment');
fig, (ax) = plt.subplots(1,1,figsize=[15, 10])

wc = WordCloud(width=600,height=400, background_color='white', colormap="Oranges").generate(" ".join(train['Content'][train['Sentiment'] == 'Negative']))



ax.imshow(wc,interpolation='bilinear')

ax.axis('off')

ax.set_title('Wordcloud of Tweets with Negative Sentiment');
fig, (ax) = plt.subplots(1,1,figsize=[15, 10])

wc = WordCloud(width=600,height=400, background_color='white', colormap="Reds").generate(" ".join(train['Content'][train['Sentiment'] == 'Extremely Negative']))



ax.imshow(wc,interpolation='bilinear')

ax.axis('off')

ax.set_title('Wordcloud of Tweets with Extremely Negative Sentiment');
# Get Tweet length



def tweet_length(text):

    return len(text)

train['TweetLength']=train['Content'].apply(lambda x:tweet_length(x))
b = sns.boxplot(y = 'TweetLength', data = train)

b.set_title("TweetLength Distribution");
b = sns.boxplot(y = train['TweetLength'][train['Sentiment'] == 'Extremely Positive'], data = train)

b.set_title("TweetLength Distribution for Extremely Positive Sentiment");
plt.figure(figsize=(15,5))

b = sns.boxplot(y='TweetLength', x='Sentiment', data=train, order=['Extremely Negative', 'Negative', 'Neutral', 'Positive', 'Extremely Positive']);

b.set_title("TweetLength Distribution for Sentiment");
train['Country'] = train['Location'].str.split(',').str[-1]