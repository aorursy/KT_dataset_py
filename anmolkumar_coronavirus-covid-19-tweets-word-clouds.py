# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Data Analysis

import pandas as pd

import matplotlib.pyplot as plt

plt.style.use('seaborn-darkgrid')

import seaborn as sns



#Data Preprocessing and Feature Engineering

from nltk import PorterStemmer

from textblob import TextBlob

import re

from nltk.corpus import stopwords

from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from wordcloud import WordCloud, ImageColorGenerator

from PIL import Image

import plotly.express as pex



import urllib

import requests
tweets = pd.read_csv('/kaggle/input/covid19-tweets/covid19_tweets.csv')

tweets.columns = tweets.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

print('Tweets data shape: ', tweets.shape)

tweets.head()
def remove_pattern(text, pattern):

    r = re.findall(pattern, text)

    for i in r:

        text = re.sub(i, "", text)

    return text
tweets['text'] = np.vectorize(remove_pattern)(tweets['text'], "@[\w]*")

tweets['text'] = np.vectorize(remove_pattern)(tweets['text'], "#[\w]*")

tweets['text'] = np.vectorize(remove_pattern)(tweets['text'], '[0-9]')

tweets['text'] = tweets['text'].str.replace("[^a-zA-Z#]", " ")

tweets['text'] = tweets['text'].apply(lambda x: ' '.join([i for i in x.split() if len(i) > 3]))

tweets.head()
tweet = tweets['text'].apply(lambda x: x.split())

tweet.head()
ps = PorterStemmer()

tweet = tweet.apply(lambda x: [ps.stem(i) for i in x])

tweet.head()
for i in range(len(tweet)):

    tweet[i] = ' '.join(tweet[i])

tweets['text'] = tweet

tweets.head()
India = ' '.join(text for text in tweets['text'][tweets['user_location'] == 'India'])

China = ' '.join(text for text in tweets['text'][tweets['user_location'] == 'China'])

USA = ' '.join(text for text in tweets['text'][tweets['user_location'] == 'USA'])

SA = ' '.join(text for text in tweets['text'][tweets['user_location'] == 'South Africa'])
# combining the image with the dataset

Mask = np.array(Image.open(requests.get('http://clipart-library.com/image_gallery2/Twitter-PNG-Image.png', stream = True).raw))



# We use the ImageColorGenerator library from Wordcloud 

# Here we take the color of the image and impose it over our wordcloud

image_colors = ImageColorGenerator(Mask)



# Now we use the WordCloud function from the wordcloud library 

wc = WordCloud(background_color = 'black', height = 1500, width = 4000, mask = Mask).generate(India)



# Size of the image generated 

plt.figure(figsize = (10, 20))



# Here we recolor the words from the dataset to the image's color

# recolor just recolors the default colors to the image's blue color

# interpolation is used to smooth the image generated 

plt.imshow(wc.recolor(color_func = image_colors), interpolation = "hamming")



plt.axis('off')

plt.show()
Mask = np.array(Image.open(requests.get('http://clipart-library.com/image_gallery2/Twitter-PNG-Image.png', stream = True).raw))

image_colors = ImageColorGenerator(Mask)

wc = WordCloud(background_color = 'black', height = 1500, width = 4000, mask = Mask).generate(China)

plt.figure(figsize = (10, 20))

plt.imshow(wc.recolor(color_func = image_colors), interpolation = "hamming")

plt.axis('off')

plt.show()
Mask = np.array(Image.open(requests.get('http://clipart-library.com/image_gallery2/Twitter-PNG-Image.png', stream = True).raw))

image_colors = ImageColorGenerator(Mask)

wc = WordCloud(background_color = 'black', height = 1500, width = 4000, mask = Mask).generate(USA)

plt.figure(figsize = (10, 20))

plt.imshow(wc.recolor(color_func = image_colors), interpolation = "hamming")

plt.axis('off')

plt.show()
Mask = np.array(Image.open(requests.get('http://clipart-library.com/new_gallery/18-189677_instagram-logo-twitter-logo-bird-twitter-logo-2017.png', stream = True).raw))

image_colors = ImageColorGenerator(Mask)

wc = WordCloud(background_color = 'black', height = 1500, width = 4000, mask = Mask).generate(SA)

plt.figure(figsize = (10, 20))

plt.imshow(wc.recolor(color_func = image_colors), interpolation = "hamming")

plt.axis('off')

plt.show()
tweets_f = tweets.copy()



tweets['date'] = pd.to_datetime(tweets['date'])



tweets['year'] = tweets['date'].dt.year

tweets['month'] = tweets['date'].dt.month

tweets['day'] = tweets['date'].dt.day

tweets['dayofweek'] = tweets['date'].dt.dayofweek

tweets['hour'] = tweets['date'].dt.hour

tweets['minute'] = tweets['date'].dt.minute

tweets['dayofyear'] = tweets['date'].dt.dayofyear

tweets['date_only'] = tweets['date'].dt.date

tweets.head()
plt.figure(figsize = (10, 5))

tweets_daily = tweets.groupby(["dayofweek"])["text"].count().reset_index()

tweets_daily.columns = ['dayofweek', 'tweet_count']

sns.lineplot(x = 'dayofweek', y = 'tweet_count', hue = None, data = tweets_daily)

plt.title('Tweets count by day of the week (Monday-Sunday)')

plt.show()
plt.figure(figsize = (10, 5))

tweets_yearly = tweets.groupby(["dayofyear"])["text"].count().reset_index()

tweets_yearly.columns = ['dayofyear', 'tweet_count']

sns.lineplot(x = 'dayofyear', y = 'tweet_count', hue = None, data = tweets_yearly)

plt.title('Tweets count by day of year')

plt.show()
tweetMax = tweets.groupby(tweets.user_name)[["user_location"]].count().reset_index()

tweetMax.columns= ['user_name','count']

tweetMax = tweetMax.sort_values(by = "count" , ascending = False)

tweetMax= tweetMax.head(10)

tweetMax





fig = pex.bar(tweetMax, x = 'user_name', y='count', title = 'User Standings')

fig.show()