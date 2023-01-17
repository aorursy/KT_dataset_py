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
df = pd.read_csv('/kaggle/input/joe-biden-tweets/JoeBidenTweets.csv', parse_dates = [2])

df.head()
# Drop some features which might not be useful for us

# We drop username since we know it is Joe Biden.

data = df.copy()

data.drop(['id', 'username', 'link'], axis = 1,inplace = True)

data.head()

data.set_index(['timestamp'], inplace = True)
data.tail()
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
plt.figure(figsize = (20, 10))

plt.title('Likes')

plt.style.use('ggplot')

plt.plot(data.index, data.likes);
plt.figure(figsize = (20, 10))

plt.title('Retweets')

plt.style.use('dark_background')

plt.plot(data.index, data.retweets);
import re



def processTweets(tweet):

    '''Cleans the tweet and returns it'''

    tweet = str(tweet)

    tweet = re.sub(r'[^\w\s]','',tweet) # Remove Punctuations

    return tweet



data['tweet'] = data['tweet'].apply(processTweets)
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
wc = WordCloud(width = 1920, height = 1080).generate(data.tweet[0])

plt.figure(figsize = (20, 10))

plt.axis('off')

plt.imshow(wc);
text = " ".join(tweet for tweet in data.tweet)
stopwords = set(STOPWORDS)

wordcloud = WordCloud(stopwords=stopwords,

                      background_color="white",

                      width = 1920,

                      height = 1080).generate(text)

plt.figure(figsize = (20, 10))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()