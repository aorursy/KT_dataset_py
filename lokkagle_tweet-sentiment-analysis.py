# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
train_data.head()
test_data = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
test_data.head()
dd = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')
dd.head()
train_data.isna().sum()
train_data.info()
train_data.dropna(inplace= True)
train_data.isna().sum()
# adding text length and wordcounts using lambda functions
train_data['length'] = train_data['text'].apply(lambda x : len(str(x)))
train_data['wordcount'] = train_data['text'].apply(lambda x : len(str(x).split()))
train_data.head()
# sentiments
sns.countplot(x = 'sentiment', data = train_data)
plt.show()
# sentiment word length distribution
senti = ['positive', 'negative', 'neutral']

for i in senti:
    print('---------'+ i +'-------------')
    train_data[train_data['sentiment'] == i]['length'].hist(figsize = (15,5))
    plt.show()
# sentiment word count distribution
senti = ['positive', 'negative', 'neutral']

for i in senti:
    print('---------'+ i +'-------------')
    train_data[train_data['sentiment'] == i]['wordcount'].hist(figsize = (15,5))
    plt.show()
import nltk
nltk.download('stopwords')
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
# Find emoji patterns
emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

# Basic function to clean the text
def clean_text(text):
    text = str(text)
    # Remove emojis
    text = emoji_pattern.sub(r'', text)
    # Remove identifications
    text = re.sub(r'@\w+', '', text)
    # Remove links
    text = re.sub(r'http.?://[^/s]+[/s]?', '', text)
    return text.strip().lower()

train_data['text'] = train_data['text'].apply(lambda x:clean_text(x))
from wordcloud import WordCloud
import matplotlib.pyplot as plt

wc = WordCloud(background_color='white', width=3000, height=2500).generate(str(train_data['text']))
plt.figure(figsize=(10,10))
plt.title('most sound words')
plt.imshow(wc)
plt.axis('off')
plt.show()
print('positive words : {}'.format(len(train_data[train_data['sentiment'] == 'positive'])))
wc = WordCloud(background_color='white', width=3000, height=2500).generate(str(train_data[train_data['sentiment'] == 'positive']['text']))
plt.figure(figsize=(10,10))
plt.title('most sound words')
plt.imshow(wc)
plt.axis('off')
plt.show()

print('negative words : {}'.format(len(train_data[train_data['sentiment'] == 'negative'])))
wc = WordCloud(background_color='white', width=3000, height=2500).generate(str(train_data[train_data['sentiment'] == 'negative']['text']))
plt.figure(figsize=(10,10))
plt.title('most sound words')
plt.imshow(wc)
plt.axis('off')
plt.show()
print('neutral words : {}'.format(len(train_data[train_data['sentiment'] == 'neutral'])))
wc = WordCloud(background_color='white', width=3000, height=2500).generate(str(train_data[train_data['sentiment'] == 'neutral']['text']))
plt.figure(figsize=(10,10))
plt.title('most sound words')
plt.imshow(wc)
plt.axis('off')
plt.show()

