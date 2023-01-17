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
## Imporing neccesary libraries
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

pd.set_option('display.max_rows',100000)
pd.set_option('display.max_colwidth',None)
# Loading the train and the test dataset
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
train.head() # Lets look at few columns in the train dataset
test.head() # Lets look at few columns in the test dataset
# First let's create a copy of our dataset so that our original dataset is not tempered
df_train = train.copy()
df_test = test.copy()
df_train.head() # check whether it is properly copied
df_test.head()
df_train.info()
# Looking for descriptive statistic 
# By defualt it will show the statistic of numeric column
df_train.describe()
# If you want to see descriptive statistic of object/string columns use *include*
df_train.describe(include = 'object')
# Checking for null value
(df_train.isnull().sum()/df_train.shape[0])*100
df_train.loc[(df_train.keyword.isnull() == True),:].head()
df_train.loc[(df_train.location.isnull() == True),:].head()
# Checking for null values in test dataset
(df_test.isnull().sum()/df_test.shape[0])*100
df_test.loc[(df_test.location.isnull() == True),:].head()
df_test.loc[(df_test.keyword.isnull() == True),:].head()
# Lets look at the shape of the train dataset
df_train.shape
# Lets look at the shape of the test dataset
df_test.shape
# Lets look at some value counts
# Checking only 5 values and the output becomes more longer. Feel free toh remo [:5] and check for all the fields
cols = df_train.columns
for i in range(0,len(cols)):
    print("Column :", cols[i].upper())
    print(df_train[cols[i]].value_counts(dropna = True)[:5])
    print('********************************************')
# Lets look at some value counts
cols = df_test.columns
for i in range(0,len(cols)):
    print("Column :", cols[i].upper())
    print(df_test[cols[i]].value_counts(dropna = True)[:5])
    print('********************************************')
# Looking for the distribution of Target variable
sns.set_style()
sns.countplot(data= df_train,x = df_train['target'],palette = 'rocket')
plt.title ('Distribution of Target variable')
plt.xlabel("Target")
plt.ylabel("Count of Target")
plt.show()
df_train['target'].value_counts()

sns.distplot(df_train['target'])
plt.show()
# Plot 20 keywords from teh dataset
sns.barplot(y=df_train['keyword'].value_counts()[:20].index,x=df_train['keyword'].value_counts()[:20])
df_train.loc[df_train['text'].str.contains('disaster', na=False, case=False)].target.value_counts()
# Plot 10 location from the dataset
sns.barplot(y=df_train['location'].value_counts()[:10].index,x=df_train['location'].value_counts()[:10],
            orient='h')
# Replacing the ambigious locations name with Standard names
df_train['location'].replace({'United States':'USA',
                           'New York':'USA',
                            "London":'UK',
                            "Los Angeles, CA":'USA',
                            "Washington, D.C.":'USA',
                            "California":'USA',
                            "Chicago, IL":'USA',
                            "Chicago":'USA',
                            "New York, NY":'USA',
                            "California, USA":'USA',
                            "FLorida":'USA',
                            "Nigeria":'Africa',
                            "Kenya":'Africa',
                            "Everywhere":'Worldwide',
                            "San Francisco":'USA',
                            "Florida":'USA',
                            "United Kingdom":'UK',
                            "Los Angeles":'USA',
                            "Toronto":'Canada',
                            "San Francisco, CA":'USA',
                            "NYC":'USA',
                            "Seattle":'USA',
                            "Earth":'Worldwide',
                            "Ireland":'UK',
                            "London, England":'UK',
                            "New York City":'USA',
                            "Texas":'USA',
                            "London, UK":'UK',
                            "Atlanta, GA":'USA',
                            "Mumbai":"India",
                            "Sao Paulo, Brazil" : "Brazil"},inplace=True)
# Plot the barplot and check whether the location column has changed or not
sns.barplot(y=df_train['location'].value_counts()[:10].index,x=df_train['location'].value_counts()[:10],
            orient='h')
# Let's look at the ss column whether we can find some insights from that data or not
df_train.loc[df_train['location']=='ss',:]
# Let's plot the character in the tweet
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))
tweet_len=df_train[df_train['target']==1]['text'].str.len()
ax1.hist(tweet_len,color='red')
ax1.set_title('disaster tweets')
tweet_len=df_train[df_train['target']==0]['text'].str.len()
ax2.hist(tweet_len,color='green')
ax2.set_title('Not disaster tweets')
fig.suptitle('Characters in tweets')
plt.show()
# Let's plot the words from the text
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))
tweet_len=df_train[df_train['target']==1]['text'].str.split().map(lambda x: len(x))
ax1.hist(tweet_len,color='red')
ax1.set_title('disaster tweets')
tweet_len=df_train[df_train['target']==0]['text'].str.split().map(lambda x: len(x))
ax2.hist(tweet_len,color='green')
ax2.set_title('Not disaster tweets')
fig.suptitle('Words in a tweet')
plt.show()
# Let's look at the top 25 rows to understand what all things we need to take care when cleaning the corpus/ text field
df_train['text'][:25]
# Importing re library to clean the text
import re

# Importing string library to remove/escape the punctuations from the text column
import string 

# Preprocessing the text field
def preprocessing (text):
    '''
    Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.
    '''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('[#|@|!|$|%||^|&|*|(|)|[|{|[|\]]','',text)
    text = re.sub('im','i am',text)
    text = re.sub('û','u',text)
    text = text.strip()
    
    return text

        
# removing emoji's
def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# applying the preprocessing and remove_emoji function on the train and test dataset
df_train['text'] = df_train['text'].apply(lambda x:preprocessing(x))
df_train['text'] = df_train['text'].apply(lambda x:remove_emoji(x))

df_test['text'] = df_test['text'].apply(lambda x:preprocessing(x))
df_test['text'] = df_test['text'].apply(lambda x:remove_emoji(x))
# checking whether the data is cleaned or not
df_train['text'][:5]
# Stopwords removal 

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

def remove_stopwords(string):
    word_list = [word for word in string.split()]
    stopwords_list = list(stopwords.words("english"))
    for word in word_list:
        if word in stopwords_list:
            word_list.remove(word)
    return(' '.join(word_list))
        
df_train['text'] = list(map(lambda x: remove_stopwords(x), df_train['text']))

df_test['text'] = list(map(lambda x: remove_stopwords(x), df_test['text']))
df_train.head()
#not a disaster tweet
non_disaster_tweets = df_train[df_train['target']==0]['text']
non_disaster_tweets.values[1]

# A disaster tweet
disaster_tweets = df_train[df_train['target']==1]['text']
disaster_tweets.values[1]

# Let's plot the wordcloud to see which words are more occuring
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[26, 8])
wordcloud1 = WordCloud( width=600,
                        height=400).generate(" ".join(disaster_tweets))
ax1.imshow(wordcloud1)
ax1.axis('off')
ax1.set_title('Disaster Tweets',fontsize=30);

wordcloud2 = WordCloud( width=600,
                        height=400).generate(" ".join(non_disaster_tweets))
ax2.imshow(wordcloud2)
ax2.axis('off')
ax2.set_title('Non Disaster Tweets',fontsize=30);
df_train['text'][66]

