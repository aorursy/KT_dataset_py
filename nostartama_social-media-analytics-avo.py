import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#import mathematical & dataframe module

import numpy as np 

import pandas as pd



#import text module

import wordcloud as wc

import numpy as np

import textblob

import re, string, unicodedata

from bs4 import BeautifulSoup

from tqdm import tqdm

from nltk.corpus import wordnet

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from textblob import TextBlob

from textblob import Word





#import visualization module 

import matplotlib.pyplot as plt

import seaborn as sns; sns.set()

sns.set(style='darkgrid')

from wordcloud import WordCloud, STOPWORDS
tweet = pd.read_csv('/kaggle/input/twitter-airline-sentiment/Tweets.csv')

tweet.head()
#copy dataframe to tweet2

tweet2 = tweet.copy()
tweet_drop = tweet2.drop(columns=['tweet_id','tweet_created','tweet_location','user_timezone','tweet_coord'],axis=1)
tweet_drop.shape
#drop duplicates from data

tweet_drop.drop_duplicates(subset ="text",keep = False, inplace = True)
#check data variables

tweet_drop.info()
#data completeness in range 0-100

tweet_drop.count().sort_values(ascending = False) / len(tweet_drop)*100
tweet_drop['text'].head()
#cleansing 

def preprocessing(text):

    

    def removeUnicode(text):

        

        text = re.sub(r'(\\u[0-9A-Fa-f]+)','', text)       

        text = re.sub(r'[^\x00-\x7f]','',text)

        return text



    def replaceURL(text):

        

        text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',text)

        text = re.sub(r'#([^\s]+)', '', text)

        return text



    def replaceAtUser(text):

        

        text = re.sub('@[^\s]+','',text)

        return text



    def removeHashtagInFrontOfWord(text):

        

        text = re.sub(r'#([^\s]+)', r'\1', text)

        return text



    def removeNumbers(text):

        

        text = ''.join([i for i in text if not i.isdigit()])         

        return text



    def replaceMultiExclamationMark(text):

        

        text = re.sub(r"(\!)\1+", '', text)

        return text



    def replaceMultiQuestionMark(text):

        

        text = re.sub(r"(\?)\1+", '', text)

        return text



    def replaceMultiStopMark(text):

        

        text = re.sub(r"(\.)\1+", '', text)

        return text

    

    def removeEmoticons(text):

        

        text = re.sub(':\)|;\)|:-\)|\(-:|:-D|=D|:P|xD|X-p|\^\^|:-*|\^\.\^|\^\-\^|\^\_\^|\,-\)|\)-:|:\'\(|:\(|:-\(|:\S|T\.T|\.\_\.|:<|:-\S|:-<|\*\-\*|:O|=O|=\-O|O\.o|XO|O\_O|:-\@|=/|:/|X\-\(|>\.<|>=\(|D:', '', text)

        return text



    

    contraction_patterns = [ (r'won\'t', 'will not'), (r'can\'t', 'cannot'), (r'i\'m', 'i am'), (r'ain\'t', 'is not'), (r'(\w+)\'ll', '\g<1> will'), (r'(\w+)n\'t', '\g<1> not'),

                             (r'(\w+)\'ve', '\g<1> have'), (r'(\w+)\'s', '\g<1> is'), (r'(\w+)\'re', '\g<1> are'), (r'(\w+)\'d', '\g<1> would'), (r'&amp', ''), (r'dammit', 'damn it'), (r'dont', 'do not'), (r'wont', 'will not') ]

    def replaceContraction(text):

        patterns = [(re.compile(regex), repl) for (regex, repl) in contraction_patterns]

        for (pattern, repl) in patterns:

            (text, count) = re.subn(pattern, repl, text)

        return text



    def replaceElongated(word):

        



        repeat_regexp = re.compile(r'(\w*)(\w)\2(\w*)')

        repl = r'\1\2\3'

        if wordnet.synsets(word):

            return word

        repl_word = repeat_regexp.sub(repl, word)

        if repl_word != word:      

            return replaceElongated(repl_word)

        else:       

            return repl_word



    def removeEmoticons(text):

        

        text = re.sub(':\)|;\)|:-\)|\(-:|:-D|=D|:P|xD|X-p|\^\^|:-*|\^\.\^|\^\-\^|\^\_\^|\,-\)|\)-:|:\'\(|:\(|:-\(|:\S|T\.T|\.\_\.|:<|:-\S|:-<|\*\-\*|:O|=O|=\-O|O\.o|XO|O\_O|:-\@|=/|:/|X\-\(|>\.<|>=\(|D:', '', text)

        return text

    # Removes unicode strings like "\u002c" and "x96"

    text = removeUnicode(text)

    # Replaces url address with "url"

    text = replaceURL(text)

    # Removes hastag in front of a word

    text = replaceAtUser(text)

    # Replaces "@user"

    text = removeHashtagInFrontOfWord(text)

    # Removes integers 

    text = removeNumbers(text)

    # Replaces repetitions of exlamation marks

    text = replaceMultiExclamationMark(text)

    # Replaces repetitions of question marks

    text = replaceMultiQuestionMark(text)

    # Replaces repetitions of stop marks

    text = replaceMultiStopMark(text)

    # Removes emoticons from text

    text = removeEmoticons(text)

    # Replaces contractions from a string to their equivalents

    text = replaceContraction(text)

    # Replaces an elongated word with its basic form, unless the word exists in the lexicon

    text = replaceElongated(text)

    # Removes emoticons from text

    text = removeEmoticons(text)

    

    return text.lower()
tweet_drop['text'] = tweet_drop['text'].apply(preprocessing)

tweet_drop['text'].head()
tweet_drop['text'] = tweet_drop['text'].str.replace('[^\w\s]','')

tweet_drop['text'].head()
stop = stopwords.words('english')

tweet_drop['text'] = tweet_drop['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

tweet_drop['text'].head()
freq = pd.Series(' '.join(tweet_drop['text']).split()).value_counts()[:3]

freq
freq = list(freq.index)

tweet_drop['text'] = tweet_drop['text'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

tweet_drop['text'].head()
rare_word = pd.Series(' '.join(tweet_drop['text']).split()).value_counts()[-5000:]

rare_word
rare_word = list(rare_word.index)

tweet_drop['text'] = tweet_drop['text'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

tweet_drop['text'].head()
tweet_drop['text'][:10].apply(lambda x: str(TextBlob(x).correct()))
#Stemming



# st = PorterStemmer()

# tweet_drop['text'] = tweet_drop['text'][:].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))

# tweet_drop['text'].head()
#Lemmatization



# tweet_drop['text'] = tweet_drop['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

# tweet_drop['text'].head()
df = tweet_drop.copy()

df.head()
tweet['tweet_created'] = tweet.tweet_created.str[:10]

date = tweet['tweet_created'].unique().tolist()

date
fig = plt.figure(figsize=(8,6))

plt.title('Count Plot',fontsize = 20)

ax=sns.countplot(data=df, x='airline_sentiment',order = df['airline_sentiment'].value_counts().index)



plt.figure(figsize=(8,6))

sns.countplot(x=df["airline"])

plt.title("Airlines Distribution")



ax.set_xlabel('airline_sentiment', fontsize = 15)

ax.tick_params(labelsize=12)
def plot_sub_sentiment(Airline):

    data=df[df['airline']==Airline]

    count=data['airline_sentiment'].value_counts().index

    ax=sns.countplot(data=data, x='airline_sentiment',order = count)

    plt.title('Count Plot '+Airline,fontsize = 15)

    plt.ylabel('Sentiment Count')

    plt.xlabel('Mood')

    

plt.figure(1,figsize=(15, 15))

plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)

plt.subplot(231)

plot_sub_sentiment('US Airways')

plt.subplot(232)

plot_sub_sentiment('United')

plt.subplot(233)

plot_sub_sentiment('American')

plt.subplot(234)

plot_sub_sentiment('Southwest')

plt.subplot(235)

plot_sub_sentiment('Delta')

plt.subplot(236)

plot_sub_sentiment('Virgin America')
df.head()
three_airlines = df.copy()

three_airlines = three_airlines[(three_airlines['airline'] == 'US Airways') | (three_airlines['airline'] == 'United') | (three_airlines['airline'] == 'American')]

three_airlines = three_airlines[three_airlines['airline_sentiment'] == 'negative']
three_airlines.head()
plt.figure(figsize=(22, 5))

ax = sns.countplot(x="negativereason", hue="airline", data=three_airlines)

plt.xticks(rotation=15)

plt.ylabel('Sentiment Count')

plt.xlabel('Negative Reason')
#define happy and not happy 

sentiment_positive = df.loc[df['airline_sentiment'] == "positive"]

sentiment_neutral  = df.loc[df['airline_sentiment'] == "neutral"]

sentiment_negative = df.loc[df['airline_sentiment'] == "negative"]
sentiment_positive.head()
#merge all the happy comments into one paragraph

all_description_happy = "".join(sentiment_positive['text'].values)

all_description_neutral = "".join(sentiment_neutral['text'].values)

all_description_not_happy = "".join(sentiment_negative['text'].values)
def create_word_cloud(string):

    plt.figure(1,figsize=(10, 10))

    cloud = WordCloud(background_color = "white",width=1000,

                      height=500, max_words = 150, stopwords = set(STOPWORDS)).generate(string)

    plt.imshow(cloud, interpolation='bilinear')

    

    plt.axis('off')

    plt.show()


create_word_cloud(all_description_happy)
#neutral

create_word_cloud(all_description_neutral)
#not happy 

create_word_cloud(all_description_not_happy)