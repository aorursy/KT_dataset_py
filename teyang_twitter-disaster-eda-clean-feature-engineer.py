import numpy as np 

import pandas as pd 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



tweets = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
tweets.head()

tweets.isnull().sum().plot(kind='bar')
import seaborn as sns

import matplotlib.pyplot as plt



color = [sns.xkcd_rgb['pale red'],sns.xkcd_rgb['medium blue']]

sns.countplot('target',data = tweets, palette = color)

plt.gca().set_ylabel('Samples')
#import nltk

#nltk.download('punkt')

from nltk import word_tokenize, sent_tokenize



# count number of characters in each tweet

tweets['char_len'] = tweets.text.str.len()



# count number of words in each tweet

word_tokens = [len(word_tokenize(tweet)) for tweet in tweets.text]

tweets['word_len'] = word_tokens



# count number of sentence in each tweet

sent_tokens = [len(sent_tokenize(tweet)) for tweet in tweets.text]

tweets['sent_len'] = sent_tokens



plot_cols = ['char_len','word_len','sent_len']

plot_titles = ['Character Length','Word Length','Sentence Length']



plt.figure(figsize=(20,4))

for counter, i in enumerate([0,1,2]):

    plt.subplot(1,3,counter+1)

    sns.distplot(tweets[tweets.target == 1][plot_cols[i]], label='Disaster', color=color[1]).set_title(plot_titles[i])

    sns.distplot(tweets[tweets.target == 0][plot_cols[i]], label='Non-Disaster', color=color[0])

    plt.legend()





# Investigate the Outliers



tweets[tweets.sent_len > 8]

tweets[tweets.word_len > 50]
## Plot most common stopwords



#nltk.download('stopwords')



from nltk.corpus import stopwords

stop = set(stopwords.words('english'))



# Get all the word tokens in dataframe for Disaster and Non-Disaster

corpus0 = [] # Non-Disaster

[corpus0.append(word.lower()) for tweet in tweets[tweets.target == 0].text for word in word_tokenize(tweet)]

corpus1 = [] # Disaster

[corpus1.append(word.lower()) for tweet in tweets[tweets.target == 1].text for word in word_tokenize(tweet)]



# Function for counting top stopwords in a corpus

def count_top_stopwords(corpus):

    stopwords_freq = {}

    for word in corpus:

        if word in stop: 

            if word in stopwords_freq:

                stopwords_freq[word] += 1

            else:

                stopwords_freq[word] = 1

    topwords = sorted(stopwords_freq.items(), key=lambda item: item[1], reverse=True)[:10] # get the top 10 stopwords

    x,y = zip(*topwords) # get key and values

    return x,y



x0,y0 = count_top_stopwords(corpus0)

x1,y1 = count_top_stopwords(corpus1)



# Plot bar plot of top stopwords for each class

plt.figure(figsize=(15,4))

plt.subplot(1,2,1)

plt.bar(x0,y0, color=color[0])

plt.title('Top Stopwords for Disaster Tweets')

plt.subplot(1,2,2)

plt.bar(x1,y1, color=color[1])

plt.title('Top Stopwords for  Non-Disaster Tweets')









## Plot most common punctuations



from string import punctuation



# Get all the punctuations in dataframe for Disaster and Non-Disaster

corpus0 = [] # Non-Disaster

[corpus0.append(c) for tweet in tweets[tweets.target == 0].text for c in tweet]

corpus0 = list(filter(lambda x: x in punctuation, corpus0)) # use filter to select only punctuations

corpus1 = [] # Disaster

[corpus1.append(c) for tweet in tweets[tweets.target == 1].text for c in tweet]

corpus1 = list(filter(lambda x: x in punctuation, corpus1)) 



from collections import Counter

x0,y0 = zip(*Counter(corpus0).most_common())

x1,y1 = zip(*Counter(corpus1).most_common())



# Plot bar plot of top punctuations for each class

plt.figure(figsize=(15,4))

plt.subplot(1,2,1)

plt.bar(x0,y0, color=color[0])

plt.title('Top Punctuations for Disaster Tweets')

plt.subplot(1,2,2)

plt.bar(x1,y1, color=color[1])

plt.title('Top Punctuations for Non-Disaster Tweets')







## Plot most common words

import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS



stop = ENGLISH_STOP_WORDS.union(stop) # combine stop words from different sources



# function for removing url from text

def remove_url(txt):

    return " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", txt).split())



# Get all the word tokens in dataframe for Disaster and Non-Disaster

# - remove url, tokenize tweet into words, lowercase words

corpus0 = [] # Non-Disaster

[corpus0.append(word.lower()) for tweet in tweets[tweets.target == 0].text for word in word_tokenize(remove_url(tweet))]

corpus0 = list(filter(lambda x: x not in stop, corpus0)) # use filter to unselect stopwords



corpus1 = [] # Disaster

[corpus1.append(word.lower()) for tweet in tweets[tweets.target == 1].text for word in word_tokenize(remove_url(tweet))]

corpus1 = list(filter(lambda x: x not in stop, corpus1)) # use filter to unselect stopwords



# Create df for word counts to use sns plots

a = Counter(corpus0).most_common()

df0 = pd.DataFrame(a, columns=['Word','Count'])



a = Counter(corpus1).most_common()

df1 = pd.DataFrame(a, columns=['Word','Count'])



# Plot for Disaster and Non-Disaster

plt.figure(figsize=(15,4))

plt.subplot(1,2,1)

sns.barplot(x='Word',y='Count',data=df0.head(10)).set_title('Most Common Words for Non-Disasters')

plt.xticks(rotation=45)

plt.subplot(1,2,2)

sns.barplot(x='Word',y='Count',data=df1.head(10)).set_title('Most Common Words for Disasters')

plt.xticks(rotation=45)









def clean(word):

    for p in punctuation: word = word.replace(p, '')

    return word



from wordcloud import WordCloud



def wc_hash(target):

    hashtag = [clean(w[1:].lower()) for tweet in tweets[tweets.target == target].text for w in tweet.split() if '#' in w and w[0] == '#']

    hashtag = ' '.join(hashtag)

    my_cloud = WordCloud(background_color='white', stopwords=stop).generate(hashtag)



    plt.subplot(1,2,target+1)

    plt.imshow(my_cloud, interpolation='bilinear') 

    plt.axis("off")



plt.figure(figsize=(15,4))

wc_hash(0)

plt.title('Non-Disaster')

wc_hash(1)

plt.title('Disaster')

from textblob import TextBlob



# polarity and subjectivity

tweets['polarity'] = [TextBlob(tweet).sentiment.polarity for tweet in tweets.text]

tweets['subjectivity'] = [TextBlob(tweet).sentiment.subjectivity for tweet in tweets.text]



#############################################################################################################################

# exclaimation and question marks

tweets['exclaimation_num'] = [tweet.count('!') for tweet in tweets.text]

tweets['questionmark_num'] = [tweet.count('?') for tweet in tweets.text]



#############################################################################################################################

# count number of hashtags and mentions

# Function for counting number of hashtags and mentions

def count_url_hashtag_mention(text):

    urls_num = len(re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text))

    word_tokens = text.split()

    hash_num = len([word for word in word_tokens if word[0] == '#' and word.count('#') == 1]) # only appears once in front of word 

    mention_num = len([word for word in word_tokens if word[0] == '@' and word.count('@') == 1]) # only appears once in front of word 

    return urls_num, hash_num, mention_num



url_num, hash_num, mention_num = zip(*[count_url_hashtag_mention(tweet) for tweet in tweets.text])

tweets = tweets.assign(url_num = url_num, hash_num = hash_num, mention_num = mention_num)



#############################################################################################################################

# count number of contractions

contractions = ["'t", "'re", "'s", "'d", "'ll", "'ve", "'m"]

tweets['contraction_num'] = [sum([tweet.count(cont) for cont in contractions]) for tweet in tweets.text]

tweets.head()
## Replace NaNs with 'None'

tweets.keyword.fillna('None', inplace=True) 



#############################################################################################################################

## Expand Contractions



# Function for expanding most common contractions https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python

def decontraction(phrase):

    # specific

    phrase = re.sub(r"won\'t", "will not", phrase)

    phrase = re.sub(r"can\'t", "can not", phrase)



    # general

    phrase = re.sub(r"n\'t", " not", phrase)

    phrase = re.sub(r"\'re", " are", phrase)

    phrase = re.sub(r"\'s", " is", phrase)

    phrase = re.sub(r"\'d", " would", phrase)

    phrase = re.sub(r"\'ll", " will", phrase)

    phrase = re.sub(r"\'t", " not", phrase)

    phrase = re.sub(r"\'ve", " have", phrase)

    phrase = re.sub(r"\'m", " am", phrase)

    return phrase



tweets.text = [decontraction(tweet) for tweet in tweets.text]



#############################################################################################################################

## Remove Emojis



# Reference : https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b

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



print(remove_emoji("OMG there is a volcano eruption!!! 😭😱😷"))



tweets.text = tweets.text.apply(lambda x: remove_emoji(x))
#############################################################################################################################

## Remove URLs

tweets.text = tweets.text.apply(lambda x: remove_url(x))



#############################################################################################################################

## Remove Punctuations except '!?'



def remove_punct(text):

    new_punct = re.sub('\ |\!|\?', '', punctuation)

    table=str.maketrans('','',new_punct)

    return text.translate(table)



tweets.text = tweets.text.apply(lambda x: remove_punct(x))



#############################################################################################################################

## Replace amp

def replace_amp(text):

    text = re.sub(r" amp ", " and ", text)

    return text



tweets.text = tweets.text.apply(lambda x: replace_amp(x))
## Plot word cloud for most common words after cleaning



def wc_words(target):

    words = [word.lower() for tweet in tweets[tweets.target == target].text for word in tweet.split()]

    words = list(filter(lambda w: w != 'like', words))

    words = list(filter(lambda w: w != 'new', words))

    words = list(filter(lambda w: w != 'people', words))

    words = ' '.join(words)

    my_cloud = WordCloud(background_color='white', stopwords=stop).generate(words)



    plt.subplot(1,2,target+1)

    plt.imshow(my_cloud, interpolation='bilinear') 

    plt.axis("off")



plt.figure(figsize=(15,4))

wc_words(0)

plt.title('Non-Disaster')

wc_words(1)

plt.title('Disaster')