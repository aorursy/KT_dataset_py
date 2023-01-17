#Import Packages

import numpy as np

import pandas as pd

import re

import matplotlib.pyplot as plt

import warnings

import nltk

import string

import seaborn as sns

from nltk.stem.porter import * 

from wordcloud import WordCloud



warnings.filterwarnings('ignore')

%matplotlib inline
#import data

train = pd.read_csv('../input/train_E6oV3lV.csv')

test = pd.read_csv('../input/test_tweets_anuFYb8.csv')

train.head(5)
test.head(5)


print('Shape of Train Dataset:',train.shape)

print('Shape of Test Dataset:',test.shape)



train[train['label'] == 0].head(5)

train[train['label'] == 1].head(5)
positive = train['label'].value_counts()[0]

negative = train['label'].value_counts()[1]



flatui = ["#15ff00", "#ff0033"]

sns.set_palette(flatui)

sns.barplot(['Positive','Negative'],[positive,negative])

plt.xlabel('Tweet Classification')

plt.ylabel('Count of Tweets')

plt.title('Balanced or Unbalanced Dataset')

plt.show()



print('No of Tweets labelled as Non-Sexist:',positive)

print('No of Tweets labelled as Sexist:',negative)



print('Data is highly unbalanced with only',round(((negative/(negative+positive))*100),2),'% negative points and ',

      round(((positive/(negative+positive))*100),2),'% positive points')
tweetLengthTrain = train['tweet'].str.len()

tweetLengthTest = test['tweet'].str.len()



plt.hist(tweetLengthTrain,bins=20,label='Train_Tweet')

plt.hist(tweetLengthTest,bins=20,label='Test_Tweet')

plt.legend()

plt.show()
#Comining both Train and Test Data Set before Data Cleaning

# since Label Column is not present in test Dataset, the values are filled with NaN

combine = train.append(test,ignore_index=True)

print('Shape of new Dataset:',combine.shape)

combine.tail()
#User Defined Function to clean unwanted text patterns from all tweets

# input - text to clean,pattern to replace

def cleantext(inputword,pattern):

    r = re.findall(pattern=pattern,string=inputword)

    for i in r:

        inputword = re.sub(pattern=i,repl='',string=inputword)

    return inputword

    
#Removing all twitter handles because they are already masked as @user due to privacy concerns.

#These twitter handles hardly give any information about the nature of the tweet.

combine['cleanedText'] = np.vectorize(cleantext)(combine['tweet'],'@[\w]*')

combine.head()

combine['cleanedText'] = combine['cleanedText'].str.replace("[^a-zA-Z#]"," ")

combine.head()



combine['cleanedText'] = combine['cleanedText'].str.replace(r'\b(\w{1,2})\b', '')

combine.head()
#Tokenize the tweets

tokenized_tweets = combine['cleanedText'].apply(lambda x:x.split())

tokenized_tweets.head()
#Stemming the words to remove words with similar meaning

stemmer = PorterStemmer()

tokenized_tweets = tokenized_tweets.apply(lambda x : [stemmer.stem(i) for i in x]  )
tokenized_tweets.head()
#Joining the tokenized tweets



for i in range(len(tokenized_tweets)):

    tokenized_tweets[i] = ' '.join(tokenized_tweets[i])    

combine['cleanedText'] = tokenized_tweets
combine.head()
# Creating word Cloud for all Words in all tweets

allWords = ' '.join([text for text in combine['cleanedText']])

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(allWords)

plt.figure(figsize=(10, 10)) 

plt.imshow(wordcloud, interpolation="bilinear") 

plt.axis('off') 

plt.show()
# Creating word Cloud for all Words in all positive tweets

positiveWords = ' '.join([text for text in combine['cleanedText'][combine['label'] == 0]])

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(positiveWords)

plt.figure(figsize=(10, 10)) 

plt.imshow(wordcloud, interpolation="bilinear") 

plt.axis('off') 

plt.show()
# Creating word Cloud for all Words in all negative tweets

positiveWords = ' '.join([text for text in combine['cleanedText'][combine['label'] == 1]])

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(positiveWords)

plt.figure(figsize=(10, 10)) 

plt.imshow(wordcloud, interpolation="bilinear") 

plt.axis('off') 

plt.show()
#Function Collecting HashTag

def collectHashtag(x):

    hashtags = []    

    for i in x:        

        ht = re.findall(r"#(\w+)", i)        

        hashtags.append(ht)     

    return hashtags
#Collect all the hashtags in positive and negative tweets

HT_positive = collectHashtag(combine['cleanedText'][combine['label'] == 0])

#Nested List to Un-nested List

HT_positive = sum(HT_positive,[])



HT_negative = collectHashtag(combine['cleanedText'][combine['label'] == 1])

HT_negative = sum(HT_negative,[])

corpus_positive = nltk.FreqDist(HT_positive)

corpus_negative = nltk.FreqDist(HT_negative)
d = pd.DataFrame({'Hashtag':list(corpus_positive.keys()),'Count':list(corpus_positive.values())})

d = d.nlargest(columns='Count',n=20)
ax = sns.barplot(data = d,x = 'Hashtag',y = 'Count')

plt.figure(figsize=(16,5))

plt.setp(ax.get_xticklabels(), rotation=90)

plt.show()
d = pd.DataFrame({'Hashtag':list(corpus_negative.keys()),'Count':list(corpus_negative.values())})

d = d.nlargest(columns='Count',n=20)



ax = sns.barplot(data = d,x = 'Hashtag',y = 'Count')

plt.figure(figsize=(16,5))

plt.setp(ax.get_xticklabels(), rotation=90)



plt.show()
#Vectorization

#Importing Required Packages



from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

import gensim
#Applying Bag of Words Vectorization to the Tweets

bow_vectorizer = CountVectorizer(stop_words= 'english')

bow = bow_vectorizer.fit_transform(combine['cleanedText'])

#Applying TF-IDF Vectorization to the Tweets

tfidf_vectorizer = TfidfVectorizer(stop_words= 'english')

tfidf = tfidf_vectorizer.fit_transform(combine['cleanedText'])