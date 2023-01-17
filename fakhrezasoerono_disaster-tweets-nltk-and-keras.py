import numpy as np

import pandas as pd
df = pd.read_csv('../input/nlp-getting-started/train.csv', encoding='utf-8')

df_new = pd.read_csv('../input/nlp-getting-started/test.csv', encoding='utf-8')

df['set'] = 'train'

df_new['set'] = 'pred'

df = df.drop(columns='id')

df_new = df_new.drop(columns='id')

data = pd.concat([df, df_new]) # concatenate for text preprocessing

data.reset_index(inplace=True)

print(df.info())

print(df.head())
print(df.target.value_counts(normalize=True))

df.target.value_counts(normalize=True).plot(kind='bar')
# Convert text to lowercase

data.text = data.text.str.lower()
# Remove/replace unwanted characters

data.text = data.text.str.replace('ûª', "'") # replace with '

data.text = data.text.str.replace('', " ")

data.text = data.text.str.replace('û', " ")

data.text = data.text.str.replace('ò', " ")

data.text = data.text.str.replace('ó', " ")

data.text = data.text.str.replace('ï', " ")

data.text = data.text.str.replace('ì', " ")

data.text = data.text.str.replace('÷', " ")

data.text = data.text.str.replace('åê', " ")

data.text = data.text.str.replace('##', "#")
# Contractions Expander

import re

contractions_dict = { 

"ain't": "am not / are not / is not / has not / have not",

"aren't": "are not / am not",

"can't": "cannot",

"can't've": "cannot have",

"'cause": "because",

"could've": "could have",

"couldn't": "could not",

"couldn't've": "could not have",

"didn't": "did not",

"doesn't": "does not",

"don't": "do not",

"hadn't": "had not",

"hadn't've": "had not have",

"hasn't": "has not",

"haven't": "have not",

"he'd": "he had / he would",

"he'd've": "he would have",

"he'll": "he shall / he will",

"he'll've": "he shall have / he will have",

"he's": "he has / he is",

"how'd": "how did",

"how'd'y": "how do you",

"how'll": "how will",

"how's": "how has / how is / how does",

"i'd": "i had / i would",

"i'd've": "i would have",

"i'll": "i shall / i will",

"i'll've": "i shall have / i will have",

"i'm": "i am",

"i've": "i have",

"isn't": "is not",

"it'd": "it had / it would",

"it'd've": "it would have",

"it'll": "it shall / it will",

"it'll've": "it shall have / it will have",

"it's": "it has / it is",

"let's": "let us",

"ma'am": "madam",

"mayn't": "may not",

"might've": "might have",

"mightn't": "might not",

"mightn't've": "might not have",

"must've": "must have",

"mustn't": "must not",

"mustn't've": "must not have",

"needn't": "need not",

"needn't've": "need not have",

"o'clock": "of the clock",

"oughtn't": "ought not",

"oughtn't've": "ought not have",

"shan't": "shall not",

"sha'n't": "shall not",

"shan't've": "shall not have",

"she'd": "she had / she would",

"she'd've": "she would have",

"she'll": "she shall / she will",

"she'll've": "she shall have / she will have",

"she's": "she has / she is",

"should've": "should have",

"shouldn't": "should not",

"shouldn't've": "should not have",

"so've": "so have",

"so's": "so as / so is",

"that'd": "that would / that had",

"that'd've": "that would have",

"that's": "that has / that is",

"there'd": "there had / there would",

"there'd've": "there would have",

"there's": "there has / there is",

"they'd": "they had / they would",

"they'd've": "they would have",

"they'll": "they shall / they will",

"they'll've": "they shall have / they will have",

"they're": "they are",

"they've": "they have",

"to've": "to have",

"wasn't": "was not",

"we'd": "we had / we would",

"we'd've": "we would have",

"we'll": "we will",

"we'll've": "we will have",

"we're": "we are",

"we've": "we have",

"weren't": "were not",

"what'll": "what shall / what will",

"what'll've": "what shall have / what will have",

"what're": "what are",

"what's": "what has / what is",

"what've": "what have",

"when's": "when has / when is",

"when've": "when have",

"where'd": "where did",

"where's": "where has / where is",

"where've": "where have",

"who'll": "who shall / who will",

"who'll've": "who shall have / who will have",

"who's": "who has / who is",

"who've": "who have",

"why's": "why has / why is",

"why've": "why have",

"will've": "will have",

"won't": "will not",

"won't've": "will not have",

"would've": "would have",

"wouldn't": "would not",

"wouldn't've": "would not have",

"y'all": "you all",

}

c_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))

def expandContractions(text, c_re=c_re):

    def replace(match):

        return contractions_dict[match.group(0)]

    return c_re.sub(replace, text)
import re

import nltk

import string

from nltk.corpus import stopwords

from nltk.tokenize.casual import TweetTokenizer



# Tweet-tokenizer function

def tweet_func(dataframe):

    """Tokenize tweet, stop words removal, and punctuation removal from 'text' column of dataframe"""

    tweets = dataframe.text # tweets are in df.text

    stop_words = set(stopwords.words('english'))

    tweet_tokenizer = TweetTokenizer() # nltk.tokenize.word_tokenize also works!!

    tweet_clean = []

    n_token = []

    for text in tweets:

        text = re.sub('#\w+', '', text) # Remove hashtags

        text = re.sub('@\w+', '', text) # Remove mentions

        text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text) # Remove url

        token = tweet_tokenizer.tokenize(text) # Tweet Tokenizing

        token = [word for word in token if not word in stop_words] # Stop Words Removal

        token = [''.join(s) for s in token if not s in string.punctuation] # Punctuation Removal

        tweet_clean.append(' '.join(token))

        n_token.append(len(token)) # Word (Token) Count

    return (tweet_clean, n_token)



# Hashtag finder & counter

def hashtag_func(dataframe):

    """Hashtag finder & counter from 'text' column of dataframe"""

    tweets = dataframe.text # tweets are in df.text

    hashtag_list = []

    hashtag_count = []

    for text in tweets:

        hashtag = re.findall('#\w+', text)

        hashtag_list.append(' '.join(hashtag))

        hashtag_count.append(len(hashtag))

    return (hashtag_list, hashtag_count)



# Mention finder & counter

def mention_func(dataframe):

    """Mention finder & counter from 'text' column of dataframe"""

    tweets = dataframe.text # tweets are in df.text

    mention_list = []

    mention_count = []

    for text in tweets:

        mention = re.findall('@\w+', text)

        mention_list.append(' '.join(mention))

        mention_count.append(len(mention))

    return (mention_list, mention_count)



# Url finder & counter

def url_func(dataframe):

    """URL finder & counter from 'text' column of dataframe"""

    tweets = dataframe.text # tweets are in df.text

    url_list = []

    url_count = []

    for text in tweets:

        url = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)

        url_list.append(' '.join(url))

        url_count.append(len(url))

    return (url_list, url_count)
data.text = data.text.apply(expandContractions)

tweet_token, n_token = tweet_func(data)

hashtag, n_hashtag = hashtag_func(data)

url, n_url = url_func(data)

mention, n_mention = mention_func(data)

print('Functions applied!')
# Text Vectorizing

from sklearn.feature_extraction.text import CountVectorizer



# Hashtag Vectorizer

cv_hashtag = CountVectorizer(token_pattern='#\w+')

hashtag_vector_train = cv_hashtag.fit_transform(hashtag[0:len(df)])

hashtag_vector_test = cv_hashtag.transform(hashtag[len(df):])

print('hashtag vector:')

print(type(hashtag_vector_train))

print(hashtag_vector_train.shape, '\n')



# Mention Vectorizer

cv_mention = CountVectorizer(token_pattern='@\w+')

mention_vector_train = cv_mention.fit_transform(mention[0:len(df)])

mention_vector_test = cv_mention.transform(mention[len(df):])

print('mention vector:')

print(type(mention_vector_train))

print(mention_vector_train.shape, '\n')



# Url Vectorizer

cv_url = CountVectorizer(token_pattern='http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

url_vector_train = cv_url.fit_transform(url[0:len(df)])

url_vector_test = cv_url.transform(url[len(df):])

print('url vector:')

print(type(url_vector_train))

print(url_vector_train.shape, '\n')



# Tweet Vectorizer

cv_tweet = CountVectorizer()

tweet_vector_train = cv_tweet.fit_transform(tweet_token[0:len(df)])

tweet_vector_test = cv_tweet.transform(tweet_token[len(df):])

print('tweet vector:')

print(type(tweet_vector_train))

print(tweet_vector_train.shape, '\n')
# Concatenate text vectors

from scipy.sparse import hstack



X = hstack([hashtag_vector_train, mention_vector_train, url_vector_train, tweet_vector_train])

X = X.todense()

print('Training features:')

print(type(X))

print(X.shape)



X_pred = hstack([hashtag_vector_test, mention_vector_test, url_vector_test, tweet_vector_test])

X_pred = X_pred.todense()

print('Prediction features:')

print(type(X_pred))

print(X_pred.shape)
from keras.utils.np_utils import to_categorical

y = to_categorical(df.target)

print(type(y))

print(y.shape)
input_shape = (X.shape[1],)
from keras.layers import Dense

from keras.models import Sequential

def new_model(input_shape=input_shape):

    model = Sequential()

    model.add(Dense(100, activation='relu', input_shape=input_shape))

    model.add(Dense(100, activation='relu'))

    model.add(Dense(2, activation='softmax'))

    return(model)
model = new_model(input_shape)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X, y, validation_split=0.25, epochs = 3)
model = new_model(input_shape)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X, y)
predictions = model.predict(X_pred)

print('First 5 predictions:')

print(predictions[:4])
results = []

for proba in predictions:

    if proba[0] > 0.5:

        results.append(int(0))

    else:

        results.append(int(1))

results = pd.Series(results)

print('Predicted class ratio:')

print(results.value_counts(normalize=True))
submission = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')

print(submission.head())

submission.target = results

submission.set_index('id', inplace=True)

print(submission.head())
submission.to_csv('submission.csv')

print('Submission saved!')