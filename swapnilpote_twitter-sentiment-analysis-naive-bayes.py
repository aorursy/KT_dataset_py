import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os

import re

import nltk

import string



from nltk.corpus import stopwords
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_trn = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')

df_test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')



print('Training data shape: ', df_trn.shape)

print('Testing data shape: ', df_test.shape)
# First few rows of the training dataset

df_trn.head()
df_trn['sentiment'].unique()
# First few rows of the testing dataset

df_test.head()
#Missing values in training set

df_trn.isnull().sum()
#Missing values in test set

df_test.isnull().sum()
df_trn[df_trn.text.isnull()]
# Dropping missing values

df_trn.dropna(axis=0, inplace=True)



print('Training data shape: ', df_trn.shape)
# Positive tweet

print("Positive Tweet example :", df_trn[df_trn['sentiment']=='positive']['text'].values[0])

#negative_text

print("Negative Tweet example :", df_trn[df_trn['sentiment']=='negative']['text'].values[0])

#neutral_text

print("Neutral tweet example  :", df_trn[df_trn['sentiment']=='neutral']['text'].values[0])
# Distribution of the Sentiment Column



df_trn['sentiment'].value_counts()
# It'll be better if we could get a relative percentage instead of the count.



df_trn['sentiment'].value_counts(normalize=True)
categories = df_trn['sentiment'].value_counts(normalize=True)

plt.figure(figsize=(10,5))

sns.barplot(categories.index, categories.values, alpha=0.8)

plt.title('Distribution of Sentiment column in the training set')

plt.ylabel('Percentage', fontsize=12)

plt.xlabel('Category', fontsize=12)

plt.show()
# It'll be better if we could get a relative percentage instead of the count.



df_test['sentiment'].value_counts(normalize=True)
categories = df_test['sentiment'].value_counts(normalize=True)

plt.figure(figsize=(10,5))

sns.barplot(categories.index, categories.values, alpha=0.8)

plt.title('Distribution of Sentiment column in the testing set')

plt.ylabel('Percentage', fontsize=12)

plt.xlabel('Category', fontsize=12)

plt.show()
sentences = ' '.join(x.lower() for x in df_trn['text'].values)

print(sentences)
special_words = re.findall('[0-9a-z%s]+' % string.punctuation, sentences)

print(sorted(set(special_words)))
df_trn['text_clean'] = df_trn['text'].str.lower()



df_trn.head()
df_trn['text_clean'] = df_trn['text_clean'].apply(lambda x: re.sub('https?://\S+|www\.\S+', ' url ', x))



clean_sentences = ' '.join(x for x in df_trn['text_clean'].values)

special_words = re.findall('[0-9a-z%s]+' % string.punctuation, clean_sentences)

print(sorted(set(special_words)))
df_trn['text_clean'] = df_trn['text_clean'].apply(lambda x: re.sub('#[\w\d]+', ' hashtag ', x))



clean_sentences = ' '.join(x for x in df_trn['text_clean'].values)

special_words = re.findall('[0-9a-z%s]+' % string.punctuation, clean_sentences)

print(sorted(set(special_words)))
df_trn['text_clean'] = df_trn['text_clean'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x))



clean_sentences = ' '.join(x.lower() for x in df_trn['text_clean'].values)

special_words = re.findall('[0-9a-z%s]+' % string.punctuation, clean_sentences)

print(sorted(set(special_words)))
df_trn['text_clean'] = df_trn['text_clean'].apply(lambda x: re.sub('\n', ' ', x))

df_trn['text_clean'] = df_trn['text_clean'].apply(lambda x: re.sub('\w*\d\w*', ' number ', x))

df_trn['text_clean'] = df_trn['text_clean'].apply(lambda x: re.sub('\s+', ' ', x))



df_trn.head(20)
print(stopwords.words('english'))
print(nltk.tokenize.word_tokenize(df_trn['text_clean'][5]))
tk = nltk.tokenize.word_tokenize(df_trn['text_clean'][5])

rs = []

for w in tk:

    if w not in stopwords.words('english'):

        rs.append(w)



print(df_trn['text_clean'][5])

print(' '.join(rs))
def text_preprocessing(text):

    """

    Cleaning and parsing the text.



    """

    tokenized_text = nltk.tokenize.word_tokenize(text)

    remove_stopwords = [w for w in tokenized_text if w not in stopwords.words('english')]

    combined_text = ' '.join(remove_stopwords)

    return combined_text
# Applying the cleaning function to both test and training datasets

df_trn['text_clean'] = df_trn['text_clean'].apply(str).apply(lambda x: text_preprocessing(x))
df_trn.head()
# Number of words

df_trn['word_count'] = df_trn['text'].apply(lambda x: len(str(x).split(" ")))

df_trn[['text','word_count']].head()
# Number of char

df_trn['char_count'] = df_trn['text'].str.len() ## this also includes spaces

df_trn[['text','char_count']].head()
pos = df_trn[df_trn['sentiment']=='positive']

neg = df_trn[df_trn['sentiment']=='negative']

neutral = df_trn[df_trn['sentiment']=='neutral']
categories = pos['char_count'].value_counts(normalize=True)

plt.figure(figsize=(50, 25))

sns.barplot(categories.index, categories.values, alpha=0.8)

plt.title('Positive Text Length Distribution')

plt.ylabel('Count', fontsize=20)

plt.xlabel('Char length', fontsize=20)

plt.show()
categories = neg['char_count'].value_counts(normalize=True)

plt.figure(figsize=(50, 25))

sns.barplot(categories.index, categories.values, alpha=0.8)

plt.title('Negative Text Length Distribution')

plt.ylabel('Count', fontsize=20)

plt.xlabel('Char length', fontsize=20)

plt.show()
categories = neutral['char_count'].value_counts(normalize=True)

plt.figure(figsize=(50, 25))

sns.barplot(categories.index, categories.values, alpha=0.8)

plt.title('Neutral Text Length Distribution')

plt.ylabel('Count', fontsize=20)

plt.xlabel('Char length', fontsize=20)

plt.show()
# Average word length



def avg_word(sentence):

    words = sentence.split()

    return (sum(len(word) for word in words)/len(words))



df_trn['avg_word'] = df_trn['text'].apply(lambda x: avg_word(x))

df_trn[['text','avg_word']].head()
# Stop words present in tweet

stop = stopwords.words('english')



df_trn['stopwords'] = df_trn['text'].apply(lambda x: len([x for x in x.split() if x in stop]))

df_trn[['text','stopwords']].head()
# Hashtag present in tweet



df_trn['hastags'] = df_trn['text'].apply(lambda x: len([w for w in x.split() if w.startswith('#')]))

df_trn[['text','hastags']].head()
df_trn['numerics'] = df_trn['text'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))

df_trn[['text','numerics']].head()