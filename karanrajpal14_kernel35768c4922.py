import numpy as np

import pandas as pd



# text processing libraries

import re

import string

import nltk

from nltk.corpus import stopwords



# sklearn

from sklearn import model_selection

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB



# matplotlib and seaborn for plotting

import matplotlib.pyplot as plt

import seaborn as sns

from wordcloud import WordCloud



import random



nltk.download('stopwords')

nltk.download('wordnet')
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_csv = "/kaggle/input/nlp-getting-started/train.csv"

test_csv = "/kaggle/input/nlp-getting-started/test.csv"

submission_csv = "./submission.csv"
train = pd.read_csv(train_csv)

print('Training data shape: ', train.shape)

print(train.head())



print()



test = pd.read_csv(test_csv)

print('Testing data shape: ', test.shape)

print(test.head())
print("Missing values in training dataset")

print(train.isnull().sum(), end='\n\n')



print("Missing values in testing dataset")

print(test.isnull().sum(), end='\n\n')
print(train['target'].value_counts())



sns.barplot(train['target'].value_counts().index,

            train['target'].value_counts(), palette='Purples_r')
# Disaster tweet

disaster_tweets = train[train['target'] == 1]['text']

print(disaster_tweets.values[1])



# Not a disaster tweet

non_disaster_tweets = train[train['target'] == 0]['text']

print(non_disaster_tweets.values[1])
sns.barplot(y=train['keyword'].value_counts()[:20].index,

            x=train['keyword'].value_counts()[:20], orient='h', palette='Purples_r')
train.loc[train['text'].str.contains(

    'disaster', na=False, case=False)].target.value_counts()
train['location'].value_counts()
# Replacing the ambigious locations name with Standard names

train['location'].replace({'United States': 'USA',

                           'New York': 'USA',

                           "London": 'UK',

                           "Los Angeles, CA": 'USA',

                           "Washington, D.C.": 'USA',

                           "California": 'USA',

                           "Chicago, IL": 'USA',

                           "Chicago": 'USA',

                           "New York, NY": 'USA',

                           "California, USA": 'USA',

                           "FLorida": 'USA',

                           "Nigeria": 'Africa',

                           "Kenya": 'Africa',

                           "Everywhere": 'Worldwide',

                           "San Francisco": 'USA',

                           "Florida": 'USA',

                           "United Kingdom": 'UK',

                           "Los Angeles": 'USA',

                           "Toronto": 'Canada',

                           "San Francisco, CA": 'USA',

                           "NYC": 'USA',

                           "Seattle": 'USA',

                           "Earth": 'Worldwide',

                           "Ireland": 'UK',

                           "London, England": 'UK',

                           "New York City": 'USA',

                           "Texas": 'USA',

                           "London, UK": 'UK',

                           "Atlanta, GA": 'USA',

                           "Mumbai": "India"}, inplace=True)



sns.barplot(y=train['location'].value_counts()[:5].index,

            x=train['location'].value_counts()[:5], orient='h', palette='Purples_r')
train['location'].value_counts()
train['text'].head()
def clean_text(text):

    text = text.lower()

    text = re.sub('\[.*?\]', '', text)

    text = re.sub('https?://\S+|www\.\S+', '', text)

    text = re.sub('<.*?>+', '', text)

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub('\n', '', text)

    text = re.sub('\w*\d\w*', '', text)

    return text





# Applying the cleaning function to both test and training datasets

train['text'] = train['text'].apply(lambda x: clean_text(x))

test['text'] = test['text'].apply(lambda x: clean_text(x))



disaster_tweets = disaster_tweets.apply(lambda x: clean_text(x))

non_disaster_tweets = non_disaster_tweets.apply(lambda x: clean_text(x))



# Let's take a look at the updated text

print(train['text'].head())

print()

print(test['text'].head())
# Tokenizing the training and the test set

tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

train['text'] = train['text'].apply(lambda x: tokenizer.tokenize(x))

test['text'] = test['text'].apply(lambda x: tokenizer.tokenize(x))

train['text'].head()
def remove_stopwords(text):

    words = [w for w in text if w not in stopwords.words('english')]

    return words





train['text'] = train['text'].apply(lambda x: remove_stopwords(x))

print(train.head())

print()

test['text'] = test['text'].apply(lambda x: remove_stopwords(x))

print(test.head())
def purple_color_func(word, font_size, position, orientation, random_state=None,

                      **kwargs):

    return "hsl(270, 50%%, %d%%)" % random.randint(50, 60)





fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[26, 8])

wordcloud1 = WordCloud(background_color='white',

                       color_func=purple_color_func,

                       random_state=3,

                       width=600,

                       height=400).generate(" ".join(disaster_tweets))

ax1.imshow(wordcloud1)

ax1.axis('off')

ax1.set_title('Disaster Tweets', fontsize=40)



wordcloud2 = WordCloud(background_color='white',

                       color_func=purple_color_func,

                       random_state=3,

                       width=600,

                       height=400).generate(" ".join(non_disaster_tweets))

ax2.imshow(wordcloud2)

ax2.axis('off')

ax2.set_title('Non Disaster Tweets', fontsize=40)
def combine_text(list_of_text):

    combined_text = ' '.join(lemmatizer.lemmatize(token)

                             for token in list_of_text)

    return combined_text





lemmatizer = nltk.stem.WordNetLemmatizer()



train['text'] = train['text'].apply(lambda x: combine_text(x))

test['text'] = test['text'].apply(lambda x: combine_text(x))



print(train['text'].head())

print()

print(test['text'].head())
tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2))

train_tfidf = tfidf.fit_transform(train['text'])

test_tfidf = tfidf.transform(test["text"])



print(train_tfidf)

print()

print(test_tfidf)
# Fitting a simple Naive Bayes on TFIDF

clf_NB_TFIDF = MultinomialNB()

scores = model_selection.cross_val_score(

    clf_NB_TFIDF, train_tfidf, train["target"], cv=5, scoring="f1")

print(scores)
clf_NB_TFIDF.fit(train_tfidf, train["target"])
df = pd.DataFrame()

predictions = clf_NB_TFIDF.predict(test_tfidf)

df["id"] = test['id']

df["target"] = predictions

print(df)



df.to_csv(submission_csv, index=False)