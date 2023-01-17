# Imports
import os
import string
import re
import numpy as np
import pandas as pd
import plotly.express as px
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from wordcloud import WordCloud
# Read input data
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
submission = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')

# Print data sizes and preview
print('Train data size: ', train.shape)
print('Test data size: ', test.shape)
train.head(3)
print('Number of unique keywords: ', train['keyword'].nunique())
print('Percentage of tweets with keywords: ', len(train['keyword'].dropna())/ len(train) * 100)
print('Top keywords in disaster tweets:')
print(train[train['target'] == 1]['keyword'].value_counts().head(), '\n')
print('Top keywords in non-disaster tweets:')
print(train[train['target'] == 0]['keyword'].value_counts().head())
# Fix variations of USA and UK
train.loc[train['location'].isin(['US', 'United States', 'United States of America']), 'location'] = 'USA'
train.loc[train['location'].isin(['United Kingdom', 'The UK']), 'location'] = 'UK'

# Apply the same to the test set
test.loc[test['location'].isin(['US', 'United States', 'United States of America']), 'location'] = 'USA'
test.loc[test['location'].isin(['United Kingdom', 'The UK']), 'location'] = 'UK'

print('Number of unique locations: ', train['location'].nunique())
print('Top tweet locations:')
location_counts = train['location'].value_counts().head()
location_counts
# First, some data cleaning - lowercase all tweet text
train['text'] = train['text'].map(lambda text: text.lower())
test['text'] = test['text'].map(lambda text: text.lower())

# Find hastags with regex
train['hashtags'] = train['text'].map(lambda text: re.findall(r"#(\w+)", text))
test['hashtags'] = test['text'].map(lambda text: re.findall(r"#(\w+)", text))

# Remove hashtags symbols from the original text but keep the words
train['text'] = train['text'].map(lambda text: text.replace('#', ''))
test['text'] = test['text'].map(lambda text: text.replace('#', ''))

# Flatten hastags list and print most common
hashtags = pd.Series([tag for hashtags in train['hashtags'] for tag in hashtags])
print('Most common hashtags:')
hashtags.value_counts().head()
disaster_hashtags = pd.Series([tag for hashtags in train[train['target'] == 1]['hashtags'] for tag in hashtags])
wordcloud = WordCloud(background_color='white', width=500, height=300).generate(' '.join(disaster_hashtags))
print('Hashtags in disaster tweets:')
wordcloud.to_image()
non_disaster_hashtags = pd.Series([tag for hashtags in train[train['target'] == 0]['hashtags'] for tag in hashtags])
wordcloud = WordCloud(background_color='white', width=500, height=300).generate(' '.join(non_disaster_hashtags))
print('Hashtags in non-disaster tweets:')
wordcloud.to_image()
# Find mentions with regex
train['mentions'] = train['text'].map(lambda text: re.findall(r"@(\w+)", text))
test['mentions'] = test['text'].map(lambda text: re.findall(r"@(\w+)", text))

# Remove mention symbols from the original text but keep the words
train['text'] = train['text'].map(lambda text: text.replace('@', ''))
test['text'] = test['text'].map(lambda text: text.replace('@', ''))

# Flatter mentions list and print most common
mentions = pd.Series([mention for mentionslist in train['mentions'] for mention in mentionslist])
mentions.value_counts().head()
# Find URLs with regex
train['url'] = np.NaN
train['url_search'] = train['text'].map(lambda text: re.search('(?P<url>https?://[^\s]+)', text))
train.loc[~train['url_search'].isnull(), 'url'] = train[~train['url_search'].isnull()]['url_search'].map(lambda result: result.group('url'))
del train['url_search']

test['url'] = np.NaN
test['url_search'] = test['text'].map(lambda text: re.search('(?P<url>https?://[^\s]+)', text))
test.loc[~test['url_search'].isnull(), 'url'] = test[~test['url_search'].isnull()]['url_search'].map(lambda result: result.group('url'))
del test['url_search']

# Extract hostname from URLs
train.loc[~train['url'].isnull(),'host'] = train['url'].dropna().map(lambda s: s.split('://')[-1].split('/')[0].split('?')[0] if s else np.NaN)
test.loc[~test['url'].isnull(), 'host'] = test['url'].dropna().map(lambda s: s.split('://')[-1].split('/')[0].split('?')[0] if s else np.NaN)


# Remove URLs from tweet text
train['text'] = train['text'].map(lambda text: re.sub('(?P<url>https?://[^\s]+)', '', text))
test['text'] = test['text'].map(lambda text: re.sub('(?P<url>https?://[^\s]+)', '', text))

# Print most common
print('Most common URLs:')
train['host'].value_counts().head()
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Tokenize text, filter out punctuations and stopwords and lemmatize the words
def get_words(text):
    words = [word.translate(str.maketrans('', '', string.punctuation)) for word in word_tokenize(text)]
    words = [word for word in words if word.isalpha() and word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return words


train['words'] = train['text'].map(lambda text: get_words(text))
test['words'] = test['text'].map(lambda text: get_words(text))

# Print top words
words = pd.Series([word for wordlists in train['words'] for word in wordlists])
print('Most common words (lemmatized) :')
words.value_counts().head()
train.head(3)
# Vectorize the training set
word_vectorizer = CountVectorizer()
X_train = word_vectorizer.fit_transform(train['words'].map(lambda words: ', '.join(words)))

# Vectorize the testing test
X_test = word_vectorizer.transform(test['words'].map(lambda words: ', '.join(words)))

# Our output variable "target" which indicates whether a tweet is diaster tweet
y_train = train['target']

X_train.shape
clf = BernoulliNB()
scores = cross_val_score(clf, X_train, y_train)
print(scores.mean())
clf = BernoulliNB()
clf.fit(X_train, y_train)
y_test = clf.predict(X_test)
output = pd.DataFrame()
output['id'] = test['id']
output['target'] = y_test
output.to_csv('submission.csv', index=False)