# Importing the necessary library


import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px 
import plotly.graph_objs as go
from plotly.offline import iplot

import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud, STOPWORDS
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,plot_confusion_matrix
from sklearn.model_selection import train_test_split
from string import punctuation
from bs4 import BeautifulSoup
import re,string,unicodedata
from nltk import pos_tag
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
from keras.layers import Dense
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import collections
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
df_train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
df_test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
df_submission = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
df_train.head()
df_test.head()
df_submission.head()
df_train.info()
df_test.info()
df_train['keyword'].unique()
# Counting the number of unique keywords in trainset

df_train['keyword'].nunique()
# Counting the number of unique keywords in test dataset

df_test['keyword'].nunique()
df_train['target'].value_counts()
labels = df_train['target'].value_counts()[:].index
values = df_train['target'].value_counts()[:].values

colors=['#2678bf', '#98adbf']

fig = go.Figure(data=[go.Pie(labels = labels, values=values, textinfo="label+percent",
                            insidetextorientation="radial", marker=dict(colors=colors))])

fig.show()
real = df_train[df_train['target'] == 1]['text']
real.values[1:5]
fake = df_train[df_train['target'] == 0]['text']
fake.values[1:5]
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
                            "Mumbai":"India"},inplace=True)
bottom_5_location = df_train.sort_values("location", ascending=True).head(5)

fig = px.bar(bottom_5_location,
            x = "keyword",
            y="location",
            orientation='v',
            height=800,
            title="Bottom 5 location ",
            color="keyword"
            )

fig.show()
labels = df_train[df_train['target'] == 1]['keyword'].value_counts()[:10].index
values = df_train[df_train['target'] == 1]['keyword'].value_counts()[:10].values

colors = df_train['keyword']

fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent', 
                             insidetextorientation='radial', marker=dict(colors=colors))])
fig.show()
labels = df_train[df_train['target'] == 0]['keyword'].value_counts()[:10].index
values = df_train[df_train['target'] == 0]['keyword'].value_counts()[:10].values

colors = df_train['keyword']

fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent', 
                             insidetextorientation='radial', marker=dict(colors=colors))])
fig.show()
labels = df_train['location'].value_counts()[:10].index
values = df_train['location'].value_counts()[:10].values

colors = df_train['location']

fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent', 
                             insidetextorientation='radial', marker=dict(colors=colors))])
fig.show()
df_train['id'].nunique()
sns.barplot(y=df_train['location'].value_counts()[:5].index, x=df_train['location'].value_counts()[:5], orient='h')
# Let's have a look at both the trainig and test set data
df_train['text'][:5]
df_test['text'][:5]
def clean_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text
    
# Now applying clean_text function to both train and test datasets

df_train['text'] = df_train['text'].apply(lambda x: clean_text(x))
df_test['text'] = df_test['text'].apply(lambda x: clean_text(x))
# Let's see how has been our train and test datasets have been changed after applying the clean_text function

df_train['text'].head()
df_test['text'].head()

word_cloud = WordCloud(
                       width=1600,
                       height=800,
                       #colormap='PuRd', 
                       margin=0,
                       max_words=500, # Maximum numbers of words we want to see 
                       min_word_length=3, # Minimum numbers of letters of each word to be part of the cloud
                       max_font_size=150, min_font_size=30,  # Font size range
                       background_color="white").generate(" ".join(real))

plt.figure(figsize=(10, 16))
plt.imshow(word_cloud, interpolation="gaussian")
plt.title('Real Tweets mentioning about Disaster', fontsize = 40)
plt.axis("off")
plt.show()
word_cloud = WordCloud(
                       width=1600,
                       height=800,
                       colormap='PuRd', 
                       margin=0,
                       max_words=500, # Maximum numbers of words we want to see 
                       min_word_length=3, # Minimum numbers of letters of each word to be part of the cloud
                       max_font_size=150, min_font_size=30,  # Font size range
                       background_color="white").generate(" ".join(fake))

plt.figure(figsize=(10, 16))
plt.imshow(word_cloud, interpolation="gaussian")
plt.title('Fake Tweets mentioning about Disaster', fontsize = 40)
plt.axis("off")
plt.show()
print()
text = "I love you, don't you"

# instantiate tokenizer class
tokenizer1 = nltk.tokenize.WhitespaceTokenizer()
tokenizer2 = nltk.tokenize.TreebankWordTokenizer()
tokenizer3 = nltk.tokenize.WordPunctTokenizer()
tokenizer4 = nltk.tokenize.RegexpTokenizer(r'\w+')

print("Example Text: ", text)
print("Tokenization by whitespace: ", tokenizer1.tokenize(text))
print("Tokenization by words using Treebank Word Tokenizer: ", tokenizer2.tokenize(text))
print("Tokenization by punctuation: ", tokenizer3.tokenize(text))
print("Tokenization by regular expression: ", tokenizer4.tokenize(text))
# Tokenizing the training and the test set

# instantiate tokenizer class
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

# Tokenizing the trainig set
df_train['text'] = df_train['text'].apply(lambda x: tokenizer.tokenize(x))
df_test['text'] = df_test['text'].apply(lambda x: tokenizer.tokenize(x))
print()
print('Tokenized string:')
df_train['text'].head()
print()
print('Tokenized string:')
df_test['text'].head()
# Definig a function to remove the stopwords

def remove_stopwords(text):
    
    words = [word for word in text if word not in stopwords.words('english')]
    return words
# Removing the stopwords from the train and test set

df_train['text'] = df_train['text'].apply(lambda x: remove_stopwords(x))
df_test['train'] = df_test['text'].apply(lambda x: remove_stopwords(x))
df_train.head()
df_test.head()
# Stemming and Lemmatization examples

text = "How is the Josh"

tokenizer = nltk.tokenize.TreebankWordTokenizer()
tokens = tokenizer.tokenize(text)

# Stemmer 
stemmer = nltk.stem.PorterStemmer()
print("Stemming the sentence: ", " ".join(stemmer.stem(token) for token in tokens))

# Lemmatizer
lemmatizer = nltk.stem.WordNetLemmatizer()
print("Lemmatizing the sentence: ", " ".join(lemmatizer.lemmatize(token) for token in tokens))
# After preprocessing, the text format
def combine_text(list_of_text):
    
    combined_text = ' '.join(list_of_text)
    return combined_text

df_train['text'] = df_train['text'].apply(lambda x : combine_text(x))
df_test['text'] = df_test['text'].apply(lambda x : combine_text(x))
df_train['text']
df_train.head()
# text preprocessing function
def text_preprocessing(text):
   
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    
    nopunc = clean_text(text)
    tokenized_text = tokenizer.tokenize(nopunc)
    remove_stopwords = [word for word in tokenized_text if word not in stopwords.words('english')]
    combined_text = ' '.join(remove_stopwords)
    return combined_text

# CountVectorizer can do all the above task of preprocessing, tokenization, and stop words removal
count_vectorizer = CountVectorizer()
train_vectors = count_vectorizer.fit_transform(df_train['text'])
test_vectors = count_vectorizer.transform(df_test['text'])
    
    
# Keeping only non-zero elements to preserve spaces
print(train_vectors[0].todense())
tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2))
train_tfidf = tfidf.fit_transform(df_train['text'])
test_tfidf = tfidf.transform(df_test['text'])
# Let's implement simple classifiers

classifiers = {
    "LogisiticRegression": LogisticRegression(),
    "KNearest": KNeighborsClassifier(),
    "Support Vector Classifier": SVC(),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "MultinimialNB": MultinomialNB()
}
# Wow our scores are getting even high scores even when applying cross validation.
# Lets apply the Classifiers 1st on Countvectoizers
from sklearn.model_selection import cross_val_score


for key, classifier in classifiers.items():
    classifier.fit(train_vectors, df_train["target"])
    training_score = cross_val_score(classifier, train_vectors, df_train["target"], cv=5)
    print("Classifiers: ", classifier.__class__.__name__, "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score")
# Lets apply the Classifiers tfidf
from sklearn.model_selection import cross_val_score


for key, classifier in classifiers.items():
    classifier.fit(train_tfidf, df_train["target"])
    training_score = cross_val_score(classifier, train_tfidf, df_train["target"], cv=5, scoring="f1")
    print("Classifiers: ", classifier.__class__.__name__, "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score")
# Fitting a simple Naive Bayes on TFIDF
from sklearn import model_selection
clf_NB_TFIDF = MultinomialNB()
scores = model_selection.cross_val_score(clf_NB_TFIDF, train_tfidf, df_train["target"], cv=5, scoring="f1")
scores
clf_NB_TFIDF.fit(train_tfidf, df_train["target"])
import xgboost as xgb
from sklearn import model_selection
clf_xgb = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 
                           subsample=0.8, nthread=10, learning_rate=0.01)

scores = model_selection.cross_val_score(clf_xgb, train_vectors, df_train["target"], cv=5, scoring="f1")
import xgboost as xgb
clf_xgb_TFIDF = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 
                        subsample=0.8, nthread=10, learning_rate=0.1)
scores = model_selection.cross_val_score(clf_xgb_TFIDF, train_tfidf, df_train["target"], cv=5, scoring="f1")
scores
def submission(submission_file_path,model,test_vectors):
    sample_submission = pd.read_csv(submission_file_path)
    sample_submission["target"] = model.predict(test_vectors)
    sample_submission.to_csv("submission.csv", index=False)
submission_file_path = "../input/nlp-getting-started/sample_submission.csv"
test_vectors=test_tfidf
submission(submission_file_path,clf_NB_TFIDF,test_vectors)
