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
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import iplot

import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud, STOPWORDS

import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud, STOPWORDS
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
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,plot_confusion_matrix
from sklearn.model_selection import train_test_split
import collections
df_train = pd.read_csv('/kaggle/input/asap-aes/training_set_rel3.tsv', sep='\t', encoding='ISO-8859-1')
df_test = pd.read_csv('/kaggle/input/asap-aes/test_set.tsv', sep='\t', encoding='ISO-8859-1')
df_train.head()
df_test.head()
# Getting more info about the test info

df_test.info()
df_train.shape
# Getting more info about the train dataset
df_train.info()
# Let's check how many values are none in each row of the train dataset
df_train.isnull().sum()
# Since a large part of the dataset has columns with more than 70-80 percent missing values so Deleting those columns
df_train.dropna(axis = 1, inplace = True)
df_train.head()
# Describing the train set
df_train.describe()
# Checking how many unique essay id were given
print(df_train['essay_set'].nunique())
df_train['essay_set'].unique()
# Counting the number of eacy essay_set

print(df_train.groupby('essay_set').size())

# Lets see the unique ratings which are being given by the rater1

print(df_train['rater1_domain1'].nunique())
df_train['rater1_domain1'].unique()
# Counting the number of rates of each rates given by rater1

print(df_train.groupby('rater1_domain1').size())
# Lets see the unique ratings which are being given by the rater2

print(df_train['rater2_domain1'].nunique())
df_train['rater2_domain1'].unique()
# Counting the number of rates of each rates given by rater1

print(df_train.groupby('rater1_domain1').size())
# Maximum domain score obtained by any essay

df_train['domain1_score'].max()
# Minimum domain score obtained by any essay

df_train['domain1_score'].min()
# Visualizing the percentage wise share of the various sets of essay 

labels = df_train['essay_set'].value_counts().index
values = df_train['essay_set'].value_counts().values

colors = df_train['essay_set']

fig = go.Figure(data = [go.Pie(labels = labels, values = values, textinfo = "label+percent",
                              marker = dict(colors = colors, line=dict(color='#000000', width=2)), 
                              title = "Distribution of sets of essay")])

fig.show()
# Visualizing the percentage wise share of top 10 grades given by rater 1 and their percentage

labels = df_train['rater1_domain1'].value_counts()[:10].index
values = df_train['rater1_domain1'].value_counts()[:10].values

colors = df_train['rater1_domain1']

fig = go.Figure(data = [go.Pie(labels = labels, values = values, textinfo = "label+percent",
                              marker = dict(colors = colors), 
                              title = "Top 10 grades given by rater 1 and their percentage")])

fig.show()
# Visualizing the percentage wise share of top 10 grades given by rater 1 and their percentage

labels = df_train['rater2_domain1'].value_counts()[:10].index
values = df_train['rater2_domain1'].value_counts()[:10].values

colors = df_train['rater2_domain1']

fig = go.Figure(data = [go.Pie(labels = labels, values = values, textinfo = "label+percent",
                              marker = dict(colors = colors), 
                              title = "Top 10 grades given by rater2 and their percentage")])

fig.show()
# Defining function to clean the text
def clean_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text
# Applying clean text function on short_description to clean the text of train set

df_train['essay'] = df_train['essay'].apply(lambda x: clean_text(x))
# Now checking whether the text of the essay columns have been changed or not

df_train.head()
word_cloud = WordCloud(
                       width=1600,
                       height=800, 
                       margin=0,
                       max_words=500, # Maximum numbers of words we want to see 
                       max_font_size=150, min_font_size=30,  # Font size range
                       background_color="white"
            ).generate(" ".join(df_train['essay']))

plt.figure(figsize=(10, 16))
plt.imshow(word_cloud, interpolation="gaussian")
plt.title('WordCloud of essay', fontsize = 40)
plt.axis("off")
plt.show()
# Applying clean text function on short_description to clean the text of test set

df_test['essay'] = df_test['essay'].apply(lambda x: clean_text(x))
word_cloud = WordCloud(
                       width=1600,
                       height=800, 
                       margin=0,
                       max_words=500, # Maximum numbers of words we want to see 
                       max_font_size=150, min_font_size=30,  # Font size range
                       background_color="white"
            ).generate(" ".join(df_test['essay']))

plt.figure(figsize=(10, 16))
plt.imshow(word_cloud, interpolation="gaussian")
plt.title('WordCloud of essay in test set', fontsize = 40)
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

# Tokenizing the training and test set

# instantiate the tokenizer class
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

# Tokenizing the training set
df_train['essay'] = df_train['essay'].apply(lambda x: tokenizer.tokenize(x))

# Tokenizing the test set
df_test['essay'] = df_test['essay'].apply(lambda x: tokenizer.tokenize(x))
# Printing the tokenized string of the training set
print()
print('Tokenized String:')
df_train['essay'].head()
# Printing the tokenized string of the testing set

print()
print('Tokenized String:')
df_test['essay'].head()
# Defining function to remove the stopwords

def remove_stopwords(text):
    
    words = [word for word in text if word not in stopwords.words('english')]
    return words
# Removing the stopwords from the training set

df_train['essay'] = df_train['essay'].apply(lambda x: remove_stopwords(x))
# Removing the stopwords from the test set

df_test['essay'] = df_test['essay'].apply(lambda x: remove_stopwords(x))
# lets now look at the training set

df_train.head()
# lets now look at the test set

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
# After preprocessing the text format

def combine_text(list_of_text):
    
    combined_text = ' '.join(list_of_text)
    return combined_text


# Preprocessing the train set

df_train['essay'] = df_train['essay'].apply(lambda x: combine_text(x))

df_train.head()
# Preprocessing the test set

df_test['essay'] = df_test['essay'].apply(lambda x: combine_text(x))
df_test.head()
# text preprocessing functions 
def text_preprocessing(text):
    
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    
    nopunc = clean_text(text)
    tokenized_text = tokenizer.tokenize(nopunc)
    remove_stopwords = [word for word in tokenized_text if word not in stopwords.words('english')]
    combined_text = ' '.join(remove_stopwords)
    return combined_text
# CountVectorizer can do all the above task of preprocessing, tokenization, and stop words removal

count_vectorizer = CountVectorizer()
train_vectors = count_vectorizer.fit_transform(df_train['essay'])
test_vectors = count_vectorizer.transform(df_test['essay'])

# Keeping only non-zero elements to preserve spaces
print(train_vectors[0].todense())
# TfidfVectorizer

tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2))
train_tfidf = tfidf.fit_transform(df_train['essay'])
test_tfidf = tfidf.transform(df_test['essay'])
# Let's implement simple classifiers

classifiers = {
    "LogisiticRegression": LogisticRegression(),
    "KNearest": KNeighborsClassifier(n_neighbors=1),
    "Support Vector Classifier": SVC(),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "MultinimialNB": MultinomialNB()
}
# Using the KNeighbors Classifiers

from sklearn.model_selection import cross_val_score

classifier = KNeighborsClassifier()

classifier.fit(train_vectors, df_train["domain1_score"])
training_score = cross_val_score(classifier, train_vectors, df_train["domain1_score"], cv=5)
print("Classifiers: ", classifier.__class__.__name__, "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score")
# Using the Logistic Regression

from sklearn.model_selection import cross_val_score

classifier2 = LogisticRegression()

classifier2.fit(train_vectors, df_train["domain1_score"])
training_score = cross_val_score(classifier2, train_vectors, df_train["domain1_score"], cv=5)
print("Classifiers: ", classifier2.__class__.__name__, "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score")
# Using the XGBoost

import xgboost as xgb
from sklearn import model_selection
clf_xgb = xgb.XGBClassifier(
    learning_rate=0.1,
    n_estimators=3000,
    max_depth=15,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softmax',
    nthread=42,
    scale_pos_weight=1,
    seed=27)

scores = model_selection.cross_val_score(clf_xgb, train_vectors, df_train["domain1_score"], cv=5, scoring="f1")
