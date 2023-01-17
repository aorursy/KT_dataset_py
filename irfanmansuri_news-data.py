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
df = pd.read_json('/kaggle/input/news-category-dataset/News_Category_Dataset_v2.json', lines = True)
df.head()
# Finding the shape of the data
df.shape
# GEtting more info about the data
df.info()
df.isnull().sum()
# Describing the data
df.describe()
# Counting the number of unique category and then priting them
print(df['category'].nunique())
df['category'].unique()
# Printing the number of words in each category
print(df.groupby('category').size())
# Trying to know about the year and month from the Data
import datetime
df['year'] = pd.DatetimeIndex(df['date']).year
df.head()
# Trying to add the month details
df['month'] = pd.DatetimeIndex(df['date']).month
df.head()
# Lets find out the unique values of months and year in the dataset
print('These data are of {} years'.format(df['year'].nunique()))
print("So below is the name of the year")
print(df['year'].unique())

print('These data are of {} months'.format(df['month'].nunique()))
print("So below is the name of the months")
print(df['month'].unique())
from sklearn.preprocessing import LabelEncoder

def category_merge(x):
    
    if x == 'THE WORLDPOST':
        return 'WORLDPOST'
    elif x == 'TASTE':
        return 'FOOD & DRINK'
    elif x == 'STYLE':
        return 'STYLE & BEAUTY'
    elif x == 'PARENTING':
        return 'PARENTS'
    elif x == 'COLLEGE':
        return 'EDUCATION'
    elif x == 'ARTS' or x == 'CULTURE & ARTS':
        return 'ARTS & CULTURE'
    
    else:
        return x
    
df['category'] = df['category'].apply(category_merge)
le = LabelEncoder()
data_labels = le.fit_transform(df['category'])
list(le.classes_)
# Counting the number of unique author and then printing the name
print(df['authors'].nunique())
df['authors'].unique()
# Removing the space between the authors name and writing them in simplified way
df['authors'] = df['authors'].apply(lambda x: x.split(',')[0])
df['authors'] = df['authors'].str.replace(' ', '', regex=False)
df['authors'].unique()
# Counting the contributions of different authors 
print(df.groupby('authors').size())
# Plotting to see the yearwise writing and publishing of the news

labels = df['year'].value_counts().index
values = df['year'].value_counts().values

colors = df['year']

fig = go.Figure(data = [go.Pie(labels = labels, values = values, textinfo = "label+percent",
                               marker = dict(colors = colors))])
fig.show()
# Plotting to see the monthwise pattern of writing and publishing of the news

labels = df['month'].value_counts().index
values = df['month'].value_counts().values

colors = df['month']

fig = go.Figure(data = [go.Pie(labels = labels, values = values, textinfo = "label+percent",
                               marker = dict(colors = colors))])
fig.show()
labels = df['category'].value_counts().index
values = df['category'].value_counts().values

colors = df['category']

fig = go.Figure(data = [go.Pie(labels = labels, values = values, textinfo = "label+percent",
                               marker = dict(colors = colors), pull=[0, 0, 0.2, 0]
       )])
fig.show()
# Visaulizing the same plot but withou using the Ploty

plt.figure(figsize=(20,20))
sizes = df['category'].value_counts().values
labels = df['category'].value_counts().index
plt.pie(sizes, labels=labels, autopct='%.1f%%',
        shadow=True, pctdistance=0.85, labeldistance=1.05, startangle=20, 
        explode = [0 if i > 0 else 0.2 for i in range(len(sizes))])
plt.axis('equal')
plt.show()
sns.barplot(y=df['category'].value_counts()[:5].index, x=df['category'].value_counts()[:5].values, orient='h')
data_labels
df['target'] = data_labels
df.head()
df['target'].unique()
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
# Applying clean text function on short_description to clean the text

df['short_description'] = df['short_description'].apply(lambda x: clean_text(x))
df.head()
# PLotting the Wordcloud

word_cloud = WordCloud(
                       width=1600,
                       height=800,
                       #colormap='PuRd', 
                       margin=0,
                       max_words=500, # Maximum numbers of words we want to see 
                        # Minimum numbers of letters of each word to be part of the cloud
                       max_font_size=150, min_font_size=30,  # Font size range
                       background_color="white").generate(" ".join(df['short_description']))

plt.figure(figsize=(10, 16))
plt.imshow(word_cloud, interpolation="gaussian")
plt.title('WordCloud of short_description', fontsize = 40)
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
# instantiate tokenizer class
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

# Tokenizing the trainig set
df['short_description'] = df['short_description'].apply(lambda x: tokenizer.tokenize(x))
print()
print('Tokenized string:')
df['short_description'].head()
nltk.download('stopwords')
# Definig a function to remove the stopwords

def remove_stopwords(text):
    
    words = [word for word in text if word not in stopwords.words('english')]
    return words
# Removing the stopwords 

df['short_description'] = df['short_description'].apply(lambda x: remove_stopwords(x))
df.head()
nltk.download('wordnet')
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

df['short_description'] = df['short_description'].apply(lambda x : combine_text(x))
df.head()
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
train_vectors = count_vectorizer.fit_transform(df['short_description'])
    
# Keeping only non-zero elements to preserve spaces
print(train_vectors[0].todense())
tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2))
train_tfidf = tfidf.fit_transform(df['short_description'])
# Let's implement simple classifiers

classifiers = {
    "LogisiticRegression": LogisticRegression(),
    "KNearest": KNeighborsClassifier(),
    "Support Vector Classifier": SVC(),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "MultinimialNB": MultinomialNB()
}
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(df['target'].values.reshape(-1,1))
scaler.transform(df['target'].values.reshape(-1,1))
# using headlines and short_description as input X

from keras.preprocessing.text import Tokenizer, text_to_word_sequence

df['text'] = df.headline + " " + df.short_description

# tokenizing

tokenizer = Tokenizer()
tokenizer.fit_on_texts(df.text)
X = tokenizer.texts_to_sequences(df.text)
df['words'] = X

# delete some empty and short data

df['word_length'] = df.words.apply(lambda i: len(i))
df = df[df.word_length >= 5]

df.head()
