# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import nbinom
from collections import defaultdict

import random
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA , TruncatedSVD
from sklearn.metrics import classification_report, confusion_matrix

import string
import re
from wordcloud import WordCloud

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import xgboost as xgb


stop = set(stopwords.words('english'))

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dataset = pd.read_csv('../input/mbti-type/mbti_1.csv')
dataset.head()
dataset.isnull().sum()
plt.figure(figsize = (15,8))
sns.countplot(y = 'type' , data = dataset)
# Length of the text
plt.figure(figsize = (12,8))
plt.subplot(2,1,1)
dataset['length'] = dataset.posts.apply(lambda x:len(x))
plt.suptitle('Length of the Posts',fontsize = 30 , color = 'blue')
sns.distplot(dataset.length , kde=True , bins = 30 ).set(title = 'Distribution of Length of Posts')
plt.figure(figsize = (12,8))
plt.subplot(2,1,2)
sns.boxplot(dataset.length).set(title = 'Boxplot of Length of Posts')
plt.show()
# Length of words in each posts

words = dataset.posts.str.split().map(lambda x:len(x))
plt.figure(figsize = (12,5))
sns.distplot(words , kde = False ).set(title = 'Length of words in each post')
def CI(words):
    unbiased_point_estimate = np.mean(words)
    std = np.std(words)
    z_star = 1.96
    estimated_se = std/len(words)**0.5
    
    lcb = np.around(unbiased_point_estimate - z_star*estimated_se,decimals = 2)
    ucb = np.around(unbiased_point_estimate + z_star*estimated_se,decimals = 2)
    return (lcb,ucb)
    
CI(words)
mean_length_word = {}
confidence_interval = {}
def category_length(data , category):
    dx = dataset[dataset.type == category]
    words = dx['posts'].str.split().map(lambda x:len(x))
    mean_length_word[category] = np.around(np.mean(words),decimals = 2)
    confidence_interval[category] = CI(words)
    sns.distplot(words).set(title = 'Length of word in ' + category)
    plt.show()
categories = dataset.type.unique()
for i in categories:
    category_length(dataset,i)
confidence_interval
dx = dataset.groupby(['type'])['length'].apply(lambda x: np.mean(x))
dx
plt.figure(figsize = (12,5))
sns.barplot(x = list(dx.index) , y = dx.values) 
def create_corpus(data):
    corpus = []
    for text in data.posts.str.split():
        for i in text:
            corpus.append(i)
    return corpus
corpus = create_corpus(dataset)

dic = defaultdict(int)
for word in corpus:
    if word in stop:
        dic[word] += 1
    else:
        dic[word] = 1

top = sorted(dic.items() , key = lambda x:x[1] , reverse = True)[:15]

plt.figure(figsize = (15,5))
x , y = zip(*top)
sns.barplot(list(x) , list(y)).set(title = 'Frequency of Top 15 words in posts')
def category_top15(dataset,category):
    dx = dataset[dataset.type == category]
    corpus = create_corpus(dx)
    dic = defaultdict(int)
    for word in corpus:
        if word in stop:
            dic[word] += 1
        else:
            dic[word] = 1

    top = sorted(dic.items() , key = lambda x:x[1] , reverse = True)[:15]

    plt.figure(figsize = (15,5))
    x , y = zip(*top)
    sns.barplot(list(x) , list(y)).set(title = 'Frequency of Top 15 words in this '+ category)

categories = dataset.type.unique()
for i in categories:
    category_top15(dataset,i)
plt.figure(figsize = (15,8))
corpus = create_corpus(dataset)
dic = defaultdict(int)
punctuation = string.punctuation

for word in corpus:
    if word in punctuation:
        dic[word] += 1
        
top = sorted(dic.items() , key = lambda x:x[1] , reverse = True)[:15]

x,y = zip(*top)
sns.barplot(list(x) , list(y)).set(title = 'Barplot for top 15 punctuation in posts')
def get_top_tweet_bigrams(corpus,n = None):
    vec = CountVectorizer(ngram_range=(2,2)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_of_words = bag_of_words.sum(axis = 0)
    words_freq = [(word,sum_of_words[0,idx]) for word,idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq , key = lambda x:x[1] , reverse = True)
    return words_freq[:n]
plt.figure(figsize=(16,5))
top_tweet_bigrams=get_top_tweet_bigrams(dataset.posts)[:10]
x,y=map(list,zip(*top_tweet_bigrams))
sns.barplot(x=y,y=x)
def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def combine_text(text):
    return ' '.join(text)

def text_preprocessing(text):
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    no_punc = clean_text(text)
    tokenized_text = tokenizer.tokenize(no_punc)
    remove_stopwords = [w for w in tokenized_text if w not in stopwords.words('english')]
    combined_text = combine_text(remove_stopwords)
    
    return combined_text
dataset['clean_posts'] = dataset['posts'].apply(lambda x: text_preprocessing(x))
mbti = {'I':'Introversion', 'E':'Extroversion', 'N':'Intuition', 
        'S':'Sensing', 'T':'Thinking', 'F': 'Feeling', 
        'J':'Judging', 'P': 'Perceiving'}
dataset['description'] = dataset.type.apply(lambda x:' '.join([mbti[l] for l in list(x)]))
dataset['clean_text_length'] = dataset.clean_posts.apply(lambda x:len(x))
def Word_Cloud(dataset , category):
    fig , ax1 = plt.subplots(1 , 1 , figsize = [26,8])
    dx = dataset[dataset.type == category]['clean_posts']
    wordcloud1 = WordCloud(background_color = 'black' , width = 600 , height = 400).generate(" ".join(dx))
    ax1.imshow(wordcloud1)
    ax1.axis('off')
    ax1.set_title('Wordcloud for posts '+ category , fontsize = 20)
for i in categories:
    Word_Cloud(dataset,i)
from sklearn.preprocessing import LabelEncoder
dx = dataset[['clean_posts','type']]
encoder = LabelEncoder()
dx['type_enc'] = encoder.fit_transform(dx.type)
dx.head()
category = list(encoder.classes_)
train_data = dx.iloc[:6940,:]
test_data = dx.iloc[6940:,]
train_data.shape,test_data.shape
train_data = train_data.dropna()
train_data.isnull().sum()
count_vectorizer = CountVectorizer()
train_vectors = count_vectorizer.fit_transform(train_data.clean_posts)
test_vectors = count_vectorizer.transform(test_data.clean_posts)
tfidf_vectorizer = TfidfVectorizer(min_df = 2, max_df = 0.5, ngram_range = (1 , 2))
train_tfidf = tfidf_vectorizer.fit_transform(train_data.clean_posts)
test_tfidf = tfidf_vectorizer.transform(test_data.clean_posts)
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(class_weight='balanced' , C = 0.005)
score = model_selection.cross_val_score(clf , train_vectors , train_data['type_enc'] , cv = 5 , scoring = 'f1_micro')
score
clf_xgb = xgb.XGBClassifier(max_depth = 7 , n_estimators = 200 , colsample_bytree = 0.8 , subsample = 0.8 , nthread = 10 , learning_rate = 0.1)
scores = model_selection.cross_val_score(clf_xgb , train_vectors , train_data['type_enc'] , cv = 5 , scoring = 'f1_micro')
scores
clf_xgb.fit(train_vectors , train_data.type_enc)
y_pred = clf_xgb.predict(test_vectors)
y_test = test_data.type_enc
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)
clf_xgb.predict(test_vectors)
cm = confusion_matrix(y_pred,y_test)

plt.figure(figsize = (20,10))
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(category); ax.yaxis.set_ticklabels(category);

