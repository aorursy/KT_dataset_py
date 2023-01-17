# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from collections import Counter
import numpy as np 
import string
import re
from datetime import datetime
from matplotlib import pyplot
import spacy
import seaborn as sns

import xgboost as xgb
from tqdm import tqdm
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics import classification_report
stop_words = stopwords.words('english')

%matplotlib inline
#Reading the data
Answers = pd.read_csv("../input/Answers.csv" , encoding='latin-1')
Questions = pd.read_csv("../input/Questions.csv", encoding='latin-1')
Tags = pd.read_csv("../input/Tags.csv", encoding='latin-1')

#Cleaning the data
def clean_text(text):
    global EMPTY
    EMPTY = ''
    
    if not isinstance(text, str): 
        return text
    text = re.sub('<pre><code>.*?</code></pre>', EMPTY, text)

    def replace_link(match):
        return EMPTY if re.match('[a-z]+://', match.group(1)) else match.group(1)
    
    text = re.sub('<a[^>]+>(.*)</a>', replace_link, text)
    return re.sub('<[^>]+>', EMPTY, text)
Questions['Text'] = Questions['Body'].apply(clean_text).str.lower()
Questions.Text = Questions.Text.apply(lambda x: x.replace('"','').replace("\n","").replace("\t","").replace("\r",""))
total = pd.merge(Questions,Tags,on='Id')
total['date'] = pd.to_datetime(total.CreationDate)
total.head()
tagCount =  Counter(list(total['Tag'])).most_common(10)
print(tagCount)
total['year'] = total['date'].apply(lambda x: x.year)
series = total.year.value_counts().sort_index()
# print(series)
series.plot(figsize=(5,5), grid=True)
pyplot.xlabel("YEARS")
pyplot.ylabel("NUMBER OF QUESTIONS ASKED")
pyplot.title("GRAPH SHOWING NUMBER OF QUESTIONS ASKED PER YEAR")
pyplot.show()
look_up = ["January",
          "Febuary",
          "March",
          "April",
          "May",
          "June",
          "July",
          "August",
          "September",
          "October",
          "November",
          "December"]
monthname = pd.Series( (v for v in look_up) )
monthname.index += 1
total['Month'] = total['date'].apply(lambda x: x.month)
series = total.Month.value_counts().sort_index()
month_wise_data = pd.concat([monthname, series], axis=1)
month_wise_data.rename(columns={0:'Months'}, inplace=True)
month_wise_data.rename(columns={'Month':'No_of_Questions'}, inplace=True)
# print(month_wise_data)
ax = month_wise_data.plot(figsize=(20,7),grid=True)
ax.set_xticks(month_wise_data.index)
ax.set_xticklabels(month_wise_data.Months)
pyplot.title("GRAPH SHOWING NUMBER OF QUESTIONS ASKED PER MONTH")
pyplot.xlabel("Months")
pyplot.ylabel("NUMBER OF QUESTIONS ASKED")
pyplot.show()

spacy_nlp = spacy.load('en_core_web_sm')
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
spacy_stopwords.add("/")
spacy_stopwords.add("=")
spacy_stopwords.add(">")
spacy_stopwords.add("<")
spacy_stopwords.add("#")

customize_stop_words = [
    '/', '=','>','<','#','`','|','}','{','¦',';'
    
]
for w in customize_stop_words:
    spacy_nlp.vocab[w].is_stop = True

print('Number of stop words: %d' % len(spacy_stopwords))
# print('First ten stop words: %s' % list(spacy_stopwords))
titleWords = []
i = 0
for word in Questions.Title:
    word = word.replace(".","").replace(",","").replace(":","").replace("\"","").replace("!","").replace("?","").replace("_","").replace("*","").replace("-","").replace("'","").replace(" ","").replace("]","").replace("[","").replace(")","").replace("(","")
    doc = spacy_nlp(word.lower())
    tokens = [token.text for token in doc if not token.is_stop]
    titleWords= titleWords + tokens
    i += 1
    if (i == 10000):
        break
# print(titleWords)
word_counter = Counter(titleWords)
# for word, count in word_counter.most_common(10):
#     print(word, ": ", count)

lst = word_counter.most_common(20)
# print(lst)
df = pd.DataFrame(lst, columns = ['Word', 'Count'])
ax = sns.barplot(x="Count", y="Word", data=df)
# df.plot.bar(x='Word',y='Count')

total.tail()
total.shape
total['wordcount'] = total.Text.apply(lambda x: len(x.split()))
total.shape
total = total[total.wordcount < 100]
total.shape
col = ['Id', 'OwnerUserId', 'CreationDate', 'date', 'Score', 'Title', 'Body', 'Month', 'year','wordcount']
total = total[(total.Tag == 'python') | (total.Tag == 'django') | (total.Tag == 'python-2.7') | (total.Tag =='pandas') | (total.Tag =='python-3.x') | (total.Tag == 'numpy') | (total.Tag == 'list') | (total.Tag == 'matplotlib') | (total.Tag == 'regex') | (total.Tag == 'dictionary')]
total.drop(col, axis=1, inplace=True)
total['Tag'].value_counts()
categories = total['Tag'].unique()
fig = pyplot.figure(figsize=(10,6))
total.groupby('Tag').Text.count().plot.bar(ylim=0)
pyplot.show()
lbl_enc = preprocessing.LabelEncoder()
y = lbl_enc.fit_transform(total.Tag.values)
xtrain, xtest , ytrain,ytest = train_test_split(total.Text, total.Tag, 
                                                  stratify=total.Tag, 
                                                  random_state=42, 
                                                  test_size=0.1, shuffle=True)
print(xtrain.shape)
print(xtest.shape)
tfv = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')

# Fitting TF-IDF to both training and test sets (semi-supervised learning)
tfv.fit(list(xtrain) + list(xtest))
xtrain_tfv =  tfv.transform(xtrain) 
xtest_tfv = tfv.transform(xtest)
clf_Multinomial = MultinomialNB(alpha=0.01)
clf_Multinomial.fit(xtrain_tfv, ytrain)
predictions_Multinomial = clf_Multinomial.predict(xtest_tfv)
score_Multinomial = metrics.accuracy_score(ytest, predictions_Multinomial)
print("accuracy:   %0.3f" % score_Multinomial)
clf = LogisticRegression(C=1.0)
clf.fit(xtrain_tfv, ytrain)
predictions = clf.predict(xtest_tfv)
score = metrics.accuracy_score(ytest, predictions)
print("accuracy:   %0.3f" % score)
print(classification_report(ytest, predictions, target_names=categories))
