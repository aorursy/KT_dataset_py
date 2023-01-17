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
import numpy as np                        ## Matrix functions
import matplotlib.pyplot as plt           ## PLotting
import pandas as pd                       ## To Work WIth Dataframes 
import plotly.express as px               ## For Interactive Visualization
import plotly.graph_objects as go         ## For Detailed visual plots
from collections import Counter         
from plotly.subplots import make_subplots ## To Plot Subplots
from wordcloud import WordCloud           ## To Generate Wordcloud
from datetime import datetime             ## Work with timeseries data

import warnings
warnings.filterwarnings('ignore')
review = pd.read_csv('../input/trip-advisor-hotel-reviews/tripadvisor_hotel_reviews.csv')
review.head()
review.apply(lambda x: sum(x.isnull()))
# Data Cleaning
review['Review']= review['Review'].apply(lambda x : str(x).replace('\n', ' '))
review['Review']= review['Review'].apply(lambda x : x.lower())
# Ratings distribution
review.apply(lambda x: len(x.unique()))
review.groupby(by='Rating')['Review'].count()
# Extracting meaningful words from reviews
import nltk
from nltk.tokenize import word_tokenize

review['Words'] = review['Review'].apply(word_tokenize)

from nltk.corpus import stopwords 

StopWords = set(stopwords.words('english'))

def clean_words(x):
    words = []
    for i in x:
        if i.isalnum() and i not in StopWords:
            words.append(i)
    return words

review['Words'] = review['Words'].apply(clean_words)
review['Word Count'] = review['Words'].apply(lambda x : len(x))
del StopWords
# Review length by Rating
fig = px.histogram(review, x='Word Count', color='Rating',
            barmode = 'overlay', nbins=50, marginal = 'box')
fig.update_layout(title = "Word Count Distribution in Reviews by Ratings.",
                 xaxis_title = "Word Count",
                 yaxis_title = "No of Reviews")
fig.show()
review.drop('Word Count', axis = 1, inplace=True)
# Most common words by rating
most_common = dict()

for group, data in review.groupby(by='Rating'):
    words = []
    for i in data['Words'].tolist():
        words.extend(i)
    words = nltk.FreqDist(words)
    words = words.most_common(10)
    most_common['{}'.format(group)] = words
print("Most Common Words by ratings and their word-counts:")
pd.DataFrame(most_common)
# Parts Of Speech
review['POS'] = review['Words'].apply(nltk.pos_tag)
# Adjectives
def get_adjective(x):
    adj = set(['JJ', 'JJR', 'JJS'])
    word = []
    for i in x:
        if i[1] in adj:
            word.append(i[0])
    return word

review['ADJ'] = review['POS'].apply(get_adjective)

most_common = dict()
for group, data in review.groupby(by='Rating'):
    words = []
    for i in data['ADJ'].tolist():
        words.extend(i)
    words = nltk.FreqDist(words)
    words = words.most_common(10)
    most_common['{}'.format(group)] = words
print("Most Common Adjectives by ratings:")
pd.DataFrame(most_common)
# Nouns
def get_noun(x):
    noun = set(['NN', 'NNS', 'NNP', 'NNPS'])
    word = []
    for i in x:
        if i[1] in noun:
            word.append(i[0])
    return word

review['Noun'] = review['POS'].apply(get_noun)

review.drop('POS', axis = 1, inplace = True)

most_common = dict()
for group, data in review.groupby(by='Rating'):
    words = []
    for i in data['Noun'].tolist():
        words.extend(i)
    words = nltk.FreqDist(words)
    words = words.most_common(10)
    most_common['{}'.format(group)] = words
print("Most Common Nouns by ratings:")
pd.DataFrame(most_common)
# Common Bigrams
most_common = dict()
for group, data in review.groupby(by='Rating'):
    words = []
    for i in data['Words'].tolist():
        words.extend(i)
    bigram = list(nltk.bigrams(words))
    bigram = nltk.FreqDist(bigram)
    bigram = bigram.most_common(10)
    most_common['{}'.format(group)] = bigram

print("Most Common Bi-grams by Ratings:")
pd.DataFrame(most_common)
# Polarity And Subjectivity
from textblob import TextBlob

review['Subjectivity'] = review['Review'].apply(lambda x : TextBlob(x).sentiment.subjectivity)
review['Polarity'] = review['Review'].apply(lambda x : TextBlob(x).sentiment.polarity)
fig = px.histogram(review, x='Subjectivity', barmode='overlay', color='Rating')
fig.update_layout(title = "Subjectivity distribution in reviews of different ratings.",
                 xaxis_title = "Subjectivity",
                 yaxis_title = "Number of Reviews")
fig.show()
fig = px.histogram(review, x='Polarity', barmode='overlay', color='Rating')

fig.update_layout(title = "Polarity distribution in reviews of different ratings.",
                 xaxis_title = "Subjectivity",
                 yaxis_title = "Number of Reviews")
fig.show()
# I. Tf-idf
from sklearn.feature_extraction.text  import TfidfVectorizer
tf = TfidfVectorizer(stop_words = 'english', ngram_range = (1,2),
                    min_df = 1)
from sklearn.model_selection import train_test_split

X = review['Review']
y = review['Rating']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 1)

tf_x_train = tf.fit_transform(x_train)
tf_x_test = tf.transform(x_test)
# Models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
performance = {'Model' : [],
              'Accuracy Score' : [],
              'Precision Score' : [],
              'Recall Score' : [],
              'f1 Score' : []}
from sklearn.linear_model import LogisticRegression

lr= LogisticRegression()
lr.fit(tf_x_train, y_train)
pred = lr.predict(tf_x_test)

performance['Model'].append('LogisticRegression')
performance['Accuracy Score'].append(accuracy_score(y_test, pred))
performance['Precision Score'].append(precision_score(y_test, pred, average='macro'))
performance['Recall Score'].append(recall_score(y_test, pred, average='macro'))
performance['f1 Score'].append(f1_score(y_test, pred, average='macro'))
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier()
sgd.fit(tf_x_train, y_train)
pred = sgd.predict(tf_x_test)

performance['Model'].append('SGD')
performance['Accuracy Score'].append(accuracy_score(y_test, pred))
performance['Precision Score'].append(precision_score(y_test, pred, average='macro'))
performance['Recall Score'].append(recall_score(y_test, pred, average='macro'))
performance['f1 Score'].append(f1_score(y_test, pred, average='macro'))
from sklearn.naive_bayes import MultinomialNB

mnb = MultinomialNB()
mnb.fit(tf_x_train, y_train)
pred = mnb.predict(tf_x_test)

performance['Model'].append('Multinomial NB')
performance['Accuracy Score'].append(accuracy_score(y_test, pred))
performance['Precision Score'].append(precision_score(y_test, pred, average='macro'))
performance['Recall Score'].append(recall_score(y_test, pred, average='macro'))
performance['f1 Score'].append(f1_score(y_test, pred, average='macro'))
from sklearn.naive_bayes import BernoulliNB

bnb = BernoulliNB()
bnb.fit(tf_x_train, y_train)
pred = bnb.predict(tf_x_test)

performance['Model'].append('Bernoulli NB')
performance['Accuracy Score'].append(accuracy_score(y_test, pred))
performance['Precision Score'].append(precision_score(y_test, pred, average='macro'))
performance['Recall Score'].append(recall_score(y_test, pred, average='macro'))
performance['f1 Score'].append(f1_score(y_test, pred, average='macro'))
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
rfc.fit(tf_x_train, y_train)
pred = rfc.predict(tf_x_test)

performance['Model'].append('Random Forest')
performance['Accuracy Score'].append(accuracy_score(y_test, pred))
performance['Precision Score'].append(precision_score(y_test, pred, average='macro'))
performance['Recall Score'].append(recall_score(y_test, pred, average='macro'))
performance['f1 Score'].append(f1_score(y_test, pred, average='macro'))
from statistics import mode

class voted_classifier():
    def __init__(self):
        self.classifiers = [lr, sgd, mnb, bnb, rfc]
        
    def classify(self, features):
        names = ['lr', 'sgd', 'mnb', 'bnb', 'rfc']
        i = 0 
        votes = pd.DataFrame()
        for classifier in self.classifiers:
            pred = classifier.predict(features)
            votes[names[i]] = pred
            i+=1
        return votes.mode(axis = 1)[0]
vc = voted_classifier()
pred = vc.classify(tf_x_test)

performance['Model'].append('Voted Classifier')
performance['Accuracy Score'].append(accuracy_score(y_test, pred))
performance['Precision Score'].append(precision_score(y_test, pred, average='macro'))
performance['Recall Score'].append(recall_score(y_test, pred, average='macro'))
performance['f1 Score'].append(f1_score(y_test, pred, average='macro'))
pd.DataFrame(performance)
