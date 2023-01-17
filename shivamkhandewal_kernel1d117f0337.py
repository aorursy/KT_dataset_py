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
tr_1=pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv',usecols=['comment_text','toxic'])

tr_2=pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv',usecols=['comment_text','toxic'])
tr_1.head()
tr_2.head()
x_text=pd.concat((tr_1,tr_2),axis=0)
x_text
# https://stackoverflow.com/a/47091490/4084039

import re



def decontracted(phrase):

    # specific

    phrase = re.sub(r"won't", "will not", phrase)

    phrase = re.sub(r"can\'t", "can not", phrase)



    # general

    phrase = re.sub(r"n\'t", " not", phrase)

    phrase = re.sub(r"\'re", " are", phrase)

    phrase = re.sub(r"\'s", " is", phrase)

    phrase = re.sub(r"\'d", " would", phrase)

    phrase = re.sub(r"\'ll", " will", phrase)

    phrase = re.sub(r"\'t", " not", phrase)

    phrase = re.sub(r"\'ve", " have", phrase)

    phrase = re.sub(r"\'m", " am", phrase)

    return phrase
# https://gist.github.com/sebleier/554280

# we are removing the words from the stop words list: 'no', 'nor', 'not'

stopwords= ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\

            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \

            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\

            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \

            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \

            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \

            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\

            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\

            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\

            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \

            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \

            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\

            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\

            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \

            'won', "won't", 'wouldn', "wouldn't"]
# Combining all the above statemennts 

from tqdm import tqdm

preprocessed_essays = []

# tqdm is for printing the status bar

for sentance in tqdm(x_text['comment_text'].values):

    sent = decontracted(sentance)

    sent = sent.replace('\\r', ' ')

    sent = sent.replace('\\"', ' ')

    sent = sent.replace('\\n', ' ')

        # https://gist.github.com/sebleier/554280

    sent = ' '.join(e for e in sent.split() if e not in stopwords)

    preprocessed_essays.append(sent.lower().strip())
x_text['clean']=preprocessed_essays
def partition(x):

    if x < 0.5:

        return 0

    return 1



#changing reviews with score less than 3 to be positive and vice-versa

actualScore = x_text['toxic']

positiveNegative = actualScore.map(partition) 

x_text['toxic'] = positiveNegative

print("Number of data points in our data", x_text.shape)

x_text.head(3)
a=set(x_text['toxic'])
a
x_v=pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')
x_v.head()
from tqdm import tqdm

preprocessed_essays = []

# tqdm is for printing the status bar

for sentance in tqdm(x_v['comment_text'].values):

    sent = decontracted(sentance)

    sent = sent.replace('\\r', ' ')

    sent = sent.replace('\\"', ' ')

    sent = sent.replace('\\n', ' ')

        # https://gist.github.com/sebleier/554280

    sent = ' '.join(e for e in sent.split() if e not in stopwords)

    preprocessed_essays.append(sent.lower().strip())
x_v['clean']=preprocessed_essays
x_test=pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')
x_test.head()
from tqdm import tqdm

preprocessed_essays = []

# tqdm is for printing the status bar

for sentance in tqdm(x_test['content'].values):

    sent = decontracted(sentance)

    sent = sent.replace('\\r', ' ')

    sent = sent.replace('\\"', ' ')

    sent = sent.replace('\\n', ' ')

        # https://gist.github.com/sebleier/554280

    sent = ' '.join(e for e in sent.split() if e not in stopwords)

    preprocessed_essays.append(sent.lower().strip())
x_test['clean']=preprocessed_essays
x_test.head()
# we use count vectorizer to convert the values into one hot encoded features

from sklearn.feature_extraction.text import CountVectorizer
# We are considering only the words which appeared in at least 10 documents(rows or projects).

vectorizer = CountVectorizer(min_df=10)

text_bow_tr = vectorizer.fit_transform(x_text['clean'])

text_bow_cv=vectorizer.transform(x_v['clean'])

text_bow_test=vectorizer.transform(x_test['clean'])

print("Shape of matrix after one hot encodig ",text_bow_test.shape)
import numpy as np

import nltk

import string

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import confusion_matrix

from sklearn import metrics

from sklearn.metrics import roc_curve, auc

from nltk.stem.porter import PorterStemmer



import re

# Tutorial about Python regular expressions: https://pymotw.com/2/re/

import string

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from nltk.stem.wordnet import WordNetLemmatizer



from gensim.models import Word2Vec

from gensim.models import KeyedVectors

import pickle



from tqdm import tqdm

import os





from collections import Counter
print("Final Data matrix")

print("train matrix=>",X_tr.shape, y_train.shape)



print("test matrix=>",X_te.shape, y_test.shape)

print("="*100)   
y_train=x_text['toxic']

y_cv=x_v['toxic']

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.datasets import *

from sklearn.linear_model import LogisticRegression
logistic=LogisticRegression(class_weight='balanced')

logistic.fit(text_bow_tr, y_train)
pred = logistic.predict(text_bow_cv)

acc = accuracy_score(y_cv, pred, normalize=True)*float(100)