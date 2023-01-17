# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np 

import random

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import string

%matplotlib inline

from plotly import graph_objs as go

import plotly.express as px

import plotly.figure_factory as ff

from collections import Counter

import nltk

from nltk.corpus import stopwords



from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



from tqdm import tqdm

import os

import nltk

import spacy

import random

from spacy.util import compounding

from spacy.util import minibatch
train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')

test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')

submission = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')
train.head()
#Distribution of Sentiment column

train['sentiment'].value_counts(normalize=True)
plt.figure(figsize=(12,8))

sns.countplot(x='sentiment',data=train)
import re

def clean_text(text):

    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation

    and remove words containing numbers.'''

    text = str(text).lower()

    text = re.sub('\[.*?\]', '', text)

    text = re.sub('https?://\S+|www\.\S+', '', text)

    text = re.sub('<.*?>+', '', text)

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub('\n', '', text)

    text = re.sub('\w*\d\w*', '', text)

    return text

train['text'] = train['text'].apply(lambda x:clean_text(x))

train['selected_text'] = train['selected_text'].apply(lambda x:clean_text(x))
train.head()
train['text_len'] = train['text'].astype(str).apply(len)

train['text_word_count'] = train['text'].apply(lambda x: len(str(x).split()))
train['temp_list'] = train['selected_text'].apply(lambda x:str(x).split())

top = Counter([item for sublist in train['temp_list'] for item in sublist])

temp = pd.DataFrame(top.most_common(20))

temp.columns = ['Common_words','count']

temp.style.background_gradient(cmap='Blues')
from nltk.stem.snowball import SnowballStemmer



# create an object of stemming function

stemmer = SnowballStemmer("english")



def stemming(text):    

    '''a function which stems each word in the given text'''

    text = [stemmer.stem(word) for word in text.split()]

    return " ".join(text) 
train['text'] = train['text'].apply(stemming)

train['selected_text'] = train['selected_text'].apply(stemming)

train.head(10)
Positive_sent = train[train['sentiment']=='positive']

Negative_sent = train[train['sentiment']=='negative']

Neutral_sent = train[train['sentiment']=='neutral']

#MosT common positive words

top = Counter([item for sublist in Positive_sent['temp_list'] for item in sublist])

temp_positive = pd.DataFrame(top.most_common(20))

temp_positive.columns = ['Common_words','count']

temp_positive.style.background_gradient(cmap='Greens')
fig = px.bar(temp_positive, x="count", y="Common_words", title='Most Commmon Positive Words', orientation='h', 

             width=700, height=700,color='Common_words')

fig.show()
count_vectorizer = CountVectorizer()

train_vectors = count_vectorizer.fit_transform(train['text'])

test_vectors = count_vectorizer.transform(test["text"])



## Keeping only non-zero elements to preserve space 

print(train_vectors[0].todense())
tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2))

train_tfidf = tfidf.fit_transform(train['text'])

test_tfidf = tfidf.transform(test["text"])
from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import TfidfVectorizer



tvec = TfidfVectorizer(stop_words=None, max_features=100000, ngram_range=(1, 3))

lr = LogisticRegression()



from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import precision_score, recall_score, f1_score



def lr_cv(splits, X, Y, pipeline, average_method):

    

    kfold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=777)

    accuracy = []

    precision = []

    recall = []

    f1 = []

    for train, test in kfold.split(X, Y):

        lr_fit = pipeline.fit(X[train], Y[train])

        prediction = lr_fit.predict(X[test])

        scores = lr_fit.score(X[test],Y[test])

        

        accuracy.append(scores * 100)

        precision.append(precision_score(Y[test], prediction, average=average_method)*100)

        print('              negative    neutral     positive')

        print('precision:',precision_score(Y[test], prediction, average=None))

        recall.append(recall_score(Y[test], prediction, average=average_method)*100)

        print('recall:   ',recall_score(Y[test], prediction, average=None))

        f1.append(f1_score(Y[test], prediction, average=average_method)*100)

        print('f1 score: ',f1_score(Y[test], prediction, average=None))

        print('-'*50)



    print("accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(accuracy), np.std(accuracy)))

    print("precision: %.2f%% (+/- %.2f%%)" % (np.mean(precision), np.std(precision)))

    print("recall: %.2f%% (+/- %.2f%%)" % (np.mean(recall), np.std(recall)))

    print("f1 score: %.2f%% (+/- %.2f%%)" % (np.mean(f1), np.std(f1)))
from sklearn.pipeline import Pipeline

original_pipeline = Pipeline([

    ('vectorizer', tvec),

    ('classifier', lr)

])

lr_cv(5, train.text, train.sentiment, original_pipeline, 'macro')
def predict_entities(text, lr):

    doc = lr(text)

    ent_array = []

    for ent in doc.ents:

        start = text.find(ent.text)

        end = start + len(ent.text)

        new_int = [start, end, ent.label_]

        if new_int not in ent_array:

            ent_array.append([start, end, ent.label_])

    selected_text = text[ent_array[0][0]: ent_array[0][1]] if len(ent_array) > 0 else text

    return selected_text

sample = pd.read_csv("../input/tweet-sentiment-extraction/sample_submission.csv")

submission['selected_text'] = test['text']

submission.to_csv("submission.csv", index=False)

display(submission.head(10))