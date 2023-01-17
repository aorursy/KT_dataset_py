import pandas as pd



data = pd.read_csv('/kaggle/input/sms-spam-collection-dataset/spam.csv', encoding='latin-1')

data = data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
%pip install urlextract

import urlextract

import re

import numpy as np



url_extractor = urlextract.URLExtract()



data = data.replace(['ham','spam'],[0, 1])

data['Count']=0

for i in np.arange(0,len(data.v2)):

    data.loc[i,'Count'] = len(data.loc[i,'v2'])

data['URL'] = 0

for i in np.arange(0,len(data.v2)):

    data.loc[i,'URL'] = 1 if len(url_extractor.find_urls(data.loc[i,'v2'])) > 0 else 0



data['NUMBER'] = 0

for i in np.arange(0,len(data.v2)):

    data.loc[i,'NUMBER'] = 1 if len(re.findall(r'\d+(?:\.\d*(?:[eE]\d+))?', data.loc[i,'v2'])) > 0 else 0







data.head()
import nltk

nltk.download('stopwords')

from nltk import stem

from nltk.corpus import stopwords

from nltk.stem.snowball import SnowballStemmer



stemmer = stem.SnowballStemmer('english')

stopwords = set(stopwords.words('english'))



def alternative_review_messages(msg):

    stemmer = SnowballStemmer("english")

    msg = re.sub(r'\W+', ' ', msg, flags=re.M)

    # converting messages to lowercase

    msg = msg.lower()

    urls = list(set(url_extractor.find_urls(msg)))

    urls.sort(key=lambda url: len(url), reverse=True)

    for url in urls:

        msg = msg.replace(url, "URL")

    msg = re.sub(r'\d+(?:\.\d*(?:[eE]\d+))?', 'NUMBER', msg)

    # removing stopwords

    msg = [word for word in msg.split() if word not in stopwords]

    # using a stemmer

    msg = " ".join([stemmer.stem(word) for word in msg])

    # print(msg)

    # print(stemmer.stem("girls"))

    return msg



print(alternative_review_messages('33Ok lar... Joking wif u oni...'))
all_ = []

for i in np.arange(0,len(data.v2)):

  all_.append(alternative_review_messages(data.loc[i,'v2']))

  

print(len(all_))
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

x = cv.fit_transform(all_).toarray()
y = data['v1']

from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(x, y,test_size= 0.20, random_state = 0)
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import classification_report



dt = MultinomialNB()

dt.fit(xtrain, ytrain)

y_pred_dt = dt.predict(xtest)

print (classification_report(ytest, y_pred_dt))

print('score: ', dt.score(xtest, ytest))

from sklearn.metrics import recall_score

print('recall: ', recall_score(ytest, y_pred_dt, average='macro'))

from sklearn.metrics import precision_score

print('prec: ', precision_score(ytest, y_pred_dt, average='macro'))

from sklearn.metrics import f1_score

print('f1_score: ', f1_score(ytest, y_pred_dt, average='macro'))