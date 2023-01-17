import numpy as np

import pandas as pd





import re

import string

import nltk

from nltk.corpus import stopwords



stop = set(stopwords.words('english'))

from wordcloud import WordCloud , STOPWORDS

import matplotlib as plty

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import sklearn



import seaborn as sns



from subprocess import check_output



import plotly.offline as py



py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



from sklearn import preprocessing



Encode = preprocessing.LabelEncoder()

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer



tfid = TfidfVectorizer()

vect = CountVectorizer()



from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn.manifold import TSNE

NB = MultinomialNB()



import nltk



from sklearn import metrics

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier



print(check_output(['ls','../input']).decode('utf8'))
chatbot = pd.read_csv('../input/deepnlp/Sheet_1.csv',usecols=['response_id','class','response_text'],encoding='latin-1')

resume = pd.read_csv('../input/deepnlp/Sheet_2.csv',encoding='latin-1')
chatbot.head()
chatbot['class'].value_counts()
def cloud(text):

    wordcloud = WordCloud(background_color='black',stopwords=stop).generate(' '.join([i for i in text.str.upper()]))

    plt.imshow(wordcloud)

    plt.axis('off')

    plt.title('Chat Bot response')

cloud(chatbot['response_text'])
chatbot['Label'] = Encode.fit_transform(chatbot['class'])
# Naive Bayes



x = chatbot.response_text

y = chatbot.Label

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1)

x_train_dtm = vect.fit_transform(x_train)

x_test_dtm = vect.transform(x_test)

NB.fit(x_train_dtm,y_train)

y_predict = NB.predict(x_test_dtm)

metrics.accuracy_score(y_test,y_predict)
rf = RandomForestClassifier(max_depth=10 , max_features=10)

rf.fit(x_train_dtm , y_train)

rf_predict = rf.predict(x_test_dtm)

metrics.accuracy_score(y_test,rf_predict)
Chatbot_Text = chatbot['response_text']

len(Chatbot_Text)
tf_idf = CountVectorizer(max_features=256).fit_transform(Chatbot_Text.values)
tsne = TSNE(

    n_components=3,

    init='random',

    random_state=101,

    method = 'barnes_hut',

    n_iter = 250,

    verbose=2,

    angle=0.5

).fit_transform(tf_idf.toarray())