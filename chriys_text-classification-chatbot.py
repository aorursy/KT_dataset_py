# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import pandas as pd

import numpy as np

import re

import string

import nltk

from nltk.corpus import stopwords

stop = set(stopwords.words('english'))

from wordcloud import WordCloud, STOPWORDS

import matplotlib as plty

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import sklearn





import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from subprocess import check_output



%matplotlib inline

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



import plotly.graph_objs as go

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

from nltk.corpus import stopwords

stopwords = stopwords.words("english")

from sklearn import metrics

from sklearn.svm import LinearSVC

from sklearn.ensemble import RandomForestClassifier



print(check_output(["ls", "../input"]).decode("utf8"))
chatbot = pd.read_csv("../input/Sheet_1.csv",usecols=['response_id','class','response_text'],encoding='latin-1')

resume = pd.read_csv("../input/Sheet_2.csv",encoding='latin-1')
chatbot.head(20)
chatbot['class'].value_counts()
def cloud(text):

    wordcloud = WordCloud(background_color="blue",stopwords=stop).generate(" ".join([i for i in text.str.upper()]))

    plt.imshow(wordcloud)

    plt.axis("off")

    plt.title("Chat Bot Response")

cloud(chatbot['response_text'])
chatbot['Label'] = Encode.fit_transform(chatbot['class'])
chatbot['Label'].value_counts()

#not_flagged    55

#flagged        25
x = chatbot.response_text

y = chatbot.Label

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1)



x_train_dtm = vect.fit_transform(x_train)

x_test_dtm = vect.transform(x_test)



NB.fit(x_train_dtm,y_train)

y_predict = NB.predict(x_test_dtm)

metrics.accuracy_score(y_test,y_predict)



rf = RandomForestClassifier(max_features=10)

rf.fit(x_train_dtm,y_train)



rf_predict = rf.predict(x_test_dtm)

metrics.accuracy_score(y_test,rf_predict)
rf = LinearSVC()

rf.fit(x_train_dtm,y_train)

rf_predict = rf.predict(x_test_dtm)

metrics.accuracy_score(y_test,rf_predict)
Chatbot_Text = chatbot["response_text"]

len(Chatbot_Text)