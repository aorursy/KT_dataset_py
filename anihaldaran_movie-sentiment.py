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
import pandas as pd

data=pd.read_csv("../input/train.tsv", sep = '\t') #path to train.tsv file
data1=pd.read_csv("../input/test.tsv", sep = '\t') #path to test.tsv file

print(data.columns)  # print the columns of the data in the training data
print(data1.columns)  # print the columns of the data in the test data
#Shows the most common phrases in the train.tsv file using a word cloud.
import matplotlib.pyplot as plt
train_P = pd.Series(data['Phrase'].tolist()).astype(str)
from wordcloud import WordCloud
cloud = WordCloud(width=1440, height=1080).generate(" ".join(train_P.astype(str)))
plt.figure(figsize=(20, 15))
plt.imshow(cloud)
plt.axis('off')
x_train = data['Phrase']
x_test = data1['Phrase']
y=data['Sentiment']
from nltk.tokenize import TweetTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
text_clf = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1,10), max_features=10000, lowercase=True, use_idf=True, smooth_idf=True, sublinear_tf=False, tokenizer=TweetTokenizer().tokenize, stop_words='english')),
                         ('clf', LogisticRegression(random_state=17, C=1.8))])
from sklearn.model_selection import RandomizedSearchCV
parameters = {
               'clf__C': np.logspace(.1,1,10),
 }
gs = RandomizedSearchCV(text_clf, parameters, n_jobs=-1, verbose=3)
text_clf.fit(x_train, y)
predicted = text_clf.predict(x_test)
data1['Sentiment'] = predicted
submission = data1[["PhraseId","Sentiment"]]
submission.to_csv("submission.csv", index = False)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go #for plotly graphs
dtt = data.assign(n=0).groupby(['Sentiment', 'SentenceId'])['n'].count().reset_index()
dtt = dtt[dtt["SentenceId"] < 2000]
ver = dtt.pivot(index='SentenceId', columns='Sentiment', values='n').fillna(0).values.tolist()
iplot([go.Surface(z=ver)])
#plotly Surface (the most impressive feature)
#shows the distribution od SentenceIds with sentiments
submission2 = data1[["PhraseId","Sentiment"]]
submission2.to_csv("submission2.csv", index = False)