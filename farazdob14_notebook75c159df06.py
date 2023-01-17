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
import nltk

import pandas as pd
messages = pd.read_csv('../input/movie-review/train.tsv', sep='\t',usecols=[1,2,3])

messages.drop(['SentenceId'], axis=1,inplace=True)

messages.columns=["message","label"]



messages.head()



messages1 = pd.read_csv('../input/movie-review/test.tsv', sep='\t',usecols=[0,2])

messages1['Sentiment']=2

messages1.columns=["PhraseId","message",'Sentiment']



messages1.head()
msg_train=messages['message']

msg_test=messages1['message']

label_train=messages['label']

label_train.describe()
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline



pipeline = Pipeline([

    ('bow', CountVectorizer()),  # strings to token integer counts

    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores

    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier

])



pipeline.fit(msg_train,label_train)

predictions = pipeline.predict(msg_test)
messages1['Sentiment']=predictions
answer=messages1[['PhraseId','Sentiment']]
answer.to_csv('mycsvfile.csv',index=False)