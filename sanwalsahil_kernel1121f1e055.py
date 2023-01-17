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

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import SVC

from wordcloud import WordCloud
df = pd.read_csv('../input/sms-spam-collection-dataset/spam.csv', encoding = "ISO-8859-1")
df.head()
df = df.drop(['Unnamed: 2','Unnamed: 4','Unnamed: 3'], axis=1)
df.columns = ['labels','data']
# create binary labels

df['b_labels'] = df['labels'].map({'ham':0, 'spam':1})
df.head()
Y = df['b_labels'].values
x_train,x_test,y_train,y_test = train_test_split(df['data'],Y,test_size=.33)
x_train.head()
tfidf = TfidfVectorizer()

x_train = tfidf.fit_transform(x_train)

x_test = tfidf.transform(x_test)
type(x_train)
model = MultinomialNB()

model.fit(x_train,y_train)

model.score(x_test,y_test)
y_pred = model.predict(x_test)

y_pred
text = ['WINNER!! As a valued network customer you have been selected to receivea å£900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.']

#tfidf = TfidfVectorizer()

trial = tfidf.transform(text)

model.predict_proba(trial)