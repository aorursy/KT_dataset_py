# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from numpy import argmax

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train=pd.read_csv('/kaggle/input/movie-review-sentiment-analysis-kernels-only/train.tsv',sep='\t')

test=pd.read_csv('/kaggle/input/movie-review-sentiment-analysis-kernels-only/test.tsv',sep='\t')
from IPython.display import display

display(train.head(20))
from wordcloud import WordCloud

import matplotlib.pyplot as plt

c=[None for i in range(6)]

for i in range(3):

    c[i]=WordCloud().generate(train['Phrase'].iloc[i])

    plt.imshow(c[i], interpolation='bilinear')

    plt.axis('off')

    plt.show()
train['len']=train['Phrase'].apply(len)

idx = train.groupby(['SentenceId','Sentiment']).apply(lambda x: x['len'].idxmax())

train=train.loc[idx]

display(train.head())
X=train['Phrase'].append(test['Phrase'])

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(ngram_range=(1,5))

X = vectorizer.fit_transform(X)

print(X.shape)

x_train=X[:train.shape[0]]

x_test=X[train.shape[0]:]

from keras.utils import to_categorical

y_train=to_categorical(train['Sentiment'].values)
list(vectorizer.vocabulary_.keys())[34:65]
from keras.layers import Dense

from keras.models import Sequential

model=Sequential()

n_cols=x_train.shape[1]

model.add(Dense(100,activation='relu',input_shape=(n_cols,)))

model.add(Dense(50,activation='relu'))

model.add(Dense(25,activation='relu'))

model.add(Dense(5,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train,y_train,validation_split=0.3,epochs=6)
p=model.predict(x_test)
predictions=[]

for i in p:

    predictions.append(argmax(i))

data_test=test.copy()

data_test['Sentiment']=pd.Series(predictions)

data_test=data_test[['PhraseId','Sentiment']]

data_test.to_csv('Submission.csv',index=False)