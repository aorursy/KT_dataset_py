# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

file_path1="../input/BuzzFeed_fake_news_content.csv"

file_path2="../input/BuzzFeed_real_news_content.csv"

df1=pd.read_csv(file_path1)

df2=pd.read_csv(file_path2)

labels=np.zeros(182)

labels[0:91]=0

labels[91:182]=1

print(labels.shape)

dataset=df1.append(df2)

News_content=dataset['text']

# Any results you write to the current directory are saved as output.
import re

import nltk

from nltk.corpus import stopwords

from nltk import word_tokenize

from nltk.stem import PorterStemmer

corpus=[]

ps=PorterStemmer()

stopwords=set(stopwords.words("english"))

for i in range(182):

    review=dataset.iloc[i][2]

    review = re.sub('[^a-zA-Z]', ' ', dataset.iloc[i][2] )

    review = re.sub('[?&#@$%!_.|,-:;"]', ' ', dataset.iloc[i][2] )    

    review=review.lower()

    words=word_tokenize(review)

    d=list()

    for w in words:

        if w not in stopwords:

            d.append(ps.stem(w))

    review=" ".join(d)

    corpus.append(review)
print(corpus[1:2])
from sklearn.feature_extraction.text import CountVectorizer

cv=CountVectorizer(max_features=1500)

X=cv.fit_transform(corpus).toarray()
print(X.shape)
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,labels,test_size=0.30,random_state=0,shuffle=True)
from xgboost import XGBClassifier

classifier = XGBClassifier()

classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, Y_pred)

print(cm)
import seaborn as sns

import matplotlib.pyplot as plt     



ax= plt.subplot()

sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells



# labels, title and ticks

ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 

ax.set_title('Confusion Matrix'); 

ax.xaxis.set_ticklabels(['real', 'fake']); ax.yaxis.set_ticklabels(['real', 'fake']);