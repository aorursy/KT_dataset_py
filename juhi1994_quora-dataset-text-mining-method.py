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
#https://www.kaggle.com/c/quora-insincere-questions-classification

path='/kaggle/input/quora-insincere-questions-classification/train.csv'

train=pd.read_csv(path,nrows=1000)

train.head()
#Converting every character to lower case

docs=train['question_text'].str.lower()

print(docs.head())

print('\n')



#Remove non-alphabets

docs.str.replace('[^a-z ]','')

print(docs.head())

print('\n')



#Remove commonly used words

import nltk

stopwords=nltk.corpus.stopwords.words('english')

stemmer=nltk.stem.PorterStemmer()

print(stopwords)

print('\n')



def clean_sentence(doc):

    words=doc.split(' ')

    words_clean=[stemmer.stem(word) for word in words if word not in stopwords]

    return ' '.join(words_clean)

    print(words_clean)

    

docs=docs.apply(clean_sentence)

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split



dtm_vectorizer = CountVectorizer()





train_x,validate_x, train_y,validate_y = train_test_split(docs, train['target'], test_size = 0.2, random_state = 1)

dtm_vectorizer.fit(train_x)

dtm_train = dtm_vectorizer.transform(train_x)

dtm_validate = dtm_vectorizer.transform(validate_x)
dtm_train
df_dtm_train = pd.DataFrame(dtm_train.toarray(),columns=dtm_vectorizer.get_feature_names(),index=train_x.index)

df_dtm_train
df_dtm_train.sum().sort_values(ascending=False).head(20).plot.bar()
from sklearn.naive_bayes import MultinomialNB

model=MultinomialNB().fit(dtm_train,train_y)

train_y_pred=model.predict(dtm_validate)



from sklearn.metrics import accuracy_score,f1_score

print(accuracy_score(validate_y,train_y_pred))

print(f1_score(validate_y,train_y_pred))

#Sentiment Analysis

from nltk.sentiment import SentimentIntensityAnalyzer

sentiment_analyzer=SentimentIntensityAnalyzer()

sentiment_analyzer.polarity_scores('i like india')