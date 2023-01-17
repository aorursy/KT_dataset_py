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
path='/kaggle/input/quora-insincere-questions-classification/train.csv'

train=pd.read_csv(path,nrows=1000)

train.head()
## Converting every character to lower case



docs=train['question_text'].str.lower()

docs.head()
## Remove non-alphabets



docs= docs.str.replace('[^a-z ]', '')

docs.head()
## Remove commly used words



import nltk

stopwords=nltk.corpus.stopwords.words('english')

stopwords



stemmer=nltk.stem.PorterStemmer()

def clean_sentence(doc):

    words=doc.split(' ')

    words_clean=[stemmer.stem(word) for word in words if word not in stopwords]

    return ' '.join(words_clean)



docs=docs.apply(clean_sentence)
#Usage of stemmer in above code



stemmer.stem('playing')
## Term Frequency- Inverse Document Frequency (TF-IDF) for my knowledge have to study to standadize text target column
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

dtm_vectorizer=CountVectorizer()



train_x,validate_x,train_y,validate_y=train_test_split(docs,train['target'],test_size=0.2,random_state=0)



dtm_vectorizer.fit(train_x)

dtm_train=dtm_vectorizer.transform(train_x)

dtm_validate=dtm_vectorizer.transform(validate_x)
df_dtm_train=pd.DataFrame(dtm_train.toarray(),columns=dtm_vectorizer.get_feature_names(),index=train_x.index)

df_dtm_train.head
df_dtm_train.sum().sort_values(ascending=False).head(20).plot.bar()
from sklearn.naive_bayes import MultinomialNB

model=MultinomialNB().fit(dtm_train,train_y)

validate_y_pred=model.predict(dtm_validate)



from sklearn.metrics import accuracy_score,f1_score

print('AccuracyScore:',accuracy_score(validate_y,validate_y_pred))

print('F1 Score:',f1_score(validate_y,validate_y_pred))
## Sentiment Analysis

from nltk.sentiment import SentimentIntensityAnalyzer

sentiment_analyzer=SentimentIntensityAnalyzer()

sentiment_analyzer.polarity_scores('i love india')