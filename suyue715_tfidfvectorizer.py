# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize 

from nltk.stem import SnowballStemmer

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import spacy

from textblob import TextBlob



# Any results you write to the current directory are saved as output.

from keras.preprocessing import sequence

from keras.utils import np_utils

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Embedding

from keras.layers import LSTM



from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.decomposition import LatentDirichletAllocation

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD
#import the datasets

train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')

sample=pd.read_csv('../input/sample_submission.csv')
#Assign the authors with numbers to a new column 'author_num'

#0 for 'EAP'

#1 for 'HPL'

#2 for 'MWS'

train['author_num']=train['author'].apply({'EAP':0,  'HPL':1,'MWS':2}.get)

train.head()
#Assign the features and target

X_text_train=train['text'].values

X_text_test=test['text'].values

y=train['author_num'].values

num_labels = len(np.unique(train['author_num']))
#Define the stopwords to remove and the stemming tool

stop_words = set(stopwords.words('english'))

stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])

stemmer = SnowballStemmer('english')
#Preprocess the text in training and testing

processed_train = []

for doc in X_text_train:

    tokens = word_tokenize(doc)

    filtered = [word for word in tokens if word not in stop_words]

    stemmed = [stemmer.stem(word) for word in filtered]

    processed_train.append(stemmed)

    

processed_test = []

for doc in X_text_test:

    tokens = word_tokenize(doc)

    filtered = [word for word in tokens if word not in stop_words]

    stemmed = [stemmer.stem(word) for word in filtered]

    processed_test.append(stemmed)
X_text_train[1]
processed_train[1]
train['processed_train']=processed_train
train.head()
train.columns
row_lst = []

for lst in train.loc[:,'processed_train']:

    text = ''

    for word in lst:

        text = text + ' ' + word

    row_lst.append(text)



train['final_processed_text'] = row_lst
test['processed_test']=processed_test

test.head()
row_lst = []

for lst in test.loc[:,'processed_test']:

    text = ''

    for word in lst:

        text = text + ' ' + word

    row_lst.append(text)



test['final_processed_test'] = row_lst



test.head()
train.head()
nlp = spacy.load('en')

content=[]

for i in train['processed_train']:

    content.append(i)



# for named_entity in content.ents:

#     print(named_entity.text, named_entity.label_)
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD

cv = CountVectorizer(stop_words='english')

cv.fit(train['text'])

X = cv.transform(train['text'])

feature_names = cv.get_feature_names()



lda = LatentDirichletAllocation(n_components=10)

lda.fit(X)



results = pd.DataFrame(lda.components_,

                      columns=feature_names)



for topic in range(10):

    print('Topic', topic)

    word_list = results.T[topic].sort_values(ascending=False).index

    print(' '.join(word_list[0:25]), '\n')
X_train, X_test, y_train, y_test = train_test_split(train['final_processed_text'],

                                                   train['author_num'],

                                                   test_size=0.33,

                                                   random_state=8675309)


cv = CountVectorizer(stop_words='english')

cv.fit(X_train)



X_train_cv = cv.transform(X_train)

X_test_cv = cv.transform(X_test)



rf = RandomForestClassifier()

rf.fit(X_train_cv, y_train)

print(rf.score(X_test_cv, y_test))

predictions = rf.predict(X_test_cv)

print(confusion_matrix(y_test, predictions))

print(classification_report(y_test, predictions))


tfidf = TfidfVectorizer(stop_words='english')

tfidf.fit(X_train)



X_train_tfidf = tfidf.transform(X_train)

X_test_tfidf = tfidf.transform(X_test)

test_tfidf = tfidf.transform(test['final_processed_test'])



rf = RandomForestClassifier()

rf.fit(X_train_tfidf, y_train)

print(rf.score(X_test_tfidf, y_test))

predictions = rf.predict(X_test_tfidf)

print(confusion_matrix(y_test, predictions))

print(classification_report(y_test, predictions))
pred=rf.predict_proba(test_tfidf)

prob=pd.DataFrame(pred,columns=['EAP','HPL','MWS'])

submit1=pd.concat([test, prob], axis=1)

del submit1['text']
del submit1['processed_test']
del submit1['final_processed_test']
submit1
submit1.to_csv('./TfidfVectorizer.csv', index=False, header=True)