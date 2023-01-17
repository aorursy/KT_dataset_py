# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

data = pd.read_csv('../input/train.csv')



# Any results you write to the current directory are saved as output.
data.head()
data.shape
data.info()
target0 = data[data['target'] == 0]
target1 = data[data['target']==1]
from wordcloud import WordCloud
import matplotlib.pyplot as plt
#create wordcloud for target 0
doc0 = target0['question_text']

wc0 = WordCloud(background_color='white').generate(''.join(doc0))
plt.imshow(wc0)
#wordcloud for target 1

doc1 = target1['question_text']

wc1 = WordCloud(background_color='white').generate(''.join(doc1))
plt.imshow(wc1)
import nltk
#data cleaning

stopwords = nltk.corpus.stopwords.words('english')
stemmer = nltk.stem.PorterStemmer()



def clean_sentence(doc):

    words = doc.split(' ')

    words_clean = [stemmer.stem(word) for word in words if word not in stopwords]

    doc_clean = ' '.join(words_clean)

    return doc_clean



docs = data['question_text'].str.lower().str.replace('[^a-z ]','')

docs_clean = docs.apply(clean_sentence)



docs_clean.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(docs_clean, data['target'], test_size=0.2, random_state=100)
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(min_df=50).fit(X_train)

X_train = vectorizer.transform(X_train)

X_test = vectorizer.transform(X_test)
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import f1_score
model_mnb = MultinomialNB().fit(X_train,y_train)

test_pred = model_mnb.predict(X_test)

print(f1_score(y_test,test_pred))
dtest = pd.read_csv('../input/test.csv')
dtest.head()
def clean_sentence(doc):

    words = doc.split(' ')

    words_clean = [stemmer.stem(word) for word in words if word not in stopwords]

    doc_clean_test = ' '.join(words_clean)

    return doc_clean_test



docs = dtest['question_text'].str.lower().str.replace('[^a-z ]','')

docs_clean_test = docs.apply(clean_sentence)



docs_clean_test.head()
test_data = vectorizer.transform(docs_clean_test)
test_predict = model_mnb.predict(test_data)
data_to_submit = pd.DataFrame({

    'qid':dtest['qid'],

    'prediction':test_predict

})
data_to_submit.to_csv('csv_to_submit.csv', index = False)
sub = pd.read_csv('csv_to_submit.csv')

sub.to_csv('submission.csv',index = False)