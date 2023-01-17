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
ama = pd.read_csv("../input/amazon_alexa.tsv",sep='\t')

ama.head()

ama['rating'].value_counts()
ama.iloc[2]['verified_reviews']



from wordcloud import WordCloud

import matplotlib.pyplot as plt
amazon = ''.join(ama['verified_reviews'])

wc = WordCloud().generate(amazon)

plt.imshow(wc)
import nltk

docs = ama['verified_reviews'].str.lower().str.replace('[^a-z ]','')

docs.head()
stemmer = nltk.stem.PorterStemmer()

def clean_sentence(text):

    stopwords=nltk.corpus.stopwords.words('english')

    words=text.split(' ')

    words_clean=[stemmer.stem(word) for word in words if word not in stopwords]

    return ' '.join(words_clean)

docs_clean=docs.apply(clean_sentence)

docs_clean.head()
from sklearn.feature_extraction.text import CountVectorizer



vectorizer = CountVectorizer()

vectorizer.fit(docs_clean)

dtm = vectorizer.transform(docs_clean)

dtm
#sparse matrix



no_of_zeros=(748*2475)-6797

sparsity=(no_of_zeros)/(748*2475)*100

sparsity
df_dtm=pd.DataFrame(dtm.toarray(),columns=vectorizer.get_feature_names())

(df_dtm==0).sum().sum()
df_dtm.sum().sort_values(ascending=False).head()
from sklearn.model_selection import train_test_split

train_x,test_x=train_test_split(df_dtm,test_size=0.3,random_state=100)
df_dtm.shape
docs_clean.head(2)
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.ensemble import AdaBoostClassifier



vectorizer=TfidfVectorizer(stop_words= "english")



vectorizer.fit(train_x)

train_dtm=vectorizer.transform(train_x)

test_dtm=vectorizer.transform(test_x)

df_train=pd.DataFrame(train_dtm.toarray(),columns=vectorizer.get_feature_names())

df_test=pd.DataFrame(test_dtm.toarray(),columns=vectorizer.get_feature_names())
train_y=ama.iloc[df_train.index]['verified_reviews']

test_y=ama.iloc[df_test.index]['verified_reviews']
model = RandomForestClassifier(n_estimators=10,random_state=100)

model.fit(df_train,train_y)

pred = model.predict(df_test)

accuracy_score(test_y,pred)