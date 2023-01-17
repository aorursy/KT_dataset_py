# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import gensim

print(os.listdir('../input/embeddings/GoogleNews-vectors-negative300/'))



# Any results you write to the current directory are saved as output.
path = "../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin"

embeddings = gensim.models.KeyedVectors.load_word2vec_format(path,binary = True)
hotstar = pd.read_csv('https://bit.ly/2W21FY7')

hotstar['Sentiment_Manual'].value_counts()
hotstar.columns
target0 = hotstar[hotstar['Sentiment_Manual'] == 'Neutral']

target1 = hotstar[hotstar['Sentiment_Manual'] == 'Positive']

target2 = hotstar[hotstar['Sentiment_Manual'] == 'Negative']

from wordcloud import WordCloud

import matplotlib.pyplot as plt
doc0 = target0['Reviews']

wc0 = WordCloud(background_color='white').generate(''.join(doc0))

plt.imshow(wc0)
doc1 = target1['Reviews']

wc1 = WordCloud(background_color='white').generate(''.join(doc1))

plt.imshow(wc1)
doc2 = target2['Reviews']

wc2 = WordCloud(background_color='white').generate(''.join(doc2))

plt.imshow(wc2)
import nltk
#data cleaning

stopwords = nltk.corpus.stopwords.words('english')



stemmer = nltk.stem.PorterStemmer()



def clean_sentence(doc):

    words = doc.split(' ')

    words_clean = [stemmer.stem(word) for word in words if word not in stopwords]

    doc_clean = ' '.join(words_clean)

    return doc_clean



docs = hotstar['Reviews'].str.lower().str.replace('[^a-z ]','')

docs_clean = docs.apply(clean_sentence)



docs_clean.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(docs_clean, hotstar['Sentiment_Manual'],

                                                    test_size=0.2, 

                                                    random_state=100)
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(min_df=5).fit(X_train)

X_train = vectorizer.transform(X_train)

X_test = vectorizer.transform(X_test)
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import f1_score,accuracy_score
model_mnb = MultinomialNB().fit(X_train,y_train)

test_pred = model_mnb.predict(X_test)

print(accuracy_score(y_test,test_pred))
from sklearn.ensemble import  RandomForestClassifier,AdaBoostClassifier

model_ran = RandomForestClassifier(n_estimators = 300).fit(X_train,y_train)

test_pred = model_ran.predict(X_test)

print(accuracy_score(y_test,test_pred))



print("Ada")





model_ada = AdaBoostClassifier(n_estimators = 100).fit(X_train,y_train)

test_pred = model_ada.predict(X_test)

print(accuracy_score(y_test,test_pred))
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(docs_clean, hotstar['Sentiment_Manual'],

                                                    test_size=0.2, 

                                                    random_state=100)
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer().fit(X_train) #it supresses the weights 

X_train = tfidf.transform(X_train)

X_test = tfidf.transform(X_test)
model_mnb = MultinomialNB().fit(X_train,y_train)

test_pred = model_mnb.predict(X_test)

print(accuracy_score(y_test,test_pred))
docs_vectors = pd.DataFrame()

for doc in docs_clean:

    words = nltk.word_tokenize(doc)

    temp = pd.DataFrame()

    for word in words : 

        try: 

            word_vec = embeddings[word]

            temp = temp.append(pd.Series(word_vec), ignore_index= True)

        except:

            pass #goes to the next word 

    docs_vectors = docs_vectors.append(temp.mean(), ignore_index = True)

docs_vectors.shape

    
s = pd.DataFrame(pd.isnull(docs_vectors).sum(axis=1))
s.head()

s.columns = ['sum']

b = s[s['sum']==300] 

X = docs_vectors.drop(b.index)

y = hotstar['Sentiment_Manual'].drop(b.index)



X.shape, y.shape
from sklearn.model_selection import train_test_split

train_x , test_x , train_y, test_y = train_test_split(X,y,

                                                     test_size = 0.2 , random_state = 100 )

train_x.shape, test_x.shape, train_y.shape, test_y.shape
from sklearn.ensemble import  RandomForestClassifier,AdaBoostClassifier

model_ran_wv = RandomForestClassifier(n_estimators = 300).fit(train_x,train_y)

test_pred_wv = model_ran_wv.predict(test_x)

print(accuracy_score(test_y,test_pred_wv))



print("Ada")





model_ada_wv = AdaBoostClassifier(n_estimators = 100).fit(train_x,train_y)

test_pred_ada= model_ada_wv.predict(test_x)

print(accuracy_score(test_y,test_pred_ada))
from nltk.sentiment import SentimentIntensityAnalyzer



analyzer = SentimentIntensityAnalyzer()
def get_sentiment(sentence , analyzer = analyzer):

    compound = analyzer.polarity_scores(sentence)['compound']

    if compound > 0.1 :

        return 'Positive'

    if compound < 0.1 : 

        return 'Negative'

    else: 

        return 'Neutral'
hotstar['sentiment_vader'] = hotstar['Reviews'].apply(get_sentiment)
accuracy_score(hotstar['Sentiment_Manual'],hotstar['sentiment_vader'])