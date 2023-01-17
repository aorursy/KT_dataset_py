import numpy as np

import pandas as pd 

import os

import gensim

print(os.listdir("../input"))

print(os.listdir("../input/embeddings/GoogleNews-vectors-negative300/"))



# Any results you write to the current directory are saved as output.
path = "../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin"

embeddings = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)

## Collection of all these word vectorings is embeddings
hotstar = pd.read_csv('https://bit.ly/2W21FY7')

hotstar.head()
import nltk
hotstar['Sentiment_Manual'].value_counts() # Checking the count
hotstar.isnull().sum() # Check for null values
!pip install wordcloud
from wordcloud import WordCloud

import matplotlib.pyplot as plt

%matplotlib inline

Neutral = hotstar[hotstar['Sentiment_Manual'] == 'Neutral']

Positive = hotstar[hotstar['Sentiment_Manual'] =='Positive']

Negative = hotstar[hotstar['Sentiment_Manual' ]=='Negative']
docs0=Neutral['Lower_Case_Reviews']

print(len(docs0))

docs1=Positive['Lower_Case_Reviews']

print(len(docs1))

docs2=Negative['Lower_Case_Reviews']

print(len(docs2))

stopwords=nltk.corpus.stopwords.words('english')

wc0 = WordCloud(background_color='white',stopwords=stopwords).generate(' '.join(docs0))

plt.imshow(wc0)
stopwords=nltk.corpus.stopwords.words('english')

wc1 = WordCloud(background_color='white',stopwords=stopwords).generate(' '.join(docs1))

plt.imshow(wc1)
stopwords=nltk.corpus.stopwords.words('english')

wc2 = WordCloud(background_color='white',stopwords=stopwords).generate(' '.join(docs2))

plt.imshow(wc2)
docs=hotstar['Lower_Case_Reviews']

len(hotstar)
docs=docs.str.replace('[^a-z A-Z #@]', '')

docs.head()
from sklearn.model_selection import train_test_split

train_x,test_x,train_y,test_y=train_test_split(docs,

                                               hotstar['Sentiment_Manual'],

                                               test_size=0.2,

                                               random_state=100)
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(min_df=5).fit(train_x)

train_x = vectorizer.transform(train_x)

test_x = vectorizer.transform(test_x)
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score

model_mnb = MultinomialNB().fit(train_x , train_y)

test_pred_mnb = model_mnb.predict(test_x)

print(accuracy_score(test_y , test_pred_mnb))
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

model = RandomForestClassifier(n_estimators=500).fit(train_x, train_y)

test_pred = model.predict(test_x)

accuracy_score(test_y, test_pred)
model = AdaBoostClassifier(n_estimators=500).fit(train_x, train_y)

test_pred = model.predict(test_x)

accuracy_score(test_y, test_pred)
from sklearn.feature_extraction.text import TfidfVectorizer



train_X,test_X,train_Y,test_Y=train_test_split(docs,

                                               hotstar['Sentiment_Manual'],

                                               test_size=0.2,

                                               random_state=100)

tfid = TfidfVectorizer(min_df=2).fit(train_X)

train_X = tfid.transform(train_X)

test_X = tfid.transform(test_X)



features = tfid.get_feature_names()

train_X = pd.DataFrame(train_X.toarray(), columns=features)

test_X = pd.DataFrame(test_X.toarray(), columns=features)
model_tfid_mnb = MultinomialNB().fit(train_X,train_Y)

test_pred_tfid_mnb = model_tfid_mnb.predict(test_X)

print(accuracy_score(test_Y,test_pred_tfid_mnb))

# mnb cannot take negative values
docs_vectors = pd.DataFrame()

for doc in docs:

    words = nltk.word_tokenize(doc)

    temp = pd.DataFrame()

    for word in words:

        try:

            word_vec = embeddings[word]

            temp = temp.append(pd.Series(word_vec), ignore_index=True)

        except:

            pass

    docs_vectors = docs_vectors.append(temp.mean(), ignore_index = True)

docs_vectors.shape
# Check for null values

null_values=pd.DataFrame(pd.isnull(docs_vectors).sum(axis = 1).sort_values(ascending = False))
null_values.head()
null_list = null_values.index[null_values[0]==300].tolist()
len(null_list) # Checking the length of null values
X = docs_vectors.drop(null_list)

y = hotstar['Sentiment_Manual'].drop(null_list)
from sklearn.model_selection import train_test_split

train_x,test_x,train_y,test_y = train_test_split(X,y, test_size = 0.2, random_state = 100)
model_rf = RandomForestClassifier(n_estimators=100).fit(train_x, train_y)

test_pred_rf = model_rf.predict(test_x)

accuracy_score(test_y, test_pred_rf)
model_ab = AdaBoostClassifier(n_estimators=100).fit(train_x, train_y)

test_pred_ab = model_ab.predict(test_x)

accuracy_score(test_y, test_pred_ab)
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
get_sentiment(hotstar.loc[1, 'Lower_Case_Reviews'])
get_sentiment(hotstar.loc[6, 'Lower_Case_Reviews'])
hotstar['sentiment_vader'] = hotstar['Lower_Case_Reviews'].apply(get_sentiment)
accuracy_score(hotstar['Sentiment_Manual'],hotstar['sentiment_vader'])