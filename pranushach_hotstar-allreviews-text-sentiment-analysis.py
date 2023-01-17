import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import gensim

import nltk
print(os.listdir("../input/embeddings/GoogleNews-vectors-negative300/"))

path = "../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin"

embeddings = gensim.models.KeyedVectors.load_word2vec_format(path , binary = True)
url = 'https://raw.githubusercontent.com/skathirmani/datasets/master/hotstar.allreviews_Sentiments.csv'

df_hotstar = pd.read_csv(url)

df_hotstar['Sentiment_Manual'].head()

df_hotstar.head()
df_hotstar['Sentiment_Manual'].value_counts()
nltk.download('stopwords')

nltk.download('vader_lexicon')

nltk.download('punkt')
Neutral = df_hotstar[df_hotstar['Sentiment_Manual'] == 'Neutral']

Positive = df_hotstar[df_hotstar['Sentiment_Manual'] == 'Positive']

Negative = df_hotstar[df_hotstar['Sentiment_Manual'] == 'Negative']
Docs1 = Neutral['Lower_Case_Reviews']

print(len(Docs1))



Docs2 = Positive['Lower_Case_Reviews']

print(len(Docs2))



Docs3 = Negative['Lower_Case_Reviews']

print(len(Docs3))
! pip install wordcloud
from wordcloud import WordCloud

import matplotlib.pyplot as plt

%matplotlib inline
StopWords = nltk.corpus.stopwords.words('english')
WC_Neutral = WordCloud(background_color = 'white' , stopwords = StopWords).generate('' . join(Docs1))

plt.imshow(WC_Neutral)
WC_Positive = WordCloud(background_color = 'white' , stopwords = StopWords).generate('' . join(Docs2))

plt.imshow(WC_Positive)
WC_Negative = WordCloud(background_color = 'white' , stopwords = StopWords).generate('' . join(Docs3))

plt.imshow(WC_Negative)
Docs = df_hotstar['Lower_Case_Reviews']

Docs = Docs.str.replace('-' , ' ').str.lower().str.replace('[^a-z ]' , ' ')
Docs.head()
StopWords = nltk.corpus.stopwords.words('english')

clean_sentence = lambda doc: ' '.join([word for word in nltk.word_tokenize(doc) if word not in StopWords])

Docs_clean = Docs.apply(clean_sentence)

Docs_clean.head()
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB , BernoulliNB

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier
x_train , x_test , y_train , y_test = train_test_split(Docs_clean , df_hotstar['Sentiment_Manual'] , 

                                                       test_size = 0.2 , random_state = 100)
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer(min_df = 5).fit(x_train)

x_train = vec.transform(x_train)

x_test = vec.transform(x_test)
model_mnb = MultinomialNB().fit(x_train , y_train)

test_pred_mnb = model_mnb.predict(x_test)

print(accuracy_score(y_test , test_pred_mnb))
model_ab = AdaBoostClassifier(n_estimators = 100 , random_state = 99).fit(x_train , y_train)

test_pred_ab = model_ab.predict(x_test)

print(accuracy_score(y_test , test_pred_ab))
model_rf = RandomForestClassifier(n_estimators = 100 , random_state = 99).fit(x_train , y_train)

test_pred_rf = model_rf.predict(x_test)

print(accuracy_score(y_test , test_pred_rf))
model_gb = GradientBoostingClassifier(n_estimators = 100 , random_state = 99).fit(x_train , y_train)

test_pred_gb = model_gb.predict(x_test)

print(accuracy_score(y_test , test_pred_gb))
from sklearn.feature_extraction.text import TfidfVectorizer
x_train , x_test , y_train , y_test = train_test_split(Docs_clean , df_hotstar['Sentiment_Manual'] , 

                                                       test_size = 0.2 , random_state = 100)

tfidf = TfidfVectorizer(min_df = 5).fit(x_train)

x_train = tfidf.transform(x_train)

x_test = tfidf.transform(x_test)
model_mnb = MultinomialNB().fit(x_train , y_train)

test_pred_mnb = model_mnb.predict(x_test)

print(accuracy_score(y_test , test_pred_mnb))
docs_vectors = pd.DataFrame() # document-Term Matrix

for doc in Docs_clean:

    words = nltk.word_tokenize(doc)

    temp = pd.DataFrame()

    for word in words:

        try:

            word_vec = embeddings[word]

            temp = temp.append(pd.Series(word_vec) , ignore_index = True)

        except:

            pass

    docs_vectors = docs_vectors.append(temp.mean() , ignore_index = True)

docs_vectors.shape
null_vec = pd.DataFrame(pd.isnull(docs_vectors).sum(axis = 1).sort_values(ascending = False))
null_vec.head()
nl = null_vec.index[null_vec[0]==300].tolist()
len(nl)
x = docs_vectors.drop(nl)

y = df_hotstar['Sentiment_Manual'].drop(nl)
x.shape , y.shape
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.2 , random_state = 100)
model_rf = RandomForestClassifier(n_estimators = 100).fit(x_train , y_train)

test_pred_rf = model_rf.predict(x_test)

print(accuracy_score(y_test , test_pred_rf))
model_ab =AdaBoostClassifier(n_estimators = 100).fit(x_train , y_train)

test_pred_ab = model_ab.predict(x_test)

print(accuracy_score(y_test , test_pred_ab))
model_gb = GradientBoostingClassifier(n_estimators = 100).fit(x_train , y_train)

test_pred_gb = model_gb.predict(x_test)

print(accuracy_score(y_test , test_pred_gb))
from nltk.sentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()



def get_sentiment (sentence , analyzer = analyzer):

    compound = analyzer.polarity_scores(sentence)['compound']

    if compound > 0.1:

        return 'Positive'

    elif compound < 0.1:

        return 'Negative'

    else:

        return 'Neutral'    
df_hotstar = df_hotstar.drop(['Sentiment_Vader'] , axis = 1)
df_hotstar.head(2)
df_hotstar['Sentiment_Vader'] = df_hotstar['Reviews'].apply(get_sentiment)
accuracy_score(df_hotstar['Sentiment_Manual'] , df_hotstar['Sentiment_Vader'])
df_hotstar.head(2)