import pandas as pd
tweets = pd.read_csv("../input/Tweets.csv")
tweets.head()
# extract only the text and airline_sentiment columns

df = tweets[['text','airline_sentiment']]
df.isnull().sum()
df.head()
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import cross_val_score

from sklearn.pipeline import Pipeline
X = df['text']

y = df['airline_sentiment']
#encode sentiment categories

le = LabelEncoder()

le.fit_transform(y)
df['airline_sentiment'].head()

# encoding: neutral = 1, positive = 2, negative = 0
naive_pipe = Pipeline([

    ('cv', CountVectorizer()),

    ('tfidf', TfidfTransformer()),

    ('lm', LogisticRegression())

    

])



scores = cross_val_score(naive_pipe, X,y, cv = 5)

print('Mean score: ',scores.mean())
from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

import string

from sklearn.base import BaseEstimator,TransformerMixin
class PunctuationRemover(BaseEstimator,TransformerMixin):

    def fit(self, column, y = None):

        return self

    

    def removePunctuation(self,text,punctuation, y = None):

        clean_words = []

        

        for element in word_tokenize(text):

            if element not in punctuation:

                clean_words.append(element)



        clean_text = ' '.join(clean_words)

        return clean_text

    

    def transform(self, column, y = None):

        punctuation = set(string.punctuation)

        return column.apply(lambda x: self.removePunctuation(x,punctuation))
class StopwordRemover(BaseEstimator,TransformerMixin):

    def fit(self, column, y = None):

        return self

    

    def removeStopwords(self,text,stop_words, y = None):

        clean_words = []

        

        for element in text.lower().split():

            if element not in stop_words:

                clean_words.append(element)



        clean_text = ' '.join(clean_words)

        return clean_text

    

    def transform(self, column, y = None):

        stop_words = set(stopwords.words('english'))

        return column.apply(lambda x: self.removeStopwords(x,stop_words))
# visualize the newly-transformed text

visualization_pipe = Pipeline([

    ('sw', StopwordRemover()),

    ('punc', PunctuationRemover()),

])



pd.DataFrame(visualization_pipe.fit_transform(X)).head()
pipe2 = Pipeline([

    ('sw', StopwordRemover()),

    ('punc', PunctuationRemover()),

    ('cv', CountVectorizer()),

    ('tfidf', TfidfTransformer()),

    ('lm', LogisticRegression())

    

])



scores = cross_val_score(pipe2, X,y, cv = 5)

print('Mean score: ',scores.mean())
smileys = [r'=(',r'=)',r':)',r':(']



for text in df['text']:

    for smiley in smileys:

        if smiley in text:

            print (text)

    