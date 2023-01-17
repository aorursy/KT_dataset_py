import pandas as pd

import numpy as np

import re

import os

from IPython.display import HTML



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction import text 

from sklearn.decomposition import PCA





import nltk

from nltk.stem.porter import PorterStemmer

from nltk.stem.snowball import SnowballStemmer

from nltk.corpus import words

from nltk.corpus import wordnet 

allEnglishWords = words.words() + [w for w in wordnet.words()]

allEnglishWords = np.unique([x.lower() for x in allEnglishWords])



import matplotlib.pyplot as plt

import plotly.offline as py

import plotly.graph_objs as go

py.init_notebook_mode(connected=True)



import warnings

warnings.filterwarnings('ignore')
amazon_data  = pd.read_csv('../input/amazon_cells_labelled.csv', sep=',')

imdb_data = pd.read_csv('../input/imdb_labelled.csv', sep=',')

yelp_data = pd.read_csv('../input/yelp_labelled.csv', sep=',')
print(amazon_data.shape)

print(imdb_data.shape)

print(yelp_data.shape)
combine_df = pd.concat([amazon_data,imdb_data,yelp_data])

print(combine_df.shape)
combine_df.shape
HTML(combine_df.Reviews.iloc[0])
class Preprocessor(object):

    ''' Preprocess data for NLP tasks. '''



    def __init__(self, alpha=True, lower=True, stemmer=True, english=False):

        self.alpha = alpha

        self.lower = lower

        self.stemmer = stemmer

        self.english = english

        

        self.uniqueWords = None

        self.uniqueStems = None

        

    def fit(self, texts):

        texts = self._doAlways(texts)



        allwords = pd.DataFrame({"word": np.concatenate(texts.apply(lambda x: x.split()).values)})

        self.uniqueWords = allwords.groupby(["word"]).size().rename("count").reset_index()

        self.uniqueWords = self.uniqueWords[self.uniqueWords["count"]>1]

        if self.stemmer:

            self.uniqueWords["stem"] = self.uniqueWords.word.apply(lambda x: PorterStemmer().stem(x)).values

            self.uniqueWords.sort_values(["stem", "count"], inplace=True, ascending=False)

            self.uniqueStems = self.uniqueWords.groupby("stem").first()

        

        #if self.english: self.words["english"] = np.in1d(self.words["mode"], allEnglishWords)

        print("Fitted.")

            

    def transform(self, texts):

        texts = self._doAlways(texts)

        if self.stemmer:

            allwords = np.concatenate(texts.apply(lambda x: x.split()).values)

            uniqueWords = pd.DataFrame(index=np.unique(allwords))

            uniqueWords["stem"] = pd.Series(uniqueWords.index).apply(lambda x: PorterStemmer().stem(x)).values

            uniqueWords["mode"] = uniqueWords.stem.apply(lambda x: self.uniqueStems.loc[x, "word"] if x in self.uniqueStems.index else "")

            texts = texts.apply(lambda x: " ".join([uniqueWords.loc[y, "mode"] for y in x.split()]))

        #if self.english: texts = self.words.apply(lambda x: " ".join([y for y in x.split() if self.words.loc[y,"english"]]))

        print("Transformed.")

        return(texts)



    def fit_transform(self, texts):

        texts = self._doAlways(texts)

        self.fit(texts)

        texts = self.transform(texts)

        return(texts)

    

    def _doAlways(self, texts):

        # Remove parts between <>'s

        texts = texts.apply(lambda x: re.sub('<.*?>', ' ', x))

        # Keep letters and digits only.

        if self.alpha: texts = texts.apply(lambda x: re.sub('[^a-zA-Z0-9 ]+', ' ', x))

        # Set everything to lower case

        if self.lower: texts = texts.apply(lambda x: x.lower())

        return texts  
preprocess = Preprocessor(alpha=True, lower=True, stemmer=True)
train = combine_df
%%time

trainX = preprocess.fit_transform(train.Reviews)

print(preprocess.uniqueWords.shape)

preprocess.uniqueWords[preprocess.uniqueWords.word.str.contains("bad")]
uniqueWordsdf = preprocess.uniqueWords.sort_values(by='count', ascending = False)
uniqueWordsdf
stop_words = text.ENGLISH_STOP_WORDS.union(["thats","the","weve","dont","lets","youre","im","thi","ha",

    "wa","st","ask","want","thank","know","susan","ryan","say","got","ought","ive","theyre","i","a","is"])

#stop_words = text.ENGLISH_STOP_WORDS

tfidf = TfidfVectorizer(min_df=3, max_features=1000, stop_words=stop_words) #, ngram_range=(1,3)
%%time

trainX = tfidf.fit_transform(trainX).toarray()
print(trainX.shape)
trainY = train.Sentiment
print(trainX.shape, trainY.shape)
from scipy.stats.stats import pearsonr
getCorrelation = np.vectorize(lambda x: pearsonr(trainX[:,x], trainY)[0])

correlations = getCorrelation(np.arange(trainX.shape[1]))

print(correlations)
allIndeces = np.argsort(-correlations)

bestIndeces = allIndeces[np.concatenate([np.arange(300), np.arange(-300, 0)])]
vocabulary = np.array(tfidf.get_feature_names())

print(vocabulary[bestIndeces][:10])

print(vocabulary[bestIndeces][-10:])
trainX = trainX[:,bestIndeces]
print(trainX.shape, trainY.shape)
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import GridSearchCV



mlp = MLPClassifier( validation_fraction = 0.1,learning_rate_init=0.01,early_stopping = True, verbose=True)



params = {'activation':('tanh','relu'), 

         'batch_size':[400],

         'hidden_layer_sizes':[(4,50)],

         'max_iter':[50]}

clf = GridSearchCV(mlp, params, cv = 2)

from sklearn.model_selection import cross_validate

mlp1 = MLPClassifier()

cross_validate(mlp1, trainX, trainY)
clf.fit(trainX,trainY)
clf.best_estimator_
clf.best_params_
clf.score(trainX,trainY)
train["prediction"] = clf.predict(trainX)

train["truth"] = train.Sentiment

train.head()
print((train.truth==train.prediction).mean())
trainCross = train.groupby(["prediction", "truth"]).size().unstack()

trainCross
unseen = pd.Series("i liked it")
unseen = preprocess.transform(unseen)       # Text preprocessing

unseen = tfidf.transform(unseen).toarray()  # Feature engineering

unseen = unseen[:,bestIndeces]              # Feature selection

probability = clf.predict(unseen)
print(probability)

print("Positive!") if probability == 1 else print("Negative!")
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

rfc.fit(trainX,trainY)
rfc.score(trainX,trainY)
train["prediction"] = rfc.predict(trainX)

train["truth"] = train.Sentiment

train.head()
trainCross = train.groupby(["prediction", "truth"]).size().unstack()

trainCross
testData = pd.read_csv('../input/testData.csv')
#unseen = pd.Series("awsome")



unseen = preprocess.transform(testData.Reviews)       # Text preprocessing

unseen = tfidf.transform(unseen).toarray()  # Feature engineering

unseen = unseen[:,bestIndeces]              # Feature selection

probability = clf.predict(unseen)



testData['prediction'] = probability

testData.head()
trainCross = testData.groupby(["prediction", "Sentiment"]).size().unstack()

trainCross
testScore = (testData.Sentiment==testData.prediction).mean()

testScore