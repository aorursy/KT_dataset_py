# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from IPython.display import HTML

import re

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



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



import plotly.offline as py

import plotly.graph_objs as go

py.init_notebook_mode(connected=True)



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv')
selected_df = train_df.groupby('SentenceId')['Phrase','Sentiment'].head(6)
train = selected_df
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
%%time

trainX = preprocess.fit_transform(train.Phrase)

stop_words = text.ENGLISH_STOP_WORDS.union(["thats","weve","dont","lets","youre","im","thi","ha",

    "wa","st","ask","want","like","thank","know","susan","ryan","say","got","ought","ive","theyre"])

tfidf = TfidfVectorizer(min_df=2, max_features=10000, stop_words=stop_words) #, ngram_range=(1,3)

%%time

trainX = tfidf.fit_transform(trainX).toarray()
trainY = train.Sentiment
from scipy.stats.stats import pearsonr

getCorrelation = np.vectorize(lambda x: pearsonr(trainX[:,x], trainY)[0])

correlations = getCorrelation(np.arange(trainX.shape[1]))

print(correlations)

allIndeces = np.argsort(-correlations)

bestIndeces = allIndeces[np.concatenate([np.arange(6000), np.arange(-6000, 0)])]
vocabulary = np.array(tfidf.get_feature_names())

print(vocabulary[bestIndeces][:10])

print(vocabulary[bestIndeces][-10:])
trainX = trainX[:,bestIndeces]

print(trainX.shape, trainY.shape)
from sklearn.neural_network import MLPClassifier



mlp = MLPClassifier(hidden_layer_sizes= (100,) ,batch_size=5000, max_iter = 80, 

                    validation_fraction = 0.3,learning_rate_init=0.01,

                    early_stopping = True, activation='tanh', verbose=True)
mlp.fit(trainX,trainY)

mlp.score(trainX,trainY)
train["Prediction"] = mlp.predict(trainX)

train["Result"] = train.Sentiment==train.Prediction

train.head()

trainCross = train.groupby(["Sentiment", "Prediction"]).size().unstack()

trainCross