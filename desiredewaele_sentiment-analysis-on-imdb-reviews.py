import pandas as pd
import numpy as np
import re
import os
from IPython.display import HTML

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text 
from sklearn.decomposition import PCA

from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras import optimizers

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

import warnings
warnings.filterwarnings('ignore')
path = "/kaggle/input/aclimdb/aclImdb/"
positiveFiles = [x for x in os.listdir(path+"train/pos/") if x.endswith(".txt")]
negativeFiles = [x for x in os.listdir(path+"train/neg/") if x.endswith(".txt")]
testFiles = [x for x in os.listdir(path+"test/") if x.endswith(".txt")]
positiveReviews, negativeReviews, testReviews = [], [], []
for pfile in positiveFiles:
    with open(path+"train/pos/"+pfile, encoding="latin1") as f:
        positiveReviews.append(f.read())
for nfile in negativeFiles:
    with open(path+"train/neg/"+nfile, encoding="latin1") as f:
        negativeReviews.append(f.read())
for tfile in testFiles:
    with open(path+"test/"+tfile, encoding="latin1") as f:
        testReviews.append(f.read())
reviews = pd.concat([
    pd.DataFrame({"review":positiveReviews, "label":1, "file":positiveFiles}),
    pd.DataFrame({"review":negativeReviews, "label":0, "file":negativeFiles}),
    pd.DataFrame({"review":testReviews, "label":-1, "file":testFiles})
], ignore_index=True).sample(frac=1, random_state=1)
reviews.head()
reviews = reviews[["review", "label", "file"]].sample(frac=1, random_state=1)
train = reviews[reviews.label!=-1].sample(frac=0.6, random_state=1)
valid = reviews[reviews.label!=-1].drop(train.index)
test = reviews[reviews.label==-1]
print(train.shape)
print(valid.shape)
print(test.shape)
HTML(train.review.iloc[0])
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
train.head()
preprocess = Preprocessor(alpha=True, lower=True, stemmer=True)
%%time
trainX = preprocess.fit_transform(train.review)
validX = preprocess.transform(valid.review)
trainX.head()
print(preprocess.uniqueWords.shape)
preprocess.uniqueWords[preprocess.uniqueWords.word.str.contains("disappoint")]
print(preprocess.uniqueStems.shape)
preprocess.uniqueStems[preprocess.uniqueStems.word.str.contains("disappoint")]
stop_words = text.ENGLISH_STOP_WORDS.union(["thats","weve","dont","lets","youre","im","thi","ha",
    "wa","st","ask","want","like","thank","know","susan","ryan","say","got","ought","ive","theyre"])
tfidf = TfidfVectorizer(min_df=2, max_features=10000, stop_words=stop_words) #, ngram_range=(1,3)
%%time
trainX = tfidf.fit_transform(trainX).toarray()
validX = tfidf.transform(validX).toarray()
print(trainX.shape)
print(validX.shape)
trainY = train.label
validY = valid.label
print(trainX.shape, trainY.shape)
print(validX.shape, validY.shape)
from scipy.stats.stats import pearsonr
getCorrelation = np.vectorize(lambda x: pearsonr(trainX[:,x], trainY)[0])
correlations = getCorrelation(np.arange(trainX.shape[1]))
print(correlations)
allIndeces = np.argsort(-correlations)
bestIndeces = allIndeces[np.concatenate([np.arange(1000), np.arange(-1000, 0)])]
vocabulary = np.array(tfidf.get_feature_names())
print(vocabulary[bestIndeces][:10])
print(vocabulary[bestIndeces][-10:])
trainX = trainX[:,bestIndeces]
validX = validX[:,bestIndeces]
print(trainX.shape, trainY.shape)
print(validX.shape, validY.shape)
DROPOUT = 0.5
ACTIVATION = "tanh"

model = Sequential([    
    Dense(int(trainX.shape[1]/2), activation=ACTIVATION, input_dim=trainX.shape[1]),
    Dropout(DROPOUT),
    Dense(int(trainX.shape[1]/2), activation=ACTIVATION, input_dim=trainX.shape[1]),
    Dropout(DROPOUT),
    Dense(int(trainX.shape[1]/4), activation=ACTIVATION),
    Dropout(DROPOUT),
    Dense(100, activation=ACTIVATION),
    Dropout(DROPOUT),
    Dense(20, activation=ACTIVATION),
    Dropout(DROPOUT),
    Dense(5, activation=ACTIVATION),
    Dropout(DROPOUT),
    Dense(1, activation='sigmoid'),
])
model.compile(optimizer=optimizers.Adam(0.00005), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
EPOCHS = 30
BATCHSIZE = 1500
model.fit(trainX, trainY, epochs=EPOCHS, batch_size=BATCHSIZE, validation_data=(validX, validY))
x = np.arange(EPOCHS)
history = model.history.history

data = [
    go.Scatter(x=x, y=history["acc"], name="Train Accuracy", marker=dict(size=5), yaxis='y2'),
    go.Scatter(x=x, y=history["val_acc"], name="Valid Accuracy", marker=dict(size=5), yaxis='y2'),
    go.Scatter(x=x, y=history["loss"], name="Train Loss", marker=dict(size=5)),
    go.Scatter(x=x, y=history["val_loss"], name="Valid Loss", marker=dict(size=5))
]
layout = go.Layout(
    title="Model Training Evolution", font=dict(family='Palatino'), xaxis=dict(title='Epoch', dtick=1),
    yaxis1=dict(title="Loss", domain=[0, 0.45]), yaxis2=dict(title="Accuracy", domain=[0.55, 1]),
)
py.iplot(go.Figure(data=data, layout=layout), show_link=False)
train["probability"] = model.predict(trainX)
train["prediction"] = train.probability-0.5>0
train["truth"] = train.label==1
train.tail()
print(model.evaluate(trainX, trainY))
print((train.truth==train.prediction).mean())
valid["probability"] = model.predict(validX)
valid["prediction"] = valid.probability-0.5>0
valid["truth"] = valid.label==1
valid.tail()
print(model.evaluate(validX, validY))
print((valid.truth==valid.prediction).mean())
trainCross = train.groupby(["prediction", "truth"]).size().unstack()
trainCross
validCross = valid.groupby(["prediction", "truth"]).size().unstack()
validCross
truepositives = valid[(valid.truth==True)&(valid.truth==valid.prediction)]
print(len(truepositives), "true positives.")
truepositives.sort_values("probability", ascending=False).head(3)
truenegatives = valid[(valid.truth==False)&(valid.truth==valid.prediction)]
print(len(truenegatives), "true negatives.")
truenegatives.sort_values("probability", ascending=True).head(3)
falsepositives = valid[(valid.truth==True)&(valid.truth!=valid.prediction)]
print(len(falsepositives), "false positives.")
falsepositives.sort_values("probability", ascending=True).head(3)
falsenegatives = valid[(valid.truth==False)&(valid.truth!=valid.prediction)]
print(len(falsenegatives), "false negatives.")
falsenegatives.sort_values("probability", ascending=False).head(3)
HTML(valid.loc[22148].review)
unseen = pd.Series("this movie very good")
unseen = preprocess.transform(unseen)       # Text preprocessing
unseen = tfidf.transform(unseen).toarray()  # Feature engineering
unseen = unseen[:,bestIndeces]              # Feature selection
probability = model.predict(unseen)[0,0]  # Network feedforward
print(probability)
print("Positive!") if probability > 0.5 else print("Negative!")