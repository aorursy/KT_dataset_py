from nltk import word_tokenize

import pandas as pd

from nltk.classify.util import apply_features,accuracy

from nltk.classify.scikitlearn import SklearnClassifier

from sklearn.naive_bayes import MultinomialNB

from nltk.corpus import stopwords

from random import shuffle

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt
df = pd.read_csv("../input/train.csv")
X_text=df['text']

y_author=df['author']
def getTextAuthorFeatures(words):

    uniqueWords=set(words)

    return dict({word:True for word in uniqueWords})
sw=set(stopwords.words('english'))

from nltk.stem import PorterStemmer

featureList=[]

ps=PorterStemmer()

for entry,author in zip(X_text,y_author):

    wordList=[ps.stem(w.lower()) for w in word_tokenize(entry) if len(w)>=2 and w not in sw]

    featureList.append([getTextAuthorFeatures(wordList),author])



shuffle(featureList)

X_train=apply_features(getTextAuthorFeatures,featureList[:int(len(featureList)*0.70)])

X_test=apply_features(getTextAuthorFeatures,featureList[int(len(featureList)*0.70):])



mnb=SklearnClassifier(MultinomialNB())

mnb.train(X_train)

print('Accuracy%:',accuracy(mnb,X_test)*100)

test=pd.read_csv("../input/test.csv")

finalProbList=[]

for idx,row in test.iterrows():

    text=row['text']

    predicted_author=mnb.classify(getTextAuthorFeatures([w.lower() for w in word_tokenize(text) ]))

    predicted_proba=mnb.prob_classify(getTextAuthorFeatures([w.lower() for w in word_tokenize(text) ]))

    finalProbList.append([row['id'],[predicted_proba.prob(i) for i in predicted_proba.samples()]])
finalProbList
from sklearn.feature_extraction.text import TfidfVectorizer
vec=TfidfVectorizer(ngram_range=(1,3),stop_words='english')
vecText=vec.fit_transform(X_text)
from sklearn.decomposition import TruncatedSVD
svd=TruncatedSVD(n_components=np.unique(y_author).shape[0])
reducedX=svd.fit_transform(vecText)
dfSVD=pd.DataFrame(reducedX)

dfSVD['author']=y_author
X=vecText

y=dfSVD['author']
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
from sklearn.naive_bayes import MultinomialNB

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split
mnb=MultinomialNB()
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.70,random_state=42)
mlpClf=MLPClassifier(random_state=42,verbose=Tr)
mlpClf.fit(X_train,y_train)
mlpClf.score(X_test,y_test)*100