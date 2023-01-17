import pandas as pd
import numpy as np
import json
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
import re
from sklearn.linear_model import LogisticRegression
trainData = pd.read_json('../input/train.json')
testData = pd.read_json('../input/test.json')
trainData.head()
lemma=WordNetLemmatizer()
def clean_review(review_col):
    review_corpus=[]
    for i in range(0,len(review_col)):
        review=str(review_col[i])
        review=re.sub('[^a-zA-Z]',' ',review)
        review=[lemma.lemmatize(w) for w in word_tokenize(str(review).lower())]
        review=' '.join(review)
        review_corpus.append(review)
    return review_corpus
trainData['clean_ingredients']=clean_review(trainData.ingredients.values)
testData['clean_ingredients']=clean_review(testData.ingredients.values)
trainData.head()
tfidf=TfidfVectorizer()
xtrain=tfidf.fit_transform(trainData.clean_ingredients).toarray()
xtest=tfidf.transform(testData.clean_ingredients).toarray()
lb = LabelEncoder()
yTrain = lb.fit_transform(trainData['cuisine'])
vclf=VotingClassifier(estimators=[('clf1',LogisticRegression(C=10,dual=False)),('clf2',SVC(C=100,gamma=1,kernel='rbf',probability=True))],voting='soft',weights=[1,2])
vclf.fit(xtrain , yTrain)
vclf.score(xtrain , yTrain)
pred = vclf.predict(xtest)
ypred = lb.inverse_transform(pred)
sub = pd.DataFrame({'id': testData['id'], 'cuisine': ypred})
sub.to_csv('svm_output.csv', index=False)
