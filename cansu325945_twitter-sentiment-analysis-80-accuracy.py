!pip install wordninja

import pandas as pd

import numpy as np

from nltk.corpus import stopwords

import re

import string

import emoji

import spacy

import wordninja #Someone else implemented this ->  https://github.com/keredson/wordninja

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

import xgboost

from xgboost import XGBClassifier

pd.set_option('display.max_colwidth', 135)

#!python -m spacy download en

nlp =spacy.load('en_core_web_sm')

from sklearn.ensemble import GradientBoostingClassifier

from catboost import CatBoostClassifier
tweets_file = '../input/airline-sentiment/Tweets.csv'

data=pd.read_csv(tweets_file)

print ("data_shape: ", data.shape)

data.head(2)
def clean_and_tokenize_tweets(input_text):

    

    #remove_mentions, urls, hash_sign:

    mention_words_removed= re.sub(r'@\w+','',input_text)

    hash_sign_removed=re.sub(r'#','',mention_words_removed)

    url_removed=' '.join(word for word in hash_sign_removed.split(" ") if not word.startswith('http'))

    

    #Transform emoji to text

    demoj=emoji.demojize(url_removed)

    

    #Split compound words coming from hashtags

    splitted=wordninja.split(demoj)

    splitted=" ".join(word for word in splitted)

    

    # Implement lemmatization & remove punctuation

    lem = nlp(splitted)

    punctuations = string.punctuation

    punctuations=punctuations+'...'



    sentence=[]

    for word in lem:

        word = word.lemma_.lower().strip()

        if ((word != '-pron-') & (word not in punctuations)):

            sentence.append(word)    

            

    #Remove stopwords

    stop_words=set(stopwords.words('english'))

    stop_words_removed=[word for word in sentence if word not in stop_words]

    

    return stop_words_removed
data["text_"]=data["text"].apply(clean_and_tokenize_tweets)

data[["text","text_"]].head()
data["text_"]=[" ".join(word) for word in data["text_"]]

X_train, X_test=data["text_"][:10000],data["text_"][10000:]

tfidf_vector = TfidfVectorizer()

X_train=tfidf_vector.fit_transform(X_train)

X_test = tfidf_vector.transform(X_test)

ylabels=data["airline_sentiment"].map({"negative":-1,"neutral":0,"positive":1})

y_train, y_test=ylabels[:10000],ylabels[10000:] 
base_classifier=LogisticRegression(random_state=0, solver='lbfgs',max_iter=500,multi_class='multinomial').fit(X_train,y_train)

pred_train_base=base_classifier.predict(X_train)

pred_test_base=base_classifier.predict(X_test)
print("Logistic Regression Train Accuracy:",np.round(metrics.accuracy_score(y_train, pred_train_base),4))

print("Logistic Regression Test Accuracy:",np.round(metrics.accuracy_score(y_test, pred_test_base),4))

print("")

print("Logistic Regression Confusion Matrix:",metrics.confusion_matrix(y_test, pred_test_base,labels=[-1, 0, 1]),sep="\n")
gbm = GradientBoostingClassifier(n_estimators=180, max_depth=6, random_state=0,learning_rate=0.1)

gbm.fit(X_train, y_train)

pred_train_gbm=gbm.predict(X_train)

pred_test_gbm=gbm.predict(X_test)
print("LGBM Train Accuracy:",np.round(metrics.accuracy_score(y_train, pred_train_gbm),4))

print("LGBM Test Accuracy:",np.round(metrics.accuracy_score(y_test, pred_test_gbm),4))

print("")

print("LGBM Confusion Matrix:",metrics.confusion_matrix(y_test, pred_test_gbm,labels=[-1, 0, 1]),sep="\n")
xgb_classifier = XGBClassifier(n_estimators=200,random_state=0,learning_rate=0.7,objective='multi:softprob',num_class=3)

xgb_classifier.fit(X_train, y_train)

pred_train_xgb=xgb_classifier.predict(X_train)

pred_test_xgb=xgb_classifier.predict(X_test)
print("XGBoost train Accuracy:",metrics.accuracy_score(y_train, pred_train_xgb))

print("XGBoost test Accuracy:",np.round(metrics.accuracy_score(y_test, pred_test_xgb),4))

print("")

print("XGBoost Confusion Matrix:",metrics.confusion_matrix(y_test, pred_test_xgb,labels=[-1, 0, 1]),sep="\n")
catboost_classifier = CatBoostClassifier(iterations=500, learning_rate=0.5, l2_leaf_reg=3.5, depth=8, rsm=0.98, eval_metric='AUC',use_best_model=True,random_seed=42,loss_function='MultiClass')

catboost_classifier.fit(X_train,y_train,eval_set=(X_test,y_test))

pred_train_catb = catboost_classifier.predict(X_train)

pred_test_catb = catboost_classifier.predict(X_test)
print("CatBoost train Accuracy:",metrics.accuracy_score(y_train, pred_train_catb))

print("CatBoost test Accuracy:",np.round(metrics.accuracy_score(y_test, pred_test_catb),4))

print("")

print("CatBoost Confusion Matrix:",metrics.confusion_matrix(y_test, pred_test_catb,labels=[-1, 0, 1]),sep="\n")