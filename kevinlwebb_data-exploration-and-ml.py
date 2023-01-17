import os
print(os.listdir("../input/twitter-hate-speech"))
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import nltk
from wordcloud import WordCloud,STOPWORDS
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline

import warnings 
warnings.filterwarnings("ignore")
train  = pd.read_csv("../input/twitter-hate-speech/train_E6oV3lV.csv")
test = pd.read_csv("../input/twitter-hate-speech/test_tweets_anuFYb8.csv")
train.head()
test.head()
train['cleaned_tweet'] = train.tweet.apply(lambda x: ' '.join([word for word in x.split() if not word.startswith('@')]))
test['cleaned_tweet'] = test.tweet.apply(lambda x: ' '.join([word for word in x.split() if not word.startswith('@')]))

train['hashtags'] = train.tweet.apply(lambda x: ' '.join([word[1:] for word in x.split() if word.startswith('#')]))
test['hashtags'] = test.tweet.apply(lambda x: ' '.join([word[1:] for word in x.split() if word.startswith('#')]))
train.head()
import nltk
nltk.download(['punkt', 'wordnet'])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
def load_data(path):
    df = pd.read_csv(path)
    
    df['cleaned_tweet'] = df.tweet.apply(lambda x: ' '.join([word for word in x.split() if not word.startswith('@')]))
    df['cleaned_tweet'] = df.cleaned_tweet.str.replace("#","")
    
    X = df.cleaned_tweet.values
    y = df.label.values
    
    return X, y

def tokenize(text):

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def display_results(y_test, y_pred):
    labels = np.unique(y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred, labels=labels)
    accuracy = (y_pred == y_test).mean()

    print("Labels:", labels)
    print("Confusion Matrix:\n", confusion_mat)
    print("Accuracy:", accuracy)
    
def main():
    url = "../input/twitter-hate-speech/train_E6oV3lV.csv"
    X, y = load_data(url)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    vect = CountVectorizer(tokenizer=tokenize)
    tfidf = TfidfTransformer()
    clf = RandomForestClassifier()

    # train classifier
    X_train_counts = vect.fit_transform(X_train)
    X_train_tfidf = tfidf.fit_transform(X_train_counts)
    clf.fit(X_train_tfidf, y_train)

    # predict on test data
    X_test_counts = vect.transform(X_test)
    X_test_tfidf = tfidf.transform(X_test_counts)
    y_pred = clf.predict(X_test_tfidf)
    
    # predict on test data
    X_test_counts = vect.transform(["whoa stop you stupid sjw"])
    X_test_tfidf = tfidf.transform(X_test_counts)
    print("Given text: 'whoa stop you stupid sjw' ")
    print("Prediction: {}\n".format(clf.predict(X_test_tfidf)))
    

    # display results
    display_results(y_test, y_pred)


main()