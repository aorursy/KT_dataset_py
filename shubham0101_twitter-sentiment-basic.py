import numpy as np 
import pandas as pd 
train = pd.read_csv("../input/train_E6oV3lV.csv")
test = pd.read_csv("../input/test_tweets_anuFYb8.csv")
train.head()
import string

punctuations = string.punctuation

from nltk.corpus import stopwords

stopword = stopwords.words("english")

from nltk.stem.wordnet import WordNetLemmatizer

lem = WordNetLemmatizer()

# creating a function to clean the text
def _clean(text):
    
    #converting to lower case 
    clean_text = text.lower()
    
    # removing punctuations
    clean_text = "". join(p for p in clean_text if p not in punctuations)
    
    # removing stopwords
    words = clean_text.split()
    words =[w for w in words if w not in stopword]
    
    # lemmatization
    words = [lem.lemmatize(word, 'v') for word in words]
    words = [lem.lemmatize(word, 'n') for word in words]
    
    cleaned_text = " ".join(words)
    
    return cleaned_text

# checking if our function works correctly or not
_clean("this is a Test Text for cleaning")
train["cleaned_tweets"] = train["tweet"].apply(_clean)
test["cleaned_tweets"] = test["tweet"].apply(_clean)
train.head()
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

cvz = CountVectorizer()
word_tfidf = TfidfVectorizer()
cvz.fit(train["cleaned_tweets"].values)
count_vector_train = cvz.transform(train["cleaned_tweets"].values)
count_vector_test = cvz.transform(test["cleaned_tweets"].values)

word_tfidf.fit(train["cleaned_tweets"].values)
word_vector_train = word_tfidf.transform(train["cleaned_tweets"].values)
word_vector_test = word_tfidf.transform(test["cleaned_tweets"].values)

train_vector = count_vector_train
test_vector = count_vector_test
target = train['label']
from sklearn.model_selection import train_test_split
trainx, valx, trainy, valy = train_test_split(train_vector,target)
from sklearn import naive_bayes
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn import ensemble
from sklearn.metrics import accuracy_score
model = naive_bayes.MultinomialNB()
model.fit(trainx,trainy)
preds = model.predict(valx)
accuracy_score(preds,valy)
model = svm.SVC()
model.fit(trainx,trainy)
preds = model.predict(valx)
accuracy_score(preds,valy)
model = LogisticRegression()
model.fit(trainx,trainy)
preds = model.predict(valx)
accuracy_score(preds,valy)
model = ensemble.ExtraTreesClassifier()
model.fit(trainx,trainy)
preds = model.predict(valx)
accuracy_score(preds,valy)