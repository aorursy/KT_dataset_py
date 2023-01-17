import pandas as pd
import numpy as np
import nltk
nltk.download('stopwords')
from nltk.stem.porter import *
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from sklearn.preprocessing import StandardScaler
# ML Libraries
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC

# Global Parameters
stop_words = set(stopwords.words('english'))
df = pd.read_csv('train.csv', )
test = pd.read_csv('test.csv')

df.head(30)
df.tail()
df['sentiment'].value_counts()
df.isnull().sum()
df['message'] = df['message'].fillna(0)
#Remove URL
pattern_url = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'
subs_url = r'url-web'
df['message'] = df['message'].replace(to_replace = pattern_url, value = subs_url, regex = True)
#Remove punctuation
df['message']  = df['message'] .str.lower()
def remove_punctuation(message):
    return ''.join([l for l in message if l not in string.punctuation])
df['message'] = df['message'].apply(remove_punctuation)
#Tokenize Data
tokeniser = TreebankWordTokenizer()
df['message'] = df['message'].apply(tokeniser.tokenize)
#Stemming
stemmer = PorterStemmer()
def df_stemmer(words, stemmer):
    return [stemmer.stem(word) for word in words]
df['message'] = df['message'].apply(df_stemmer, args=(stemmer, ))
#Lemmatization
lemmatizer = WordNetLemmatizer()
def df_lemma(words, lemmatizer):
    return [lemmatizer.lemmatize(word) for word in words]    
df['message'] = df['message'].apply(df_lemma, args=(lemmatizer, ))
#Remove Stopwords

def remove_stop_words(tokens):    
    return [t for t in tokens if t not in stopwords.words('english')]
df['message']=df['message'].apply(remove_stop_words)
#Feature Selection Splitting out Variables
y =df['sentiment']
X = df['message']

bow_vect = CountVectorizer(max_features=1000, stop_words='english')
X_vect=bow_vect.fit_transform(df[['message']])

trans = StandardScaler(with_mean=False)
X_vect = trans.fit_transform(X_vect)
#Training Model
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2, random_state=42)
X_vect
#Model training
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_val)
#Test Set

testx = test['message']
test_vect = bow_vect.transform(testx)
#Predications
y_pred = rfc.predict(test_vect)
#Test
test['sentiment'] = y_pred
test.head()
test[['tweetid','sentiment']].to_csv('testsubmission.csv', index=False)
