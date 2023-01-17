
from sklearn import preprocessing,metrics
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer,CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix
import pandas as pd, numpy as np
import re,nltk,textblob,string
import matplotlib.pyplot as plt

from string import punctuation


df = pd.read_excel("C:/Users/DAVE/Documents/Notebooks/NLP/Assignment 15/training-ReviewsFileName/ReviewsFileName.xlsx")


def cleanData(data):
    data['Review_cleaned'] = data['Review'].str.replace('[^a-zA-Z]+', ' ', regex = True) 
    data['Review_cleaned'] = data['Review_cleaned'].str.replace('((www\.[\s]+)|(https?://[^\s]+))','',regex=True)
    
    data["Review_cleaned"] = data["Review_cleaned"].str.lower()
    data["Review_cleaned"] = data["Review_cleaned"].str.split()
    stop = stopwords.words('english')
    data['Review_cleaned']=data['Review_cleaned'].apply(lambda x:[item for item in x if item not in stop])
    
    ps = PorterStemmer()
    data['Review_cleaned'] = data['Review_cleaned'].apply(lambda x:[ps.stem(y) for y in x])  
 
   
    

cleanData(df)

from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize, sent_tokenize
from heapq import nlargest


w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in (text)]

df['Review_cleaned'] = df['Review_cleaned'].apply(lemmatize_text)

def rejoin_words(row):
    my_list = row['Review_cleaned']
    joined_words = (" ".join(my_list))
    return joined_words

df['Review_cleaned'] = df.apply(rejoin_words, axis=1)

txt = df['Review_cleaned'].str.lower().str.cat(sep=' ')
words = nltk.tokenize.word_tokenize(txt)
word_dist = nltk.FreqDist(words)

output = word_dist.most_common(30)

tfid = TfidfVectorizer()

vect = tfid.fit_transform(df['Review_cleaned'])

from sklearn.model_selection import train_test_split

count_vect = CountVectorizer(lowercase=True, stop_words='english')
X_count_vect = count_vect.fit_transform(df['Review_cleaned'])
X_names= count_vect.get_feature_names()
X = pd.DataFrame(X_count_vect.toarray(), columns=X_names)



y = df['Sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
# predictions
rfc_predict = rfc.predict(X_test)

score_clf_cv = confusion_matrix(y_test,rfc_predict)

score_clf_cv