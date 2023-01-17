#Importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from string import punctuation
from nltk.corpus import stopwords, wordnet
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from wordcloud import WordCloud
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression,SGDClassifier, LinearRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

df = pd.read_csv('C:/Users/PUKHRAJ/Desktop/INTERNSHIPS/Datasets/Corona_NLP_train.csv',encoding='latin-1')
df.head()
df.info()
df.isna().sum()
df1 = df.iloc[:,4:]
df1
df.Sentiment.unique() 
# making list stopwords for removing stopwords from our text 

stop = set(stopwords.words('english'))
stop.update(punctuation)
print(stop)
# this function return the part of speech of a word.
def get_simple_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
# Function to clean our text.
lemmatizer = WordNetLemmatizer()
def clean_review(OriginalTweet):
    clean_text = []
    for w in word_tokenize(OriginalTweet):
        if w.lower() not in stop:
            pos = pos_tag([w])
            new_w = lemmatizer.lemmatize(w, pos=get_simple_pos(pos[0][1]))
            clean_text.append(new_w)
    return clean_text

def join_text(OriginalTweet):
    return " ".join(OriginalTweet)
df1.OriginalTweet = df1.OriginalTweet.apply(clean_review)
df1.OriginalTweet = df1.OriginalTweet.apply(join_text)
df1.head()
# splitting data.
x_train,x_test,y_train,y_test = train_test_split(df1.OriginalTweet,df1.Sentiment,test_size = 0.3 , random_state = 0)
x_train.shape, y_train.shape, x_test.shape, y_test.shape
pos = x_train[y_train[y_train=='Positive'].index]
ex_pos = x_train[y_train[y_train=='Extremely Positive'].index]
neg = x_train[y_train[y_train=='Negative'].index]
ex_neg = x_train[y_train[y_train=='Extremely Negative'].index]
neutral = x_train[y_train[y_train=='Neutral'].index]
plt.figure(figsize = (18,24)) # Text Reviews with positive Ratings
wordcloud = WordCloud(min_font_size = 3,  max_words = 2500 , width = 1200 , height = 800).generate(" ".join(pos))
plt.imshow(wordcloud,interpolation = 'bilinear')
plt.figure(figsize = (18,24)) # Text Reviews with extreame positive Ratings
wordcloud = WordCloud(min_font_size = 3,  max_words = 2500 , width = 1200 , height = 800).generate(" ".join(ex_pos))
plt.imshow(wordcloud,interpolation = 'bilinear')
plt.figure(figsize = (18,24)) # Text Reviews with negative Ratings
wordcloud = WordCloud(min_font_size = 3,  max_words = 2500 , width = 1200 , height = 800).generate(" ".join(neg))
plt.imshow(wordcloud,interpolation = 'bilinear')
plt.figure(figsize = (18,24)) # Text Reviews with Extreame negative Ratings
wordcloud = WordCloud(min_font_size = 3,  max_words = 2500 , width = 1200 , height = 800).generate(" ".join(ex_neg))
plt.imshow(wordcloud,interpolation = 'bilinear')
plt.figure(figsize = (18,24)) # Text Reviews with neutral Ratings
wordcloud = WordCloud(min_font_size = 3,  max_words = 2500 , width = 1200 , height = 800).generate(" ".join(neutral))
plt.imshow(wordcloud,interpolation = 'bilinear')
# creating a variable for count vectorizer which gives us features using the whole text of data.
count_vec = CountVectorizer(max_features=4000, ngram_range=(1,2), max_df=0.9, min_df=0)
# max_df insures to remove most frequent words as we discussed earlier.
# ngram_range is used to select words at a time like 1 or 2 like if a sentence have 'not happy' in text then it can mean two things if we pick the word 'happy' and pick the words 'not happy' both.
x_train_features = count_vec.fit_transform(x_train).todense()
x_test_features = count_vec.transform(x_test).todense()
x_train_features.shape, x_test_features.shape
lr = LogisticRegression()
lr.fit(x_train_features, y_train)
y_pred = lr.predict(x_test_features)
print(accuracy_score(y_test,y_pred)*100)
print(classification_report(y_test, y_pred))
nb_clf = MultinomialNB()
nb_clf.fit(x_train_features, y_train)
y_pred = nb_clf.predict(x_test_features)
print(accuracy_score(y_test,y_pred)*100)
print(classification_report(y_test, y_pred))

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100)
model = model.fit(x_train_features,y_train)
from sklearn.metrics import confusion_matrix
from sklearn import metrics

y_pred = model.predict(x_test_features)

cm = confusion_matrix(y_test,y_pred)
print("Classification Report:");print(metrics.classification_report(y_test, y_pred))
