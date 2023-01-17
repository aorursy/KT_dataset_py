# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import inflection as infl

import re
import nltk
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn import model_selection

from sklearn.svm import LinearSVC

# Any results you write to the current directory are saved as output.
def custom_fixes(word):
    
    #large, all, purpose, non, fra, che, for, added, ready, not, the, 
    #and, fine, five, leav, ton, sec, msg, min, sum, tel
    dropped_words = ["all", "purpose", "the", "large", "fra", "che", "for", "added", "ready", 
                     "the", "and", "five", "ton", "sec", "msg", "min", "sum", "tel"]
    
    if word == "chilli" or word == "chily" or word == "chile":
        return "chili"
    if word == "leafe" or word == "leav":
        return "leaf"
    if word == "olife":
        return "olive"
    if word == "clofe":
        return "clove"
    
    if word in dropped_words or len(word) < 3:
        return ""
    
    return word
train_data = pd.read_json("../input/train.json")
test_data = pd.read_json("../input/test.json")
infl.singularize("olives")
train_data['another_clean_string'] = [ ' '.join(
    [ WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', ingredient)) for ingredient in recipe ]
).strip().lower() for recipe in train_data['ingredients'] ]


train_data['clean_string'] = [ ' '.join(
    [ custom_fixes(infl.singularize(word)) for word in string.split(" ")]
) for string in train_data['another_clean_string']]

train_data.head(25)
test_data['another_clean_string'] = [ ' '.join(
    [ WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', ingredient)) for ingredient in recipe ]
).strip().lower() for recipe in test_data['ingredients'] ]



test_data['clean_string'] = [ ' '.join(
    [ custom_fixes(infl.singularize(word)) for word in string.split(" ")]
) for string in test_data['another_clean_string']]

test_data.head(25)
testing = train_data['clean_string'].values
testing

array = []

for recipe in testing:
    recipe_ingredients = recipe.split(" ")
    for recipe_ingredient in recipe_ingredients:
        array.append(recipe_ingredient)
        
df = pd.DataFrame(array)
values = df[0].value_counts()
values

print(values)
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt
% matplotlib inline
string = [' '.join(word for word in df[0])]
#print(string)
wordcloud = WordCloud().generate(string[0])

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
wordcloud = WordCloud(width=3000, height=1200, max_words=100, background_color="white").generate(string[0])
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

wordcloud.to_file("words.png")
train_data_corpus = train_data['clean_string']
train_data_vectorizer = TfidfVectorizer(stop_words='english',
                             ngram_range = ( 1 , 1 ),analyzer="word", 
                             max_df = .57 , binary=False , token_pattern=r'(\w+?)(?:,\s|\s|$)' , sublinear_tf=False)

#train_data_vectorizer = TfidfVectorizer(stop_words='english',
#                             ngram_range = ( 1 , 1 ),analyzer="word", 
#                             max_df = .57 , binary=False , token_pattern=r'\w+' , sublinear_tf=False)

#train_data_vectorizer = TfidfVectorizer(stop_words='english',
#                             ngram_range = ( 1 , 1 ),analyzer="word", 
#                             max_df = .50 , binary=False , token_pattern=r'\w+' , sublinear_tf=False)

#train_data_vectorizer = TfidfVectorizer(stop_words='english',
#                             ngram_range = ( 1 , 1 ),analyzer="word", 
#                             max_df = .67 , binary=False , token_pattern=r'\w+' , sublinear_tf=False)

# This is only .25% accurate
#train_data_vectorizer = TfidfVectorizer(ngram_range = ( 1 , 1 ),analyzer="char", 
#                             max_df = .57 )

train_data_tfidf = train_data_vectorizer.fit_transform(train_data_corpus).todense()

#TFIDF stands for term frequency- inverse document frequency.
#The TFIDF weight is used in text mining and IR. The weight is a measure used to evaluate how important a word is to a document in a collection of documents.
test_data_corpus = test_data['clean_string']
test_data_vectorizer = TfidfVectorizer(stop_words='english')
test_data_tfidf = train_data_vectorizer.transform(test_data_corpus)
train_data_predictors = train_data_tfidf
train_data_targets = train_data['cuisine']
test_data_predictors = test_data_tfidf
# Logistic Regression
#https://medium.com/@aneesha/svm-parameter-tuning-in-scikit-learn-using-gridsearchcv-2413c02125a0

parameters = {'C':[1, 10]}
clf = LogisticRegression()
classifier = model_selection.GridSearchCV(clf, parameters)

classifier = classifier.fit(train_data_predictors,train_data_targets)
# Linear SVC

clf = LinearSVC()
classifier = LinearSVC(C=0.90, penalty="l2", dual=False)
classifier = classifier.fit(train_data_predictors,train_data_targets)
predictions = classifier.predict(test_data_predictors)
test_data['cuisine'] = predictions
print(test_data[[ 'id', 'cuisine' ]].head(10))
test_data[[ 'id', 'cuisine' ]].to_csv("testsubmission.csv", index=False)