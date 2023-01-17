import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
card_data = pd.read_csv( '../input/card_data.csv')
print( card_data.head() )
print( "Card Data Shape: " + str(card_data.shape))
flavor_data = pd.read_csv( '../input/flavor_data.csv', encoding = 'unicode_escape')
print( flavor_data.head() )
print ( "Flavor Data Shape: " + str(flavor_data.shape))
card_data.sum()
trunc_data = card_data.drop(['Name','is_a','is_l'], axis=1)
trunc_data.head()
trunc_data['gold'] = trunc_data[['is_w','is_u','is_b','is_r','is_g']].sum(axis=1)
trunc_data.head()
trunc_data.gold.value_counts()
one_color = trunc_data[trunc_data.gold == 1]
one_color.head()
one_color.sum()
Y = one_color[['is_w','is_u','is_b','is_r','is_g']].idxmax( axis=1)
X = flavor_data[ trunc_data.gold == 1]['Flavor']
print( X.shape )
print( Y.shape)
Y.head()
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.33, random_state=25)
count_vectorizer = CountVectorizer( stop_words="english")
count_train = count_vectorizer.fit_transform( X_train )
count_test = count_vectorizer.transform( X_test )
print( count_vectorizer.get_feature_names()[:10])
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
nb_classifier = MultinomialNB()
nb_classifier.fit( count_train, Y_train)
Y_pred = nb_classifier.predict( count_test )
ac = accuracy_score( Y_test, Y_pred )
print( ac )
cm = confusion_matrix( Y_test, Y_pred, labels=['is_w','is_u','is_b','is_r','is_g'])
print( cm )
from sklearn.feature_extraction.text import TfidfVectorizer

X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.33, random_state=25)
tfidf_vectorizer = TfidfVectorizer( stop_words="english", max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform( X_train )
tfidf_test = tfidf_vectorizer.transform( X_test )
print( tfidf_vectorizer.get_feature_names()[:10])
nb_classifier = MultinomialNB()
nb_classifier.fit( tfidf_train, Y_train)
Y_pred = nb_classifier.predict( tfidf_test )
ac = accuracy_score( Y_test, Y_pred )
print( ac )
cm = confusion_matrix( Y_test, Y_pred, labels=['is_w','is_u','is_b','is_r','is_g'])
print( cm )
