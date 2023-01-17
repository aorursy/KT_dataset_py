import nltk

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import re



import sklearn

from sklearn import preprocessing

from sklearn.metrics import log_loss

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn import metrics 

from sklearn.metrics import classification_report

from nltk.classify import NaiveBayesClassifier



import os

print(os.listdir("../input"))
mbti = pd.read_csv('../input/train.csv', encoding='utf-8')

mbti_test = pd.read_csv('../input/test.csv', encoding='utf-8')



#mbti = pd.read_csv('train.csv')

#mbti_test = pd.read_csv('test.csv')



type_labels = ['ISTJ', 'ISFJ', 'INFJ', 'INTJ', 

               'ISTP', 'ISFP', 'INFP', 'INTP', 

               'ESTP', 'ESFP', 'ENFP', 'ENTP', 

               'ESTJ', 'ESFJ', 'ENFJ', 'ENTJ']



test_ID = mbti_test['id']
mbti.head(10)
mbti_test.head(10)
mbti.info()
mbti['type'].value_counts().plot(kind = 'bar')

plt.show()
#all_mbti = []

#for i, row in mbti.iterrows():

    #for post in row['posts'].split('|||'):

        #all_mbti.append([row['type'], post])

#all_mbti = pd.DataFrame(all_mbti, columns=['type', 'post'])
#all_mbti_test = []

#for i, row in mbti_test.iterrows():

    #for post in row['posts'].split('|||'):

        #all_mbti_test.append([post])

#all_mbti_test = pd.DataFrame(all_mbti_test, columns=['post'])
mbti_test.head()
mbti.head()
pattern_url = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'

subs_url = r'url-web'

mbti['posts'] = mbti['posts'].replace(to_replace = pattern_url, value = subs_url, regex = True)

mbti_test['posts'] = mbti_test['posts'].replace(to_replace = pattern_url, value = subs_url, regex = True)
mbti_test.head()
mbti.head()
mbti['posts'] = mbti['posts'].str.lower()

mbti_test['posts'] = mbti_test['posts'].str.lower()

mbti.head()
import string

def remove_punctuation_numbers(post):

    punc_numbers = string.punctuation + '0123456789'

    return ''.join([l for l in post if l not in punc_numbers])

mbti['posts'] = mbti['posts'].apply(remove_punctuation_numbers)

mbti_test['posts'] = mbti_test['posts'].apply(remove_punctuation_numbers)

mbti.head()
mbti.head(10)


#all_mbti['I'] = all_mbti['type'].apply(lambda x: x[0] == 'I').astype('int')

#all_mbti['E'] = all_mbti['type'].apply(lambda x: x[0] == 'E').astype('int')

#all_mbti['S'] = all_mbti['type'].apply(lambda x: x[1] == 'S').astype('int')

#all_mbti['N'] = all_mbti['type'].apply(lambda x: x[1] == 'N').astype('int')

#all_mbti['F'] = all_mbti['type'].apply(lambda x: x[2] == 'F').astype('int')

#all_mbti['T'] = all_mbti['type'].apply(lambda x: x[2] == 'T').astype('int')

#all_mbti['J'] = all_mbti['type'].apply(lambda x: x[3] == 'P').astype('int')

#all_mbti['P'] = all_mbti['type'].apply(lambda x: x[3] == 'J').astype('int')





mbti['mind'] = mbti['type'].apply(lambda x: x[0] == 'E').astype('int')

mbti['energy'] = mbti['type'].apply(lambda x: x[1] == 'N').astype('int')

mbti['nature'] = mbti['type'].apply(lambda x: x[2] == 'T').astype('int')

mbti['tactics'] = mbti['type'].apply(lambda x: x[3] == 'J').astype('int')
mbti.info()
from nltk.tokenize import word_tokenize, TreebankWordTokenizer
tokeniser = TreebankWordTokenizer()

mbti['posts'] = mbti['posts'].apply(tokeniser.tokenize)

mbti_test['posts'] = mbti_test['posts'].apply(tokeniser.tokenize)
#Stem

from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")



def mbti_stemma(words, stemmer):

    return [stemmer.stem(word) for word in words]



mbti['posts'] = mbti['posts'].apply(mbti_stemma, args=(stemmer, ))

mbti_test['posts'] = mbti_test['posts'].apply(mbti_stemma, args=(stemmer, ))





#mbti["posts"] = mbti["posts"].apply(lambda x: " ".join(stemmer.stem(p) for p in x.split(" ")))

#mbti_test["posts"] = mbti_test["posts"].apply(lambda x: " ".join(stemmer.stem(p) for p in x.split(" ")))
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
def mbti_lemma(words, lemmatizer):

    return [lemmatizer.lemmatize(word) for word in words]
# lemmatize all words in dataframe

mbti['posts'] = mbti['posts'].apply( mbti_lemma, args=(lemmatizer, ))

mbti['posts'] = [' '.join(line) for line in mbti.posts]



mbti_test['posts'] = mbti_test['posts'].apply( mbti_lemma, args=(lemmatizer, ))

mbti_test['posts'] = [' '.join(line) for line in mbti_test.posts]
mbti.head()
from sklearn.feature_extraction.text import CountVectorizer

#vect = CountVectorizer()

#X_count = vect.fit_transform(mbti['posts'])
#X_count_test = vect.fit_transform(mbti_test['posts'])
#max_feat = X_count_test.shape[1]

#max_feat
vect_10 = CountVectorizer(lowercase=True, stop_words='english', max_features = 500)

X_count = vect_10.fit_transform(mbti['posts'])

X_count_test = vect_10.fit_transform(mbti_test['posts'])
X_count.shape

#mbti_test.drop("id", axis = 1, inplace = True)
X_count_test.shape
row = X_count_test.shape[0]

row
X_count_test.shape
X = X_count

#data.iloc[0:10, :]

X_test = X_count_test

y_mind = mbti['mind'].values

#y_mind_test = y_mind[0:row]



y_energy = mbti['energy'].values

#y_energy_test = y_energy[0:row]



y_nature = mbti['nature'].values

#y_nature_test = y_nature[0:row]



y_tactics = mbti['tactics'].values

#y_tactics_test = y_tactics[0:row]



y_mind.shape
#X_train, X_test, y_train, y_test = train_test_split(X, y_mind, test_size = 0.2, stratify = y_mind, random_state = 42)

#X_train, X_test, y_train2, y_test2 = train_test_split(X, y_energy, test_size = 0.2, stratify = y_energy, random_state = 42)

#X_train, X_test, y_train3, y_test3 = train_test_split(X, y_nature, test_size = 0.2, stratify = y_nature, random_state = 42)

#X_train, X_test, y_train4, y_test4 = train_test_split(X, y_tactics, test_size = 0.2, stratify = y_tactics, random_state = 42)
logreg = LogisticRegression(C = 0.25, penalty='l1')
X_train_mind, X_test_mind, y_train_mind, y_test_mind = train_test_split(X, y_mind)
logreg.fit(X_train_mind, y_train_mind)

y_pred1 = logreg.predict(X_test_mind)

metrics.accuracy_score(y_test_mind, y_pred1)
loglos= log_loss(y_test_mind, y_pred1)

print(loglos) 
logreg2 = LogisticRegression(C = 0.25, penalty='l1')

X_train_energy, X_test_energy, y_train_energy, y_test_energy = train_test_split(X, y_energy)
logreg2.fit(X_train_energy, y_train_energy)

y_pred2 = logreg2.predict(X_test_energy)

metrics.accuracy_score(y_test_energy, y_pred2)
loglos= log_loss(y_test_energy, y_pred2)

print(loglos) 
logreg3 = LogisticRegression(C = 0.25, penalty='l1')

X_train_nature, X_test_nature, y_train_nature, y_test_nature = train_test_split(X, y_nature)
logreg3.fit(X_train_nature, y_train_nature)

y_pred3 = logreg3.predict(X_test_nature)

metrics.accuracy_score(y_test_nature, y_pred3)
loglos= log_loss(y_test_nature, y_pred3)

print(loglos) 
logreg4 = LogisticRegression(C = 0.25, penalty='l1')

X_train_tactics, X_test_tactics, y_train_tactics, y_test_tactics = train_test_split(X, y_tactics)
logreg4.fit(X_train_tactics, y_train_tactics)

y_pred4 = logreg4.predict(X_test_tactics)

metrics.accuracy_score(y_test_tactics, y_pred4)
loglos= log_loss(y_test_tactics, y_pred4)

print(loglos) 
logreg = LogisticRegression(C = 0.25, penalty='l1')

logreg.fit(X, y_mind)

y_pred_mind = logreg.predict(X_count_test)
logreg2 = LogisticRegression(C = 0.25, penalty='l1')

logreg2.fit(X, y_energy)

y_pred_energy = logreg2.predict(X_count_test)
logreg3 = LogisticRegression(C = 0.25, penalty='l1')

logreg3.fit(X, y_nature)

y_pred_nature = logreg3.predict(X_count_test)
logreg4 = LogisticRegression(C = 0.25, penalty='l1')

logreg4.fit(X, y_tactics)

y_pred_tactics = logreg4.predict(X_count_test)
#submission



sub = pd.DataFrame({'Id': test_ID, 'mind': y_pred_mind, 'energy':y_pred_energy, 'nature': y_pred_nature, 'tactics':y_pred_tactics})
sub.to_csv('submission2.csv', index =False)







