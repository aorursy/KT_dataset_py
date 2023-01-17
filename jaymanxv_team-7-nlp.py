import nltk

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import re

import warnings

warnings.filterwarnings('ignore')



from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer

from sklearn.model_selection import train_test_split, GridSearchCV



from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC



from sklearn.metrics import log_loss, classification_report
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
test.head()
train.info()
plt.figure(figsize=(40,20))

plt.xticks(fontsize=24, rotation=0)

plt.yticks(fontsize=24, rotation=0)

sns.countplot(data=train, x='type');
y = train['type']

test_id = test['id']
train['mind'] = train['type'].map(lambda x: 'I' if x[0] == 'I' else 'E')

train['energy'] = train['type'].map(lambda x: 'N' if x[1] == 'N' else 'S')

train['nature'] = train['type'].map(lambda x: 'T' if x[2] == 'T' else 'F')

train['tactics'] = train['type'].map(lambda x: 'J' if x[3] == 'J' else 'P')
#Now convert the alpha chars to numeric numbers 



train['mind'] = train['mind'].apply(lambda x: 0 if x == 'I' else 1)

train['energy'] = train['energy'].apply(lambda x: 0 if x == 'S' else 1)

train['nature'] = train['nature'].apply(lambda x: 0 if x == 'F' else 1)

train['tactics'] = train['tactics'].apply(lambda x: 0 if x == 'P' else 1)
train.head()
N = 4

but = (train['mind'].value_counts()[1], train['energy'].value_counts()[0],\

       train['nature'].value_counts()[0], train['tactics'].value_counts()[0])



top = (train['mind'].value_counts()[0], train['energy'].value_counts()[1],\

       train['nature'].value_counts()[1], train['tactics'].value_counts()[1])



ind = np.arange(N)    # the x locations for the groups

width = 0.7      # the width of the bars: can also be len(x) sequence



p1 = plt.bar(ind, but, width)

p2 = plt.bar(ind, top, width, bottom=but)



plt.ylabel('Count')

plt.title('Distribution accoss types indicators')

plt.xticks(ind, ('I/E',  'N/S', 'T/F', 'J/P',))



plt.show()
combined = pd.concat([train[['posts']].copy(), test[['posts']].copy()], axis=0)
combined.head()
#Function that preprocess text using Spacy

import spacy

import en_core_web_sm



from spacy.lang.en.stop_words import STOP_WORDS



#loading the en_core_web_sm_model

stopwords = STOP_WORDS

nlp = en_core_web_sm.load()

#nlp = spacy.load('en_core_web_sm')



def preprocess(train):

    #creating a Doc object

    doc = nlp(train, disable = ['ner', 'parser'])

    #Generating lemmas

    lemmas = [token.lemma_ for token in doc]

    #remove stopwords and non-alphabetic characters

    a_lemma = [lemma for lemma in lemmas

              if lemma.isalpha() and lemma not in stopwords]

    return ' ' .join(a_lemma)#.lower()



#apply preprocessing to posts

combined['clean_posts'] = combined['posts'].apply(preprocess)
combined.head()
#Made a copy

all_data = combined.copy()
all_data.head()
from nltk.tokenize import TweetTokenizer

from nltk.stem import WordNetLemmatizer

tweettoken = TweetTokenizer()

wordnet = WordNetLemmatizer()



all_data['final_posts'] = all_data.apply(lambda row: [wordnet.lemmatize(w) for w in tweettoken.tokenize(row['clean_posts'])], axis=1)
all_data.head()
all_data['final_posts']=[''.join(post) for post in all_data['clean_posts']]
all_data.head()
tfidf_vectorizer = TfidfVectorizer(min_df=3, ngram_range=(1,2))
counts = tfidf_vectorizer.fit_transform(all_data['final_posts'].values)



train_new = counts[:len(train), :]

test_new = counts[len(train):, :]
# Train test split 

X_train, X_test, y_train, y_test = train_test_split(train_new, y, test_size=0.2)
logistic = LogisticRegression()

logistic.fit(X_train, y_train)



y_pred_log = logistic.predict(X_test)

y_pred_log1 = logistic.predict_proba(X_test)



print('Logistic Regression:')

print('\n\nClassification Report:\n', classification_report(y_test,y_pred_log))

print('Log Loss:', log_loss(y_test,y_pred_log1))
nb = MultinomialNB()

nb.fit(X_train, y_train)



y_pred_nb = nb.predict(X_test)

y_pred_nb1 = nb.predict_proba(X_test)



print('Naive Bayes:')

print('\n\nClassification Report:\n', classification_report(y_test,y_pred_nb))

print('Log Loss:', log_loss(y_test,y_pred_nb1))
rf = RandomForestClassifier()

rf.fit(X_train, y_train)



y_pred_rf = rf.predict(X_test)

y_pred_rf1 = rf.predict_proba(X_test)



print('Random Forest:')

print('\n\nClassification Report:\n', classification_report(y_test,y_pred_rf))

print('Log Loss:', log_loss(y_test,y_pred_rf1))
svc = SVC(kernel = 'linear', probability=True)

svc.fit(X_train, y_train)



y_pred_svc = svc.predict(X_test)

y_pred_svc1 = svc.predict_proba(X_test)



print('Support Vector Classifier:')

print('\n\nClassification Report:', classification_report(y_test,y_pred_svc))

print('Log Loss:', log_loss(y_test,y_pred_svc))
'''

def tune_model(X_train, y_train): 

    

    C_list = [0.001,0.01,0.1, 1,2,5,10]

    

    log = LogisticRegression()

    parameters = {'C':(C_list)}

    

    clf = GridSearchCV(log, parameters)

    best = clf.fit(X_train,y_train)

    return best.best_params_



tune_model(X_train, y_train)

'''
logistic_hp = LogisticRegression(C=10, class_weight='balanced',solver='newton-cg', multi_class='multinomial',penalty='l2')

logistic_hp.fit(X_train, y_train)



y_pred_log_hp = logistic_hp.predict(X_test)

y_pred_log_hp1 = logistic_hp.predict_proba(X_test)



print('Logisitc Regression with hyperparameters:')

print('\n\nClass Report:', classification_report(y_test,y_pred_log_hp))

print('Log Loss:', log_loss(y_test,y_pred_log_hp1))
y_logistic = logistic_hp.predict(test_new)

submission_log_hp = pd.DataFrame()

submission_log_hp['id'] = test_id

submission_log_hp['type'] = y_logistic
submission_log_hp['mind'] = submission_log_hp['type'].apply(lambda X: 0 if X[0] == 'I' else 1)

submission_log_hp['energy'] = submission_log_hp['type'].apply(lambda X: 0 if X[1] == 'S' else 1)

submission_log_hp['nature'] = submission_log_hp['type'].apply(lambda X: 0 if X[2] == 'F' else 1)

submission_log_hp['tactics'] = submission_log_hp['type'].apply(lambda X: 0 if X[3] == 'P' else 1)
submission_file_log_hp = submission_log_hp.drop('type', axis =1)
submission_file_log_hp.to_csv('final_submission.csv', index = False)
df = submission_log_hp.groupby('type').count().sort_values(by='id',ascending=False)

df = df.id

df.plot(kind='bar',figsize=(12,8),legend=False,title="Personality Predictions");