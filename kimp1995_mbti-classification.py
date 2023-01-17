import nltk

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from collections import Counter

from nltk.corpus import stopwords

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer

from sklearn.metrics import log_loss, confusion_matrix, classification_report, roc_auc_score, accuracy_score





import seaborn as sns

%matplotlib inline

import re
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
#Now converting to numeric numbers 



train['mind'] = train['mind'].apply(lambda x: 0 if x == 'I' else 1)

train['energy'] = train['energy'].apply(lambda x: 0 if x == 'S' else 1)

train['nature'] = train['nature'].apply(lambda x: 0 if x == 'F' else 1)

train['tactics'] = train['tactics'].apply(lambda x: 0 if x == 'P' else 1)
train.head()
N = 4

but = (train['mind'].value_counts()[1], train['energy'].value_counts()[0], train['nature'].value_counts()[0], train['tactics'].value_counts()[0])

top = (train['mind'].value_counts()[0], train['energy'].value_counts()[1], train['nature'].value_counts()[1], train['tactics'].value_counts()[1])



ind = np.arange(N)    # the x locations for the groups

width = 0.7      # the width of the bars: can also be len(x) sequence



p1 = plt.bar(ind, but, width)

p2 = plt.bar(ind, top, width, bottom=but)



plt.ylabel('Count')

plt.title('Distribution accoss types indicators')

plt.xticks(ind, ('I/E',  'N/S', 'T/F', 'J /P',))



plt.show()
combined =  pd.concat([train[['posts']].copy(), test[['posts']].copy()], axis=0)
combined.info
#Function that preprocess text using Spacy

import spacy

from spacy.lang.en.stop_words import STOP_WORDS



#loading the en_core_web_sm_model

stopwords = STOP_WORDS

nlp = spacy.load('en_core_web_sm')





def preprocess(train):

    #creating a Doc object

    doc = nlp(train, disable = ['ner', 'parser'])

    #Generating lemmas

    lemmas = [token.lemma_ for token in doc]

    #remove stopwords and non-alphabetic characters

    a_lemma = [lemma for lemma in lemmas

              if lemma.isalpha() and lemma not in stopwords ]

    return ' ' .join(a_lemma)



#apply preprocessing to posts

combined['clean_posts']= combined['posts'].apply(preprocess)
combined.head()
all_data= combined.copy()
all_data.head()
from nltk.tokenize import TweetTokenizer

from nltk.stem import WordNetLemmatizer

tweettoken = TweetTokenizer()

wordnet = WordNetLemmatizer()



all_data['final_posts'] = all_data.apply(lambda row: [wordnet.lemmatize(w) for w in tweettoken.tokenize(row['clean_posts'])], axis=1)
all_data.head()
all_data['final_posts']=[''.join(post) for post in all_data['clean_posts']]
all_data.head()
tfidf_vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1,2))
counts = tfidf_vectorizer.fit_transform(all_data['final_posts'].values)



train_new = counts[:len(train), :]

test_new = counts[len(train):, :]
# Train test split 

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_new, y, test_size=0.2)

from sklearn.linear_model import LogisticRegression



logistic = LogisticRegression()

logistic.fit(X_train, y_train)



y_pred_log = logistic.predict(X_test)

y_pred_log1 = logistic.predict_proba(X_test)



from sklearn.metrics import classification_report, log_loss



print('Logistic Regression Report')

print( classification_report(y_test,y_pred_log))

print('Log Loss:', log_loss(y_test,y_pred_log1))

from sklearn.naive_bayes import MultinomialNB



nb = MultinomialNB()

nb.fit(X_train, y_train)



y_pred_nb = nb.predict(X_test)

y_pred_nb1 = nb.predict_proba(X_test)





from sklearn.metrics import classification_report, log_loss



print('Naive Bayes')

print( classification_report(y_test,y_pred_nb))

print('Log Loss:', log_loss(y_test,y_pred_nb1))

from sklearn.ensemble import RandomForestClassifier 

randomforest = RandomForestClassifier()

randomforest.fit(X_train, y_train)

y_pred_rf = randomforest.predict(X_test)

y_pred_rf1 = randomforest.predict_proba(X_test)

print('Random Forest')

print( classification_report(y_test,y_pred_rf))

print('Log Loss:', log_loss(y_test,y_pred_rf1))
from sklearn.svm import SVC



svc = SVC(kernel ='linear')

svc.fit(X_train, y_train)



y_pred_svc = svc.predict(X_test)

#y_pred_svc1 = svc.predict_proba(X_test)



from sklearn.metrics import classification_report, log_loss



print('Support Vector Classifier:')

print('\n\nClassification Report:', classification_report(y_test,y_pred_svc))

print('Log Loss:', log_loss(y_test,y_pred_svc))





def tune_model(X_train, y_train): 

    

    C_list = [0.001,0.01,0.1, 1,2,5,10]

    

    logistic = LogisticRegression()    

    parameters = {'C':(C_list)}

    

    clf = GridSearchCV(log, parameters)

    best = clf.fit(X_train,y_train)

    return best.best_params_



tune_model(X_train, y_train)
#Logistic Regression with hyperparameters

from sklearn.linear_model import LogisticRegression



logistic = LogisticRegression(C=10, class_weight='balanced',solver='newton-cg', multi_class='multinomial',penalty='l2')

logistic.fit(X_train, y_train)



y_pred = logistic.predict(X_test)

y_pred1 = logistic.predict_proba(X_test)



from sklearn.metrics import classification_report, log_loss



print('Logisitc Regression with hyperparameters:')

print('\n\nClass Report:', classification_report(y_test,y_pred))

print('Log Loss:', log_loss(y_test,y_pred1))
y_multinomial = nb.predict(test_new)

y_logistic = logistic.predict(test_new)

y_randomforest = randomforest.predict(test_new)

y_svc = svc.predict(test_new)

y_knn =  classifier.predict(test_new)

dtc = clf.predict(test_new)

y_ada = ada.predict(test_new)



submission_multinomial = pd.DataFrame()

submission_logistic = pd.DataFrame()

submission_randomforest = pd.DataFrame()

submission_svc = pd.DataFrame()

submission_knn = pd.DataFrame()

submission_dtc = pd.DataFrame()

submission_ada = pd.DataFrame()







submission_multinomial['id'] = test_id

submission_multinomial['type'] = y_multinomial



submission_logistic['id'] = test_id

submission_logistic['type'] = y_logistic



submission_randomforest['id'] = test_id

submission_randomforest['type'] = y_randomforest



submission_svc['id'] = test_id

submission_svc['type'] = y_svc



submission_knn['id'] = test_id

submission_knn['type'] = y_knn



submission_dtc['id'] = test_id

submission_dtc['type'] = dtc

submission_ada['id'] = test_id

submission_ada['type'] = y_ada





print(submission_knn['type'].value_counts())

print(submission_svc['type'].value_counts())
# Encoding the different type to each category



submission_multinomial['mind'] = submission_multinomial['type'].apply(lambda X: 0 if X[0] == 'I' else 1)

submission_multinomial['energy'] = submission_multinomial['type'].apply(lambda X: 0 if X[1] == 'S' else 1)

submission_multinomial['nature'] = submission_multinomial['type'].apply(lambda X: 0 if X[2] == 'F' else 1)

submission_multinomial['tactics'] = submission_multinomial['type'].apply(lambda X: 0 if X[3] == 'P' else 1)





submission_logistic['mind'] = submission_logistic['type'].apply(lambda X: 0 if X[0] == 'I' else 1)

submission_logistic['energy'] = submission_logistic['type'].apply(lambda X: 0 if X[1] == 'S' else 1)

submission_logistic['nature'] = submission_logistic['type'].apply(lambda X: 0 if X[2] == 'F' else 1)

submission_logistic['tactics'] = submission_logistic['type'].apply(lambda X: 0 if X[3] == 'P' else 1)





submission_randomforest['mind'] = submission_randomforest['type'].apply(lambda X: 0 if X[0] == 'I' else 1)

submission_randomforest['energy'] = submission_randomforest['type'].apply(lambda X: 0 if X[1] == 'S' else 1)

submission_randomforest['nature'] = submission_randomforest['type'].apply(lambda X: 0 if X[2] == 'F' else 1)

submission_randomforest['tactics'] = submission_randomforest['type'].apply(lambda X: 0 if X[3] == 'P' else 1)





submission_svc['mind'] = submission_svc['type'].apply(lambda X: 0 if X[0] == 'I' else 1)

submission_svc['energy'] = submission_svc['type'].apply(lambda X: 0 if X[1] == 'S' else 1)

submission_svc['nature'] = submission_svc['type'].apply(lambda X: 0 if X[2] == 'F' else 1)

submission_svc['tactics'] = submission_svc['type'].apply(lambda X: 0 if X[3] == 'P' else 1)





submission_knn['mind'] = submission_knn['type'].apply(lambda X: 0 if X[0] == 'I' else 1)

submission_knn['energy'] = submission_knn['type'].apply(lambda X: 0 if X[1] == 'S' else 1)

submission_knn['nature'] = submission_knn['type'].apply(lambda X: 0 if X[2] == 'F' else 1)

submission_knn['tactics'] = submission_knn['type'].apply(lambda X: 0 if X[3] == 'P' else 1)



submission_dtc['mind'] = submission_dtc['type'].apply(lambda X: 0 if X[0] == 'I' else 1)

submission_dtc['energy'] = submission_dtc['type'].apply(lambda X: 0 if X[1] == 'S' else 1)

submission_dtc['nature'] = submission_dtc['type'].apply(lambda X: 0 if X[2] == 'F' else 1)

submission_dtc['tactics'] = submission_dtc['type'].apply(lambda X: 0 if X[3] == 'P' else 1)



submission_ada['mind'] = submission_ada['type'].apply(lambda X: 0 if X[0] == 'I' else 1)

submission_ada['energy'] = submission_ada['type'].apply(lambda X: 0 if X[1] == 'S' else 1)

submission_ada['nature'] = submission_ada['type'].apply(lambda X: 0 if X[2] == 'F' else 1)

submission_ada['tactics'] = submission_ada['type'].apply(lambda X: 0 if X[3] == 'P' else 1)



#Preparing the dataset for submission



submission_file_multinomial = submission_multinomial.drop('type', axis =1)



submission_file_logistic = submission_logistic.drop('type', axis =1)



submission_file_randomforest = submission_randomforest.drop('type', axis =1)



submission_file_svc = submission_svc.drop('type', axis =1)



submission_file_knn = submission_knn.drop('type', axis =1)

submission_file_dtc = submission_dtc.drop('type', axis =1)



submission_file_ada = submission_ada.drop('type', axis =1)











#Creating a csv for each model 



submission_file_multinomial.to_csv('submission_multinomial.csv', index = False)

submission_file_logistic.to_csv('submission_logistic.csv', index = False)



submission_file_randomforest.to_csv('submission_randomforest.csv', index = False)



submission_file_svc.to_csv('submission_svc.csv', index = False)



submission_file_knn.to_csv('submission_knn.csv', index = False)



submission_file_dtc.to_csv('submission_dtc.csv', index = False)



submission_file_ada.to_csv('submission_ada.csv', index = False)








